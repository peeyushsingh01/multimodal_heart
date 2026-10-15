import os
import glob
import logging
import tempfile
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import pyvista as pv

try:
    import nest_asyncio2
    nest_asyncio2.apply()
except ImportError:
    pass

# Off-screen rendering for server deployment
pv.OFF_SCREEN = True
pv.set_jupyter_backend('static')

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from authlib.integrations.flask_client import OAuth
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("cardiac")

# Load Predict Module
try:
    from predict import predict_patient
    PREDICT_MODULE_LOADED = True
    MODULE_ERROR = None
except Exception as e:
    predict_patient = None
    PREDICT_MODULE_LOADED = False
    MODULE_ERROR = str(e)
    log.error("predict module import failed: %s", MODULE_ERROR)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-heart-check")

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("INSECURE_COOKIES") != "1",
    MAX_CONTENT_LENGTH=256 * 1024,
)

DEBUG = os.environ.get("FLASK_DEBUG") == "1"
REQUIRE_LOGIN = os.environ.get("REQUIRE_LOGIN", "1") == "1"
MAX_BATCH = 50

if os.environ.get("OAUTH_INSECURE_LOCAL") == "1":
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

db_url = os.environ.get("DATABASE_URL", "sqlite:///" + os.path.join(BASE_DIR, "cardiac_app.db"))
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}

db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=True)
    provider = db.Column(db.String(20), default="local")
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    assessments = db.relationship("Assessment", backref="user", lazy=True, cascade="all, delete-orphan")

class Assessment(db.Model):
    __tablename__ = "assessments"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    timestamp = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    input_data = db.Column(db.JSON, nullable=False)
    result_data = db.Column(db.JSON, nullable=False)

with app.app_context():
    try:
        db.create_all()
    except Exception as ex:
        log.warning("Database auto-creation skipped or failed: %s", ex)

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
oauth = OAuth(app)
google = None
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    google = oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

FEATURE_ORDER = [
    "Age", "Weight", "Length", "Sex", "BMI", "DM", "HTN", "Current Smoker", "EX-Smoker",
    "FH", "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP",
    "BP", "PR", "Edema", "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur",
    "Diastolic Murmur", "Typical Chest Pain", "Dyspnea", "Function Class", "Atypical",
    "Nonanginal", "Exertional CP", "LowTH Ang", "Q Wave", "St Elevation", "St Depression",
    "Tinversion", "LVH", "Poor R Progression", "BBB", "FBS", "CR", "TG", "LDL", "HDL",
    "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT", "EF-TTE", "Region RWMA", "VHD"
]

_NORMALISED = {c.lower().replace(" ", "").replace("_", "").replace("-", ""): c for c in FEATURE_ORDER}
VESSELS = ("LAD", "LCX", "RCA")
POSITIVE_TERMS = {"cad", "disease", "positive", "yes", "stenosis"}

def _norm(col: str) -> str:
    return str(col).lower().replace(" ", "").replace("_", "").replace("-", "")

def build_input_frame(rows: List[Dict[str, Any]]) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    df = pd.DataFrame(rows)
    df = df.rename(columns={c: _NORMALISED.get(_norm(c), c) for c in df.columns})
    df = df.replace(r"^\s*$", np.nan, regex=True)

    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)

    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    extra = [c for c in df.columns if c not in FEATURE_ORDER]
    if missing:
        return None, {"error": "Missing features", "missing": missing, "unexpected": extra}

    df = df[FEATURE_ORDER].copy()
    for c in df.columns:
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.isna().sum() == df[c].isna().sum():
            df[c] = converted

    return df, None

def clean_json(val: Any) -> Any:
    if isinstance(val, dict):
        return {k: clean_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [clean_json(v) for v in val]
    if isinstance(val, np.generic):
        val = val.item()
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val

def get_current_user() -> Optional[User]:
    email = session.get("user_email")
    if not email:
        return None
    return User.query.filter_by(email=email).first()

def login_required_check():
    if REQUIRE_LOGIN and not get_current_user():
        return jsonify({"error": "Authentication required to perform this action."}), 401
    return None

def _is_positive(raw: Any) -> int:
    s = str(raw).strip().lower()
    try:
        return 1 if float(s) == 1.0 else 0
    except ValueError:
        return 1 if s in POSITIVE_TERMS else 0

def _safe_prob(raw: Any) -> float:
    try:
        p = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if (np.isnan(p) or np.isinf(p)) else p

def process_single_patient_prediction(patient_data_dict: Dict[str, Any]) -> Dict[str, Any]:
    raw_results = predict_patient(patient_data_dict)
    formatted_results = {}
    vessels_positive = []
    vessel_probs = []

    for target_name, res_dict in raw_results.items():
        key_upper = target_name.upper()
        raw_prob = _safe_prob(res_dict.get("probability", 0.0))
        is_positive = _is_positive(res_dict.get("prediction"))

        formatted_results[key_upper] = {
            "prediction": is_positive,
            "probability": round(raw_prob, 4),
            "raw_label": clean_json(res_dict.get("prediction")),
        }

        if key_upper in VESSELS:
            if is_positive == 1:
                vessels_positive.append(key_upper)
            vessel_probs.append(raw_prob)

    cath_res = formatted_results.get("CATH", {})
    if vessels_positive:
        label = f"Vessel Disease ({', '.join(vessels_positive)})"
    elif cath_res.get("prediction") == 1:
        label = "Coronary Artery Disease Indicated"
    else:
        label = "No vessel disease flagged by model"

    headline_prob = max(vessel_probs) if vessel_probs else cath_res.get("probability", 0.0)

    return {
        "status": "success",
        "label": label,
        "probability": round(headline_prob, 4),
        "results": formatted_results,
    }

def extract_polydata_list(mesh_or_block) -> List[pv.PolyData]:
    """
    Recursively extract all valid PolyData surfaces from a PyVista dataset or MultiBlock container.
    """
    res = []
    if isinstance(mesh_or_block, pv.MultiBlock):
        for block in mesh_or_block:
            if block is not None:
                res.extend(extract_polydata_list(block))
    elif isinstance(mesh_or_block, pv.PolyData):
        if mesh_or_block.n_points > 0:
            res.append(mesh_or_block)
    elif hasattr(mesh_or_block, "extract_surface"):
        try:
            surf = mesh_or_block.extract_surface()
            if surf.n_points > 0:
                res.append(surf)
        except Exception:
            pass
    return res

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/render-heart-html")
def render_heart_html():
    try:
        folder_name = os.path.join(BASE_DIR, "heart_folder2")
        if not os.path.exists(folder_name):
            return f"<div style='color:white;padding:20px;'>Folder '{folder_name}' not found.</div>", 404

        # Search for GLB, GLTF, and OBJ files (case-insensitive extensions)
        patterns = ["*.glb", "*.GLB", "*.gltf", "*.GLTF", "*.obj", "*.OBJ"]
        model_files = []
        for pattern in patterns:
            model_files.extend(glob.glob(os.path.join(folder_name, pattern)))
        model_files = sorted(list(set(model_files)))

        if not model_files:
            return "<div style='color:white;padding:20px;'>No 3D model files (.glb, .gltf, .obj) found inside 'heart_folder2'.</div>", 404

        bpm = float(request.args.get("bpm", 70))
        atrial_amp = float(request.args.get("atrial_amp", 0.10))
        vent_amp = float(request.args.get("vent_amp", 0.15))
        cut_axis = request.args.get("cut_axis", "y")
        cut_val = float(request.args.get("cut_val", 0))
        mesh_opacity = float(request.args.get("opacity", 1.0))

        color_map = {
            1: request.args.get("c_red", "#FF0000"),
            2: request.args.get("c_orange", "#FF7F00"),
            3: request.args.get("c_yellow", "#FFFF00"),
            4: request.args.get("c_green", "#00FF00"),
            5: request.args.get("c_blue", "#007FFF")
        }

        submeshes = []
        z_centroids = []

        for filepath in model_files:
            try:
                m = pv.read(filepath)
                extracted_surfaces = extract_polydata_list(m)
                for poly in extracted_surfaces:
                    submeshes.append(poly)
                    z_centroids.append(poly.center[2])
            except Exception as read_ex:
                log.warning("Skipping file %s: %s", filepath, read_ex)
                continue

        if not submeshes:
            return "<div style='color:white;padding:20px;'>Failed to parse any valid 3D geometry from model files.</div>", 500

        z_min, z_max = min(z_centroids), max(z_centroids)
        z_range = z_max - z_min + 1e-6

        combined = submeshes[0].merge(submeshes[1:])
        bounds = combined.bounds
        y_min_g, y_max_g = bounds[2], bounds[3]
        z_min_g, z_max_g = bounds[4], bounds[5]
        y_range = y_max_g - y_min_g
        z_range = z_max_g - z_min_g

        plotter = pv.Plotter(window_size=[900, 600], off_screen=True)
        plotter.set_background("#0e1117")

        if mesh_opacity < 1.0:
            plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)

        for mesh, z_c in zip(submeshes, z_centroids):
            z_norm = (z_c - z_min) / z_range
            if z_norm < 0.20: zone_id = 1
            elif z_norm < 0.40: zone_id = 2
            elif z_norm < 0.60: zone_id = 3
            elif z_norm < 0.80: zone_id = 4
            else: zone_id = 5

            zone_color = color_map.get(zone_id, "#FFFFFF")

            if cut_axis == "y":
                if cut_val > 0:
                    cut_y = y_max_g - ((cut_val / 100.0) * y_range)
                    clipped_mesh = mesh.clip(normal=(0, 1, 0), origin=(0, cut_y, 0))
                elif cut_val < 0:
                    cut_y = y_min_g + ((abs(cut_val) / 100.0) * y_range)
                    clipped_mesh = mesh.clip(normal=(0, -1, 0), origin=(0, cut_y, 0))
                else:
                    clipped_mesh = mesh
            else:
                if cut_val > 0:
                    cut_z = z_max_g - ((cut_val / 100.0) * z_range)
                    clipped_mesh = mesh.clip(normal=(0, 0, 1), origin=(0, 0, cut_z))
                elif cut_val < 0:
                    cut_z = z_min_g + ((abs(cut_val) / 100.0) * z_range)
                    clipped_mesh = mesh.clip(normal=(0, 0, -1), origin=(0, 0, cut_z))
                else:
                    clipped_mesh = mesh

            if clipped_mesh.n_points > 0:
                plotter.add_mesh(
                    clipped_mesh,
                    color=zone_color,
                    opacity=mesh_opacity,
                    smooth_shading=True,
                    specular=0.5,
                    specular_power=20,
                    show_edges=False,
                )

        plotter.reset_camera()
        plotter.camera.zoom(1.2)

        temp_fd, temp_path = tempfile.mkstemp(suffix=".html")
        os.close(temp_fd)
        try:
            plotter.export_html(temp_path)
            with open(temp_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        js_chamber_animation = f"""
        <script>
        (function checkVTK() {{
            const canvas = document.querySelector('canvas');
            if (!canvas) {{
                setTimeout(checkVTK, 100);
                return;
            }}

            const bpm = {bpm};
            const atriaAmp = {atrial_amp};
            const ventAmp = {vent_amp};
            const freq = (bpm / 60.0) * 2.0 * Math.PI;
            const startTime = performance.now();

            function animate() {{
                const t = (performance.now() - startTime) / 1000.0;
                const cyclePhase = (freq * t) % (2.0 * Math.PI);

                const atrialPulse = Math.sin(cyclePhase) > 0 ? Math.sin(cyclePhase) : 0;
                const ventPhase = cyclePhase - 0.8;
