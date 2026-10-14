import os
import logging
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from authlib.integrations.flask_client import OAuth
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("cardiac")

# ==========================================
# ML Predict Module Setup
# ==========================================
try:
    from predict import predict_patient

    PREDICT_MODULE_LOADED = True
    MODULE_ERROR = None
except Exception as e:
    predict_patient = None
    PREDICT_MODULE_LOADED = False
    MODULE_ERROR = str(e)
    log.error("predict module import failed: %s", MODULE_ERROR)

# ==========================================
# Flask Application & Config
# ==========================================
app = Flask(__name__)

# Render terminates TLS at its proxy; trust its forwarded headers so
# url_for(_external=True) produces https:// URLs (needed for Google OAuth).
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Required: crash at startup if missing, so workers never use different random keys.
app.secret_key = os.environ["FLASK_SECRET_KEY"]

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    # Secure cookies by default. For local http dev only: export INSECURE_COOKIES=1
    SESSION_COOKIE_SECURE=os.environ.get("INSECURE_COOKIES") != "1",
    MAX_CONTENT_LENGTH=256 * 1024,  # 256 KB request cap
)

DEBUG = os.environ.get("FLASK_DEBUG") == "1"
REQUIRE_LOGIN = os.environ.get("REQUIRE_LOGIN", "1") == "1"  # on by default
MAX_BATCH = 50

if os.environ.get("OAUTH_INSECURE_LOCAL") == "1":
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# NOTE: memory:// storage is per-worker. Fine for a demo; use Redis for strict limits.
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

# ==========================================
# Database Configuration
# ==========================================
db_url = os.environ["DATABASE_URL"]  # no fallback: never hardcode credentials

# Legacy URI prefix fix (postgres:// -> postgresql://)
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True, "pool_recycle": 280}

db = SQLAlchemy(app)


# ==========================================
# Database Models
# ==========================================
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=True)
    provider = db.Column(db.String(20), default="local")
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    assessments = db.relationship("Assessment", backref="user", lazy=True, cascade="all, delete-orphan")


class Assessment(db.Model):
    __tablename__ = "assessments"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    timestamp = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    input_data = db.Column(db.JSON, nullable=False)
    result_data = db.Column(db.JSON, nullable=False)


@app.cli.command("init-db")
def init_db():
    """Create tables once: `flask --app app init-db`"""
    db.create_all()
    print("tables created")


# Demo convenience only (use with --workers 1): set AUTO_CREATE_DB=1
if os.environ.get("AUTO_CREATE_DB") == "1":
    with app.app_context():
        db.create_all()


# ==========================================
# OAuth Setup
# ==========================================
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
else:
    log.warning("Google OAuth disabled: GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not configured.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


# ==========================================
# Helpers & Data Cleaning
# ==========================================
def build_input_frame(rows: List[Dict[str, Any]]) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Normalise column names, validate feature presence, and coerce numeric values."""
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
    """Recursively convert NumPy types to Python primitives and NaN/Inf to None (valid for Postgres JSON)."""
    if isinstance(val, dict):
        return {k: clean_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [clean_json(v) for v in val]
    if isinstance(val, np.generic):
        val = val.item()
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


def sanitize_dict_types(data: Dict[str, Any]) -> Dict[str, Any]:
    return clean_json(data)


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
    """Runs prediction model on single patient record and formats risk markers."""
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
        # Deliberately not phrased as reassurance. Update the UI copy to match.
        label = "No vessel disease flagged by model"

    headline_prob = max(vessel_probs) if vessel_probs else cath_res.get("probability", 0.0)

    return {
        "status": "success",
        "label": label,
        "probability": round(headline_prob, 4),
        "results": formatted_results,
    }


# ==========================================
# Routes: Base & Health Checks
# ==========================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        log.exception("Health check DB failure")
        db_ok = False

    ok = db_ok and PREDICT_MODULE_LOADED
    return jsonify({
        "status": "online" if ok else "degraded",
        "predict_module_loaded": PREDICT_MODULE_LOADED,
        "database_connected": db_ok,
        "n_features_expected": len(FEATURE_ORDER),
    }), (200 if ok else 503)


# ==========================================
# Routes: Authentication
# ==========================================
@app.route("/login/google")
def google_login():
    if google is None:
        return jsonify({"error": "Google login is not configured on this server."}), 503
    redirect_uri = url_for("google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route("/login/google/callback")
def google_callback():
    if google is None:
        return jsonify({"error": "Google login is not configured on this server."}), 503
    try:
        token = google.authorize_access_token()
        user_info = token.get("userinfo") or {}
        if not user_info.get("email") or not user_info.get("email_verified"):
            return redirect("/?error=email_unverified")

        email = user_info["email"].lower()
        user = User.query.filter_by(email=email).first()
        if user and user.provider != "google":
            # Prevents pre-hijacking via an attacker-created local account.
            return redirect("/?error=account_exists_use_password")
        if not user:
            user = User(email=email, provider="google")
            db.session.add(user)
            db.session.commit()

        session.clear()
        session["user_email"] = user.email
        return redirect("/")
    except Exception:
        db.session.rollback()
        log.exception("Google OAuth callback failed")
        return redirect("/?error=oauth_failed")


@app.route("/api/me", methods=["GET"])
def check_me():
    user = get_current_user()
    if user:
        return jsonify({"logged_in": True, "email": user.email})
    return jsonify({"logged_in": False})


@app.route("/login", methods=["POST"])
@limiter.limit("10 per minute")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if user and user.password_hash and check_password_hash(user.password_hash, password):
        session.clear()
        session["user_email"] = user.email
        return jsonify({"status": "success", "email": user.email})
    return jsonify({"status": "error", "message": "Invalid email or password."}), 401


@app.route("/signup", methods=["POST"])
@limiter.limit("5 per minute")
def signup():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or "@" not in email or len(email) > 120:
        return jsonify({"status": "error", "message": "Valid email required."}), 400
    if len(password) < 8:
        return jsonify({"status": "error", "message": "Password must be at least 8 characters long."}), 400
    if len(password) > 256:
        return jsonify({"status": "error", "message": "Password too long."}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"status": "error", "message": "An account with this email already exists."}), 400

    user = User(email=email, password_hash=generate_password_hash(password), provider="local")
    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({"status": "error", "message": "An account with this email already exists."}), 400

    session.clear()
    session["user_email"] = user.email
    return jsonify({"status": "success", "email": user.email})


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "success"})


# ==========================================
# Routes: Assessment History
# ==========================================
@app.route("/api/history", methods=["GET"])
def get_history():
    user = get_current_user()
    if not user:
        return jsonify([])

    records = Assessment.query.filter_by(user_id=user.id).order_by(Assessment.timestamp.desc()).all()
    history = [
        {
            "id": r.id,
            "ts": int(r.timestamp.timestamp() * 1000),
            "data": r.input_data,
            "result": r.result_data,
        }
        for r in records
    ]
    return jsonify(history)


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_history(record_id: int):
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    record = Assessment.query.filter_by(id=record_id, user_id=user.id).first()
    if record:
        db.session.delete(record)
        db.session.commit()
        return jsonify({"status": "success"})
    return jsonify({"error": "Record not found"}), 404


# ==========================================
# Routes: Inference & Auto-Save
# ==========================================
@app.route("/predict", methods=["POST"])
@limiter.limit("30 per minute")
def predict():
    denied = login_required_check()
    if denied:
        return denied

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON or empty payload."}), 400

    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    elif isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        return jsonify({"error": "Unsupported request format."}), 400

    if not rows or len(rows) > MAX_BATCH or not all(isinstance(r, dict) for r in rows):
        return jsonify({"error": f"Provide 1-{MAX_BATCH} JSON objects."}), 400

    if not PREDICT_MODULE_LOADED or predict_patient is None:
        return jsonify({"status": "error", "error": "Inference engine unavailable."}), 503

    input_df, problem = build_input_frame(rows)
    if problem:
        return jsonify({"status": "error", **problem}), 422

    user = get_current_user()
    evaluations = []

    try:
        for idx in range(len(input_df)):
            patient_data_dict = input_df.iloc[idx].to_dict()
            response_payload = clean_json(process_single_patient_prediction(patient_data_dict))

            # Only persist for authenticated users; anonymous rows would be unreachable.
            if user:
                db.session.add(Assessment(
                    user_id=user.id,
                    input_data=sanitize_dict_types(patient_data_dict),
                    result_data=response_payload,
                ))
            evaluations.append(response_payload)

        if user:
            db.session.commit()
            log.info("Persisted %d assessment record(s).", len(evaluations))

    except Exception:
        db.session.rollback()
        log.exception("Error executing inference engine or writing to database.")
        return jsonify({
            "status": "error",
            "error": "Prediction failed. Please check your input and try again.",
        }), 500

    if len(evaluations) == 1:
        return jsonify(evaluations[0])
    return jsonify({"status": "success", "batch_count": len(evaluations), "evaluations": evaluations})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=DEBUG)