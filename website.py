import glob
import os
import tempfile
import numpy as np
import pyvista as pv
import streamlit as st
import streamlit.components.v1 as components

# Enable off-screen rendering
pv.OFF_SCREEN = True

st.set_page_config(
    layout="wide", page_title="Multi-Axis Cross-Sectional Cardiac Viewer"
)

st.title("3D Human Heart Complete Analysis")
st.write(
    "3D anatomical heart with multi-axis bi-directional clipping, opacity controls"
)

# Sidebar Controls
st.sidebar.header("Cardiac Dynamics")
bpm = st.sidebar.slider("Heart Rate (BPM)", 40, 180, 75)
atrial_amp = st.sidebar.slider("Atrial Contraction", 0.0, 0.30, 0.12)
vent_amp = st.sidebar.slider("Ventricular Contraction", 0.0, 0.30, 0.18)

st.sidebar.header("Anatomical Cross-Section")
cut_axis = st.sidebar.radio(
    "Clipping Plane Axis",
    ["Front / Back (Y-Axis)", "Top / Bottom (Z-Axis)"],
    help="Select which spatial axis to slice through.",
)

cut_val = st.sidebar.slider("Cut Position", -100, 100, 0, step=1)

if cut_axis == "Front / Back (Y-Axis)":
    st.sidebar.caption(
        "**0** = Full Heart | **+1 to +100** = Cut Front to Back | **-1 to -100** = Cut Back to Front"
    )
else:
    st.sidebar.caption(
        "**0** = Full Heart | **+1 to +100** = Cut Top to Bottom | **-1 to -100** = Cut Bottom to Top"
    )

st.sidebar.header("Translucency & Visualization")
mesh_opacity = st.sidebar.slider(
    "Model Opacity",
    0.1,
    1.0,
    1.0,
    step=0.05,
    help="Adjust opacity to view internal cardiac structures.",
)

st.sidebar.header("Damage/Failure Risk Colors")
c_red = st.sidebar.color_picker("Zone 1 - Extreme Risk (Red)", "#FF0000")
c_orange = st.sidebar.color_picker("Zone 2 - High Risk (Orange)", "#FF7F00")
c_yellow = st.sidebar.color_picker("Zone 3 - Moderate Risk (Yellow)", "#FFFF00")
c_green = st.sidebar.color_picker("Zone 4 - Low Risk (Green)", "#00FF00")
c_blue = st.sidebar.color_picker("Zone 5 - Minimal Risk (Blue)", "#007FFF")

folder_name = "heart_folder"


# Step 1: Load Sub-meshes, Categorize Risk Zones & Bound Limits
@st.cache_resource
def load_and_classify_risk_zones(folder):
    if not os.path.exists(folder):
        return (
            None,
            f"Folder '{folder}' not found. Ensure 'heart_folder' is in the same directory.",
            0.0,
            1.0,
            0.0,
            1.0,
        )

    obj_files = glob.glob(os.path.join(folder, "*.obj"))
    if not obj_files:
        return None, f"No .obj files found inside '{folder}'.", 0.0, 1.0, 0.0, 1.0

    submeshes = []
    z_centroids = []

    for filepath in sorted(obj_files):
        try:
            m = pv.read(filepath)
            poly = m if isinstance(m, pv.PolyData) else m.extract_surface()
            if poly.n_points > 0:
                submeshes.append(poly)
                # Z-axis corresponds to vertical orientation (Apex to Base)
                z_centroids.append(poly.center[2])
        except Exception:
            continue

    if not submeshes:
        return None, "Failed to parse any valid .obj files.", 0.0, 1.0, 0.0, 1.0

    z_min, z_max = min(z_centroids), max(z_centroids)
    z_range = z_max - z_min + 1e-6

    classified_blocks = pv.MultiBlock()

    for idx, (mesh, z_c) in enumerate(zip(submeshes, z_centroids)):
        z_norm = (z_c - z_min) / z_range

        # Classify into 5 gradient risk zones along vertical height (Z-axis)
        if z_norm < 0.20:
            zone_id = 1  # Extreme (Apex / Lower Ventricles)
        elif z_norm < 0.40:
            zone_id = 2  # High (Mid Ventricles)
        elif z_norm < 0.60:
            zone_id = 3  # Moderate (AV Junction / Valves)
        elif z_norm < 0.80:
            zone_id = 4  # Low (Atria)
        else:
            zone_id = 5  # Minimal (Aorta & Major Vessels)

        mesh.field_data["risk_zone"] = [zone_id]
        classified_blocks.append(mesh, name=f"mesh_{idx}_zone_{zone_id}")

    # Compute global bounding box bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
    combined = submeshes[0].merge(submeshes[1:])
    bounds = combined.bounds

    return (
        classified_blocks,
        None,
        bounds[2],  # ymin
        bounds[3],  # ymax
        bounds[4],  # zmin
        bounds[5],  # zmax
    )


(
    blocks,
    error_msg,
    y_min_global,
    y_max_global,
    z_min_global,
    z_max_global,
) = load_and_classify_risk_zones(folder_name)

if error_msg or blocks is None:
    st.error(error_msg)
    st.stop()

st.sidebar.success(f"Categorized {len(blocks)} sub-meshes into 5 Risk Zones!")

# Step 2: Corrected Multi-Axis Bi-Directional Clipping & Rendering
plotter = pv.Plotter(window_size=[900, 600], off_screen=True)
plotter.set_background("#0e1117")

# Enable Order-Independent Transparency for clean internal visualization
if mesh_opacity < 1.0:
    plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)

color_map = {
    1: c_red,
    2: c_orange,
    3: c_yellow,
    4: c_green,
    5: c_blue,
}

y_range = y_max_global - y_min_global
z_range = z_max_global - z_min_global

for name in blocks.keys():
    mesh = blocks[name]
    zone_id = mesh.field_data["risk_zone"][0]
    zone_color = color_map.get(zone_id, "#FFFFFF")

    if cut_axis == "Front / Back (Y-Axis)":
        if cut_val > 0:
            # Cut Front to Back along Y-axis
            cut_y = y_max_global - ((cut_val / 100.0) * y_range)
            clipped_mesh = mesh.clip(normal=(0, 1, 0), origin=(0, cut_y, 0))
        elif cut_val < 0:
            # Cut Back to Front along Y-axis
            cut_y = y_min_global + ((abs(cut_val) / 100.0) * y_range)
            clipped_mesh = mesh.clip(normal=(0, -1, 0), origin=(0, cut_y, 0))
        else:
            clipped_mesh = mesh
    else:  # Top / Bottom (Z-Axis)
        if cut_val > 0:
            # Cut Top to Bottom along Z-axis
            cut_z = z_max_global - ((cut_val / 100.0) * z_range)
            clipped_mesh = mesh.clip(normal=(0, 0, 1), origin=(0, 0, cut_z))
        elif cut_val < 0:
            # Cut Bottom to Top along Z-axis
            cut_z = z_min_global + ((abs(cut_val) / 100.0) * z_range)
            clipped_mesh = mesh.clip(normal=(0, 0, -1), origin=(0, 0, cut_z))
        else:
            clipped_mesh = mesh

    # Render remaining clipped sub-mesh if non-empty
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

# Export base VTK WebGL HTML
with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
    plotter.export_html(tf.name)
    with open(tf.name, "r", encoding="utf-8") as f:
        html_content = f.read()


# Step 3: Inject Dual-Phase Pulsation Script
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

        // Atrial Phase
        const atrialPulse = Math.sin(cyclePhase) > 0 ? Math.sin(cyclePhase) : 0;

        // Ventricular Phase (0.8 rad AV delay)
        const ventPhase = cyclePhase - 0.8;
        const ventPulse = Math.sin(ventPhase) > 0 ? Math.pow(Math.sin(ventPhase), 2) : 0;

        // Non-uniform spatial scaling factors
        const scaleAtriaX = 1.0 - (atriaAmp * atrialPulse);
        const scaleAtriaY = 1.0 - (atriaAmp * 0.5 * atrialPulse);
        
        const scaleVentX  = 1.0 - (ventAmp * ventPulse);
        const scaleVentY  = 1.0 - (ventAmp * 0.3 * ventPulse);

        const scaleZ = 1.0 - (atriaAmp * 0.3 * atrialPulse + ventAmp * 0.5 * ventPulse);

        // WebGL Matrix transformation
        canvas.style.transform = `matrix3d(
            ${{scaleAtriaX}}, 0, 0, 0,
            0, ${{scaleVentY}}, 0, 0,
            0, 0, ${{scaleZ}}, 0,
            0, 0, 0, 1
        )`;
        canvas.style.transformOrigin = 'center center';

        requestAnimationFrame(animate);
    }}
    animate();
}})();
</script>
"""

components.html(html_content + js_chamber_animation, height=600, scrolling=False)

# Open cmd
# cd "C:\Users\aakar\OneDrive\Desktop\VS Code Python\Multimodal
# python -m streamlit run website.py