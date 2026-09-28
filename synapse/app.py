"""
app.py — Synapse 1.0 Streamlit MVP
------------------------------------
Real-Time Coronary Narrowing Detection Prototype

MEDICAL DISCLAIMER:
This application is a RESEARCH PROTOTYPE, NOT a medical device.
It is NOT clinically validated and must NOT be used for clinical decisions.

Detection scores are heuristic image-processing outputs, NOT trained ML
model probabilities. Displayed confidence is a prototype heuristic blend
of image-based candidate score and temporal persistence.

Labels used in this app:
  "Suspicious Candidate"      — a region flagged by the OpenCV heuristic
  "Detection Score"           — heuristic image-based score, NOT ML probability
  "Temporal Persistence"      — how many frames the candidate was tracked
  "AI Model Score"            — CADICA-trained prototype classifier output
                                (NOT a calibrated clinical probability)
"""

import time
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ---- Ensure repo root is on sys.path ----
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.sample_data import generate_synthetic_sequence
from pipeline.preprocessor import preprocess_frame
from pipeline.detector import detect_candidates
from pipeline.tracker import TemporalTracker
from pipeline.confidence import DISPLAYED_CONFIDENCE_LABEL
from pipeline.ml_model import MLClassifier
from utils.visualization import annotate_frame

@st.cache_resource
def load_classifier():
    clf = MLClassifier()
    return clf

ml_classifier = load_classifier()


# ================================================================== #
# PAGE CONFIG
# ================================================================== #

st.set_page_config(
    page_title="Synapse 1.0 — Coronary Narrowing Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ================================================================== #
# CUSTOM CSS — minimal, clean, dark-leaning
# ================================================================== #

st.markdown("""
<style>
    /* 
      SYNAPSE 1.0 MODULAR CSS 
      Background: #080D18 | Surface: #0F172A | Elev: #131F33
      Cyan: #22D3EE | Blue: #38BDF8 | Violet: #8B5CF6
    */
    .stApp {
        background-color: #080D18;
    }

    /* Typography & Core Elements */
    .stApp {
        font-family: "Inter", "Segoe UI", Roboto, sans-serif;
    }
    h1, h2, h3, h4, p {
        font-family: "Inter", "Segoe UI", Roboto, sans-serif !important;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -1px;
        color: #F8FAFC;
        margin-bottom: -0.5rem;
    }
    .hero-title span {
        color: #22D3EE;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94A3B8;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .hero-meta {
        font-size: 0.85rem;
        font-weight: 500;
        color: #38BDF8;
        letter-spacing: 1px;
    }
    .hero-meta span {
        margin-right: 15px;
    }
    .hero-disclaimer {
        display: inline-block;
        margin-top: 15px;
        background-color: #3d1a1a;
        border: 1px solid #F43F5E;
        color: #F43F5E;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
    }

    /* Modular Status Strip */
    .status-card {
        background: #0F172A;
        border: 1px solid #26354D;
        border-radius: 6px;
        padding: 15px;
        text-align: left;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .status-label {
        font-size: 0.75rem;
        color: #94A3B8;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 5px;
        text-transform: uppercase;
    }
    .status-value {
        font-size: 2rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1;
    }
    .status-value.accent-cyan { color: #22D3EE; }
    .status-value.accent-blue { color: #38BDF8; }
    .status-value.accent-violet { color: #8B5CF6; }
    .status-value.accent-emerald { color: #10B981; }

    /* Module Containers */
    .module-container {
        background: #0F172A;
        border: 1px solid #26354D;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .module-header {
        font-size: 0.9rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 15px;
        border-bottom: 1px solid #26354D;
        padding-bottom: 8px;
    }
    
    /* Pipeline Nodes */
    .pipeline-node {
        display: inline-block;
        background: #131F33;
        border: 1px solid #26354D;
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #F8FAFC;
        text-align: center;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .pipeline-arrow {
        display: inline-block;
        color: #38BDF8;
        font-size: 1.2rem;
        vertical-align: middle;
        margin: 0 2px;
    }
    .node-cyan { border-left: 3px solid #22D3EE; }
    .node-violet { border-left: 3px solid #8B5CF6; }
    .node-blue { border-left: 3px solid #38BDF8; }
    .node-emerald { border-left: 3px solid #10B981; }

    /* Image Overlays */
    .image-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #22D3EE;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    /* Dataframe tweaks */
    .stDataFrame {
        border: 1px solid #26354D !important;
        border-radius: 6px !important;
    }
    
    /* Legend */
    .legend-item {
        display: inline-block;
        margin-right: 15px;
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 600;
    }
    .legend-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-high { background: rgb(80, 200, 0); }
    .dot-mid { background: rgb(220, 180, 0); }
    .dot-low { background: rgb(200, 80, 80); }


    /* Misc */
    hr {
        border-top: 1px solid #26354D;
    }
    
    img {
        border: 1px solid #26354D;
        border-radius: 4px;
        image-rendering: pixelated;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================== #
# HEADER
# ================================================================== #

st.markdown(f"""
<div style="margin-bottom: 30px;">
    <div class="hero-title">SYNAPSE <span>1.0</span></div>
    <div class="hero-subtitle">Real-Time Coronary Narrowing Analysis</div>
    <div class="hero-meta">
        <span>● AI-ASSISTED</span>
        <span>● TEMPORAL ANALYSIS</span>
        <span>● RESEARCH PROTOTYPE</span>
    </div>
    <div class="hero-disclaimer">
        ⚠ PROTOTYPE — NOT FOR CLINICAL USE
    </div>
</div>
""", unsafe_allow_html=True)


# ================================================================== #
# SIDEBAR — SETTINGS
# ================================================================== #

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Demo parameters")

    n_frames_demo = st.slider(
        "Sample demo frames", min_value=20, max_value=60, value=60, step=10
    )
    display_delay = st.slider(
        "Display delay (ms) — slows playback for readability only",
        min_value=0, max_value=200, value=30, step=10,
        help="Adds a pause between frames for visual clarity. Does NOT affect measured processing FPS.",
    )
    min_persistence_show = st.slider(
        "Min persistence to show in table", min_value=1, max_value=10, value=2
    )
    iou_threshold = st.slider(
        "IoU threshold (tracker)", min_value=0.05, max_value=0.50, value=0.15, step=0.05
    )
    max_missing = st.slider(
        "Max missing frames (tracker)", min_value=1, max_value=8, value=3
    )

    st.divider()
    st.header("🧠 AI Model")
    st.caption("CADICA HOG + Logistic Regression")
    st.markdown("""
    **Prototype held-out test metrics**  
    - Training: 29 patients  
    - Held-out test: 13 patients  
    - Test samples: 799  
    - Accuracy: 79.97%  
    - Precision: 82.72%  
    - Recall: 50.38%  
    - F1: 0.6262  
    """)

    st.divider()
    st.caption("⚠️ Prototype — Not for Clinical Use")


# ================================================================== #
# INPUT SECTION
# ================================================================== #

st.subheader("📂 Input")

tab_demo, tab_upload = st.tabs(["▶ Sample Demo", "📤 Video Upload"])

source_frames = None   # list of np.ndarray (raw grayscale)
source_label  = ""

with tab_demo:
    st.markdown(
        "Click **Run Sample Demo** to process a synthetic angiography-like sequence "
        "with **3 simulated narrowing regions** of known temporal extent.\n\n"
        "_These frames are procedurally generated test fixtures — NOT real patient data._"
    )
    run_demo = st.button("▶ Run Sample Demo", type="primary", key="btn_demo")
    if run_demo:
        with st.spinner("Generating synthetic sequence..."):
            source_frames = generate_synthetic_sequence(n_frames=n_frames_demo)
        source_label = f"Synthetic demo ({n_frames_demo} frames)"
        st.success(f"Loaded {len(source_frames)} synthetic frames.")

with tab_upload:
    st.markdown(
        "Upload a grayscale angiography video (MP4, AVI, MOV). "
        "The detector is tuned for the synthetic demo; real video results are **not validated**."
    )
    uploaded = st.file_uploader(
        "Upload video file", type=["mp4", "avi", "mov", "mkv"], key="video_upload"
    )
    if uploaded is not None:
        # Save to temp file so OpenCV can read it
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            cap = cv2.VideoCapture(tmp_path)
            vid_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vid_frames.append(frame)
            cap.release()
            os.unlink(tmp_path)
            if not vid_frames:
                st.error("Could not read frames from the uploaded file.")
            else:
                source_frames = vid_frames
                source_label  = f"Uploaded: {uploaded.name} ({len(vid_frames)} frames)"
                st.success(f"Loaded {len(vid_frames)} frames from upload.")
        except Exception as e:
            st.error(f"Video load failed: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ================================================================== #
# PROCESSING SECTION
# ================================================================== #

if source_frames is None:
    st.info("👆 Click **Run Sample Demo** above to start, or upload a video.")
    st.stop()

# Validate
if len(source_frames) == 0:
    st.error("No frames available. Please try again.")
    st.stop()

total_frames = len(source_frames)

st.divider()
st.subheader("🔬 Frame-by-Frame Analysis")
st.caption(f"Source: {source_label}")

# ---- Layout ----
st.markdown('<div class="module-header" style="margin-top: 30px;">LIVE FRAME ANALYSIS</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2)
with col_left:
    st.markdown('<div class="image-title">ORIGINAL FRAME</div>', unsafe_allow_html=True)
    orig_placeholder = st.empty()
with col_right:
    st.markdown('<div class="image-title">AI DETECTION OVERLAY</div>', unsafe_allow_html=True)
    overlay_placeholder = st.empty()

st.markdown("""
<div style="margin-top: 10px; margin-bottom: 20px;">
    <span class="legend-item"><span class="legend-dot dot-high"></span>HIGH AI SCORE (≥65%)</span>
    <span class="legend-item"><span class="legend-dot dot-mid"></span>MED AI SCORE (40-64%)</span>
    <span class="legend-item"><span class="legend-dot dot-low"></span>LOW AI SCORE (<40%)</span>
</div>
""", unsafe_allow_html=True)

# ---- Modular Status Strip ----
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
frame_counter_ph  = mc1.empty()
fps_ph            = mc2.empty()
active_lesions_ph = mc3.empty()
avg_score_ph      = mc4.empty()
status_ph         = mc5.empty()

def render_metric(label, value, accent_class=""):
    return f"""
    <div class="status-card">
        <div class="status-label">{label}</div>
        <div class="status-value {accent_class}">{value}</div>
    </div>
    """

st.divider()

# ---- Lesion summary table ----

st.divider()

# ---- Temporal persistence explanation ----


# ================================================================== #
# CORE PROCESSING LOOP
# Each frame: detect → track → visualize → display
# This is NOT pre-computed — the pipeline runs sequentially per frame.
# ================================================================== #

tracker = TemporalTracker(
    iou_threshold=iou_threshold,
    max_missing_frames=max_missing,
    min_persistence=1,
)

all_scores_seen: list = []
all_ml_scores_seen: list = []
processing_times: list = []

try:
    for fi, raw_frame in enumerate(source_frames):

        # ---- Validate frame ----
        if raw_frame is None or raw_frame.size == 0:
            continue
        if raw_frame.ndim == 3:
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        if raw_frame.dtype != np.uint8:
            raw_frame = np.clip(raw_frame, 0, 255).astype(np.uint8)

        # ---- Measure actual pipeline time ----
        t0 = time.perf_counter()
        enhanced = preprocess_frame(raw_frame)
        dets      = detect_candidates(raw_frame, frame_idx=fi, ml_model=ml_classifier)
        active    = tracker.update(fi, dets)
        ann       = annotate_frame(raw_frame, active, frame_idx=fi)
        t1        = time.perf_counter()

        frame_time = t1 - t0
        processing_times.append(frame_time)

        # ---- Collect scores ----
        for d in dets:
            all_scores_seen.append(d.raw_score)
            all_ml_scores_seen.append(d.ml_score)

        # ---- Compute live FPS (actual processing only) ----
        elapsed_processing = sum(processing_times)
        actual_fps = (fi + 1) / elapsed_processing if elapsed_processing > 0 else 0.0

        # ---- Display original frame ----
        orig_rgb = cv2.cvtColor(
            cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR),
            cv2.COLOR_BGR2RGB,
        )
        orig_placeholder.image(orig_rgb, use_container_width=True)

        # ---- Display annotated overlay ----
        ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
        overlay_placeholder.image(ann_rgb, use_container_width=True)

        # ---- Update live metrics ----
        frame_counter_ph.markdown(render_metric("FRAME", f"{fi + 1} / {total_frames}"), unsafe_allow_html=True)
        fps_ph.markdown(render_metric("PROCESSING", f"{actual_fps:.1f} FPS", "accent-cyan"), unsafe_allow_html=True)
        active_lesions_ph.markdown(render_metric("ACTIVE CANDIDATES", len(active), "accent-blue"), unsafe_allow_html=True)
        
        avg_ml = float(np.mean(all_ml_scores_seen)) if all_ml_scores_seen else 0.0
        avg_score_ph.markdown(render_metric("AI SCORE", f"{avg_ml:.1%}", "accent-violet"), unsafe_allow_html=True)
        
        stat = "PROCESSING" if fi < total_frames - 1 else "COMPLETE"
        stat_col = "accent-emerald" if stat == "COMPLETE" else "accent-cyan"
        status_ph.markdown(render_metric("STATUS", stat, stat_col), unsafe_allow_html=True)

        # (Live table update removed in modular UI)

        # ---- Display delay (playback only — NOT included in FPS) ----
        if display_delay > 0:
            time.sleep(display_delay / 1000.0)

except Exception as exc:
    st.error(f"Processing error on frame {fi}: {exc}")
    with st.expander("Debug info"):
        st.code(traceback.format_exc())
    st.stop()


# ================================================================== #
# POST-PROCESSING SUMMARY & MODULES
# ================================================================== #

st.markdown('<div class="module-header" style="margin-top: 40px;">SYSTEM ARCHITECTURE & INTELLIGENCE</div>', unsafe_allow_html=True)

pc1, pc2, pc3 = st.columns([2, 1.5, 1.5])

with pc1:
    st.markdown("""
    <div class="module-container">
        <div class="module-header">AI PIPELINE</div>
        <div style="text-align: center; margin-top: 15px;">
            <div class="pipeline-node node-cyan">VIDEO / FRAME</div> <span class="pipeline-arrow">↓</span><br>
            <div class="pipeline-node node-cyan">CANDIDATE PROPOSAL</div> <span class="pipeline-arrow">↓</span><br>
            <div class="pipeline-node node-violet">HOG FEATURES</div> <span class="pipeline-arrow">↓</span><br>
            <div class="pipeline-node node-violet">LOGISTIC REGRESSION</div> <span class="pipeline-arrow">↓</span><br>
            <div class="pipeline-node node-blue">TEMPORAL TRACKING</div> <span class="pipeline-arrow">↓</span><br>
            <div class="pipeline-node node-emerald">PERSISTENCE ANALYSIS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with pc2:
    st.markdown("""
    <div class="module-container">
        <div class="module-header">AI MODEL</div>
        <div style="color: #F8FAFC; font-weight: 600; margin-bottom: 10px;">CADICA TRAINED</div>
        <div style="color: #38BDF8; font-size: 0.9rem; font-weight: 700; margin-bottom: 10px;">HOG FEATURES + LOGISTIC REGRESSION</div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 15px;">SUPERVISED CLASSIFIER</div>
        
        <div style="font-size: 0.8rem; color: #8A9BB5;">
            • Patient-level train/test split<br>
            • Prototype crop classification<br>
            • Real angiography data
        </div>
    </div>
    """, unsafe_allow_html=True)

with pc3:
    total_elapsed_processing = sum(processing_times)
    st.markdown(f"""
    <div class="module-container">
        <div class="module-header">TEMPORAL INTELLIGENCE</div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 15px;">
            Tracks candidate regions across consecutive frames to identify persistent regions.
        </div>
        <div style="font-family: monospace; color: #22D3EE; font-size: 0.75rem; text-align: center; background: #131F33; padding: 10px; border-radius: 4px;">
            FR 01 &nbsp; FR 02 &nbsp; FR 03 &nbsp; FR 04<br>
            ● ─── ● ─── ● ─── ●<br>
            <span style="color:#10B981; font-weight: bold; margin-top: 5px; display: inline-block;">PERSISTENT</span>
        </div>
        
        <div class="module-header" style="margin-top: 20px;">PERFORMANCE</div>
        <div style="color: #94A3B8; font-size: 0.85rem; display: flex; justify-content: space-between;">
            <span>PIPELINE TIME</span>
            <span style="color: #10B981; font-weight: bold;">{total_elapsed_processing:.3f}s</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


all_tracks_final = tracker.get_all_tracks(min_persistence=min_persistence_show)

# Final lesion table
if all_tracks_final:
    st.markdown('<div class="module-header" style="margin-top: 20px;">FINAL CANDIDATE REPORT</div>', unsafe_allow_html=True)
    st.caption("Tracked candidate regions across the analyzed sequence.")

    final_rows = []
    for t in sorted(all_tracks_final, key=lambda x: -x.displayed_confidence):
        final_rows.append({
            "Candidate ID":     f"#{t.lesion_id}",
            "First Frame":      t.first_frame,
            "Last Frame":       t.last_frame,
            "Persistence":      t.persistence_count,
            "Lifespan (fr)":    t.last_frame - t.first_frame + 1,
            "Persist Ratio":    f"{t.persistence_ratio:.0%}",
            "Mean Detect Score": f"{t.mean_raw_score:.3f}",
            "AI Score":         f"{t.mean_ml_score:.0%}",
            "Status":           "Active" if t.active else "Closed",
        })
    df_final = pd.DataFrame(final_rows)
    st.dataframe(df_final, use_container_width=True, hide_index=True)

st.markdown("""
<div style="background: #1A1211; border-left: 4px solid #F59E0B; padding: 15px; border-radius: 4px; margin-top: 30px; border-right: 1px solid #26354D; border-top: 1px solid #26354D; border-bottom: 1px solid #26354D;">
    <div style="color: #F59E0B; font-weight: 700; margin-bottom: 8px;">⚠ RESEARCH PROTOTYPE / LIMITATIONS</div>
    <div style="color: #ff9494; font-size: 0.85rem; line-height: 1.5;">
        AI model scores are prototype classification outputs. They are not calibrated clinical probabilities, diagnostic confidence values, or stenosis percentages.<br><br>
        Overlay visibility filtering is for visualization clarity only; it is not a clinical detection threshold.
    </div>
</div>
""", unsafe_allow_html=True)
