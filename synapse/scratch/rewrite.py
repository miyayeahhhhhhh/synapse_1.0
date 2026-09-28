import re
import sys

def rewrite():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    css_block = """st.markdown(\"\"\"
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
    h1, h2, h3, h4, p, span, div {
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
    .dot-ai { background: #8B5CF6; }
    .dot-persist { background: #10B981; }
    .dot-transient { background: #F59E0B; }
    .dot-filtered { border: 1px solid #94A3B8; }

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
\"\"\", unsafe_allow_html=True)"""

    # Replace the CSS block
    content = re.sub(r'st\.markdown\("""\n<style>.*?</style>\n""", unsafe_allow_html=True)', css_block, content, flags=re.DOTALL)
    
    # Replace HEADER
    header_block = """# ================================================================== #
# HERO HEADER
# ================================================================== #

st.markdown(f\"\"\"
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
\"\"\", unsafe_allow_html=True)"""
    
    content = re.sub(r'# ==.*?HEADER.*?# ==.*?\n(.*?)# ==.*?SIDEBAR', header_block + '\n\n\n# ================================================================== #\n# SIDEBAR', content, flags=re.DOTALL)

    # Layout area
    layout_str = """# ---- Layout ----
st.markdown('<div class="module-header">LIVE FRAME ANALYSIS</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2)
with col_left:
    st.markdown('<div class="image-title">ORIGINAL FRAME</div>', unsafe_allow_html=True)
    orig_placeholder = st.empty()
with col_right:
    st.markdown('<div class="image-title">AI DETECTION OVERLAY</div>', unsafe_allow_html=True)
    overlay_placeholder = st.empty()

st.markdown(\"\"\"
<div style="margin-top: 10px; margin-bottom: 20px;">
    <span class="legend-item"><span class="legend-dot dot-ai"></span>AI CANDIDATE</span>
    <span class="legend-item"><span class="legend-dot dot-persist"></span>PERSISTENT</span>
    <span class="legend-item"><span class="legend-dot dot-transient"></span>TRANSIENT</span>
    <span class="legend-item"><span class="legend-dot dot-filtered"></span>FILTERED FROM VIEW</span>
</div>
\"\"\", unsafe_allow_html=True)"""
    
    content = re.sub(r'# ---- Layout ----.*?overlay_placeholder = st\.empty\(\)', layout_str, content, flags=re.DOTALL)

    # Metrics row replacement
    metric_str = """# ---- Modular Status Strip ----
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
frame_counter_ph  = mc1.empty()
fps_ph            = mc2.empty()
active_lesions_ph = mc3.empty()
avg_score_ph      = mc4.empty()
status_ph         = mc5.empty()

def render_metric(label, value, accent_class=""):
    return f\"\"\"
    <div class="status-card">
        <div class="status-label">{label}</div>
        <div class="status-value {accent_class}">{value}</div>
    </div>
    \"\"\"
"""
    content = re.sub(r'# ---- Live metrics row ----.*?status_ph\s*=\s*mc5\.empty\(\)', metric_str, content, flags=re.DOTALL)

    # Update metrics in loop
    update_metrics_str = """        # ---- Update live metrics ----
        frame_counter_ph.markdown(render_metric("FRAME", f"{fi + 1} / {total_frames}"), unsafe_allow_html=True)
        fps_ph.markdown(render_metric("PROCESSING", f"{actual_fps:.1f} FPS", "accent-cyan"), unsafe_allow_html=True)
        active_lesions_ph.markdown(render_metric("ACTIVE CANDIDATES", len(active), "accent-blue"), unsafe_allow_html=True)
        
        avg_ml = float(np.mean(all_ml_scores_seen)) if all_ml_scores_seen else 0.0
        avg_score_ph.markdown(render_metric("AI SCORE", f"{avg_ml:.1%}", "accent-violet"), unsafe_allow_html=True)
        
        stat = "PROCESSING" if fi < total_frames - 1 else "COMPLETE"
        stat_col = "accent-emerald" if stat == "COMPLETE" else "accent-cyan"
        status_ph.markdown(render_metric("STATUS", stat, stat_col), unsafe_allow_html=True)"""
        
    content = re.sub(r'# ---- Update live metrics ----.*?status_ph\.metric.*?$', update_metrics_str, content, flags=re.DOTALL | re.MULTILINE)

    # Remove the st.divider and old pipeline logic, put in modular blocks
    post_proc_str = """# ================================================================== #
# POST-PROCESSING SUMMARY & MODULES
# ================================================================== #

st.markdown('<div class="module-header">SYSTEM ARCHITECTURE & INTELLIGENCE</div>', unsafe_allow_html=True)

pc1, pc2, pc3 = st.columns([2, 1.5, 1.5])

with pc1:
    st.markdown('<div class="module-container">', unsafe_allow_html=True)
    st.markdown('<div class="module-header">AI PIPELINE</div>', unsafe_allow_html=True)
    st.markdown(\"\"\"
    <div style="text-align: center; margin-top: 15px;">
        <div class="pipeline-node node-cyan">VIDEO / FRAME</div> <span class="pipeline-arrow">↓</span><br>
        <div class="pipeline-node node-cyan">CANDIDATE PROPOSAL</div> <span class="pipeline-arrow">↓</span><br>
        <div class="pipeline-node node-violet">HOG FEATURES</div> <span class="pipeline-arrow">↓</span><br>
        <div class="pipeline-node node-violet">LOGISTIC REGRESSION</div> <span class="pipeline-arrow">↓</span><br>
        <div class="pipeline-node node-blue">TEMPORAL TRACKING</div> <span class="pipeline-arrow">↓</span><br>
        <div class="pipeline-node node-emerald">PERSISTENCE ANALYSIS</div>
    </div>
    \"\"\", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with pc2:
    st.markdown('<div class="module-container">', unsafe_allow_html=True)
    st.markdown('<div class="module-header">AI MODEL</div>', unsafe_allow_html=True)
    st.markdown(\"\"\"
    <div style="color: #F8FAFC; font-weight: 600; margin-bottom: 10px;">CADICA TRAINED</div>
    <div style="color: #38BDF8; font-size: 0.9rem; font-weight: 700; margin-bottom: 10px;">HOG FEATURES + LOGISTIC REGRESSION</div>
    <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 15px;">SUPERVISED CLASSIFIER</div>
    
    <div style="font-size: 0.8rem; color: #8A9BB5;">
        • Patient-level train/test split<br>
        • Prototype crop classification<br>
        • Real angiography data
    </div>
    \"\"\", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with pc3:
    st.markdown('<div class="module-container">', unsafe_allow_html=True)
    st.markdown('<div class="module-header">TEMPORAL INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown(\"\"\"
    <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 15px;">
        Tracks candidate regions across consecutive frames to identify persistent regions.
    </div>
    <div style="font-family: monospace; color: #22D3EE; font-size: 0.75rem; text-align: center; background: #131F33; padding: 10px; border-radius: 4px;">
        FR 01 &nbsp; FR 02 &nbsp; FR 03 &nbsp; FR 04<br>
        ● ─── ● ─── ● ─── ●<br>
        <span style="color:#10B981; font-weight: bold; margin-top: 5px; display: inline-block;">PERSISTENT</span>
    </div>
    \"\"\", unsafe_allow_html=True)
    
    total_elapsed_processing = sum(processing_times)
    st.markdown('<div class="module-header" style="margin-top: 20px;">PERFORMANCE</div>', unsafe_allow_html=True)
    st.markdown(f\"\"\"
    <div style="color: #94A3B8; font-size: 0.85rem; display: flex; justify-content: space-between;">
        <span>PIPELINE TIME</span>
        <span style="color: #10B981; font-weight: bold;">{total_elapsed_processing:.3f}s</span>
    </div>
    \"\"\", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


all_tracks_final = tracker.get_all_tracks(min_persistence=min_persistence_show)

# Final lesion table
if all_tracks_final:
    st.markdown('<div class="module-header">FINAL CANDIDATE REPORT</div>', unsafe_allow_html=True)
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

st.markdown(\"\"\"
<div style="background: #3d1a1a; border-left: 4px solid #F59E0B; padding: 15px; border-radius: 4px; margin-top: 30px;">
    <div style="color: #F59E0B; font-weight: 700; margin-bottom: 8px;">⚠ RESEARCH PROTOTYPE / LIMITATIONS</div>
    <div style="color: #ff9494; font-size: 0.85rem; line-height: 1.5;">
        AI model scores are prototype classification outputs. They are not calibrated clinical probabilities, diagnostic confidence values, or stenosis percentages.<br><br>
        Overlay visibility filtering is for visualization clarity only; it is not a clinical detection threshold.
    </div>
</div>
\"\"\", unsafe_allow_html=True)
"""
    
    # Remove everything from # ================================================================== #
    # POST-PROCESSING SUMMARY to the end
    content = re.sub(r'# ================================================================== #\n# POST-PROCESSING SUMMARY.*?$', post_proc_str, content, flags=re.DOTALL)

    # Remove the intermediate dividers and captions
    content = re.sub(r'st\.divider\(\)\n\n# ---- Live metrics row ----', '# ---- Live metrics row ----', content, flags=re.DOTALL)
    content = re.sub(r'st\.divider\(\)\n\n# ---- Lesion summary table ----.*?table_placeholder = st\.empty\(\)\n\nst\.divider\(\)\n\n# ---- Temporal persistence explanation ----.*?\"\"\")', '', content, flags=re.DOTALL)


    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

rewrite()
