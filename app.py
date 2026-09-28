"""Synapse — Streamlit entrypoint.

Phase 0 placeholder only: confirms the app boots. Live-view UI, file
upload, and pipeline wiring (video_loader -> detector -> tracker ->
visualization -> report) will be added in later phases.
"""

import streamlit as st

st.set_page_config(page_title="Synapse — Coronary Blockage Detection", layout="wide")

st.title("Synapse")
st.caption("Real-Time Coronary Blockage Detection — Phase 0 skeleton")

st.info(
    "Pipeline not implemented yet. This placeholder confirms the app "
    "boots correctly. See README.md for the planned architecture."
)
