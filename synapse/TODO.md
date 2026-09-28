# TODO — Synapse 1.0

## Phase 1: MVP (0–90 min) — MUST HAVE

### Sprint 1 (0–35 min): Foundation + Detection
- [x] `requirements.txt` -- streamlit, opencv-python-headless, numpy, pandas
- [x] `pipeline/__init__.py`, `utils/__init__.py`
- [x] `utils/sample_data.py` -- 60 synthetic frames, 3 narrowing candidates (NOT clinical data)
- [x] `pipeline/preprocessor.py` -- grayscale, CLAHE, blur (used for display)
- [x] `pipeline/detector.py` -- vessel-mask + intermediate-brightness filter (operates on RAW frame)
- [x] `pipeline/confidence.py` -- 0.5*mean_raw + 0.5*persistence_ratio, labeled as prototype

### Sprint 2 (35–60 min): Tracking + Visualization
- [x] `pipeline/tracker.py` -- IoU greedy match, iou_thresh=0.15, max_missing=3
- [x] `utils/visualization.py` -- OpenCV overlay; bbox, ID, confidence%, persistence, disclaimer

### Sprint 3 (60–90 min): Streamlit App + Demo
- [ ] `app.py` — full Streamlit UI: upload/sample, playback, annotations, lesion table
- [ ] Integration test with sample data
- [ ] Medical disclaimer in UI
- [ ] Demo stabilization + bug fixes

## Phase 2: Post-Review Polish (90–150 min) — NICE TO HAVE
- [ ] CADICA dataset loader
- [ ] PDF summary report (ReportLab)
- [ ] Confidence history chart (Plotly)
- [ ] README.md with screenshots
- [ ] Review-driven fixes
