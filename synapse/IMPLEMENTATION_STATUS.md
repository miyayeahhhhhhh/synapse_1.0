| Component | Status | Owner | Notes |
|---|---|---|---|
| `app.py` | ❌ Not started | Sonnet | Sprint 3 |
| `pipeline/preprocessor.py` | ✅ Done | Sonnet | CLAHE+blur; used for display |
| `pipeline/detector.py` | ✅ Done | Sonnet | Vessel-mask + narrowing range; raw frame |
| `pipeline/tracker.py` | ✅ Done | Sonnet | IoU greedy match, max_missing=3, iou_thresh=0.15 |
| `pipeline/confidence.py` | ✅ Done | Sonnet | 0.5*raw+0.5*persist, labeled |
| `utils/visualization.py` | ✅ Done | Sonnet | OpenCV overlay, disclaimer, color bands |
| `utils/sample_data.py` | ✅ Done | Sonnet | 60 frames, 3 narrowing candidates |
| `requirements.txt` | ✅ Done | Gemini Pro | opencv-python-headless, numpy, pandas |

**Last updated:** Sprint 2 complete. Tests 1-7 all PASSED.
**Next:** Sprint 3 — app.py (Streamlit MVP)
