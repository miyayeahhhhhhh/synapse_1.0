# AGENTS.md — Synapse 1.0

## Agent Assignments

### Claude Sonnet (PRIMARY IMPLEMENTER)
**Priority: Build the entire MVP pipeline in order.**

1. `app.py` — Streamlit app (entry point, UI, frame display, controls)
2. `pipeline/detector.py` — Frame-level candidate region detector (OpenCV heuristic, NOT a trained model)
3. `pipeline/tracker.py` — Temporal persistence tracker (IoU-based)
4. `pipeline/preprocessor.py` — Frame extraction + preprocessing
5. `pipeline/confidence.py` — Prototype displayed confidence (combines detection score + persistence)
6. `utils/visualization.py` — Bounding box + annotation overlay
7. `utils/sample_data.py` — Synthetic test frame generator (NO fake clinical data)

**Rules for Sonnet:**
- Build incrementally. Each file must work standalone before integration.
- `app.py` must show SOMETHING within 30 minutes (even static frame + dummy boxes).
- Use OpenCV morphological ops for candidate detection (NOT a trained model unless YOLO weight exists).
- OpenCV detector outputs are "candidate region scores" or "detection scores" — NEVER "AI confidence" or "model probability".
- Temporal tracker is the CORE innovation — implement it properly.
- The displayed confidence formula must be labeled: "Prototype confidence — combines image-based candidate score and temporal persistence."
- UI labels must use: "Suspicious Candidate", "Detection Score", "Temporal Persistence", "Displayed Confidence".
- NEVER call the OpenCV detector a clinically validated stenosis detector.
- NEVER claim synthetic test data demonstrates medical detection accuracy.

### Gemini Pro (REVIEW + ARCHITECTURE)
- Architecture docs, planning, code review.
- Post-90min: PDF report generation if time permits.

### Gemini Flash (SUPPORT)
- Quick fixes, formatting, docstrings.
- Post-90min: README polish, demo script.

### GPT-OSS (SUPPORT)
- Data pipeline helpers if needed.
- Post-90min: Additional visualizations.

## Critical Rules
- NO agent may fabricate clinical metrics, stenosis percentages, or model accuracy claims.
- NO agent may present OpenCV heuristic scores as trained ML model outputs or clinical probabilities.
- All scores from the OpenCV detector must be called "detection scores" or "candidate region scores", never "AI confidence".
