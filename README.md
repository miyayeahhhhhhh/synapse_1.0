# Synapse 1.0 — Real-Time Coronary Blockage Detection

**Hackathon project:** Synapse 1.0

**Problem statement:** AI that analyzes live coronary angiography video to
detect and highlight suspicious arterial narrowing or blockages with
confidence scores.

## Status

> **Phase 0 — repository skeleton.** No detection, tracking, or UI logic
> is implemented yet. This commit only sets up structure, dependencies,
> and placeholder tests so later phases have a clean base to build on.

## Architecture

```
video/frame loader
        │
        ▼
YOLO-family stenosis detector
        │
        ▼
false-positive suppression
        │
        ▼
IoU-based temporal tracker            ◄── core innovation
(persistence count + confidence curve
 per tracked lesion across frames)
        │
        ▼
visualization overlay
        │
        ▼
Streamlit live view ──► PDF report
```

1. **Video/frame loader** — reads either a video file or a directory of
   sequential CADICA-style PNG frames (with per-frame bounding boxes and
   severity labels, when available) and feeds frames into the pipeline.
2. **YOLO-family stenosis detector** — a YOLO model (via `ultralytics`,
   or an exported model run through `onnxruntime`) detects candidate
   regions of arterial narrowing in each frame.
3. **False-positive suppression** — filters raw per-frame detections
   before they reach the tracker.
4. **IoU-based temporal tracker (core innovation)** — links detections
   across consecutive frames by IoU, maintaining a **persistence count**
   and a **confidence curve** per tracked lesion. A blockage that is
   only briefly detected is treated very differently from one that
   persists and strengthens across many frames.
5. **Visualization overlay** — draws tracked boxes, confidence, and
   persistence indicators onto frames.
6. **Streamlit live view** — displays the annotated stream interactively.
7. **PDF report** — summarizes a completed run (tracks, confidence
   curves, representative frames) into a shareable report via `fpdf2`.

## Dataset

[CADICA](https://data.mendeley.com/datasets/p9bpx9ctcv/1) — sequential
PNG angiography frames with per-frame bounding boxes and severity labels.
Expected locally under `data/cadica/` (git-ignored; not checked in).

## Project layout

```
synapse/
├── app.py                  # Streamlit entrypoint (placeholder)
├── requirements.txt
├── src/
│   ├── video_loader.py     # frame/video loading (CADICA + video files)
│   ├── detector.py         # YOLO detector + false-positive suppression
│   ├── tracker.py          # IoU-based temporal tracker
│   ├── visualization.py    # overlay drawing
│   └── report.py           # PDF report generation
├── tests/
│   └── test_placeholder.py
└── data/
    └── cadica/              # dataset goes here (git-ignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

## Testing

```bash
pytest
```

## Roadmap

- **Phase 0 (this commit):** repo skeleton, dependencies, placeholder app + tests.
- **Phase 1:** frame/video loader + CADICA dataset integration.
- **Phase 2:** YOLO stenosis detector + false-positive suppression.
- **Phase 3:** IoU-based temporal tracker (persistence + confidence curve).
- **Phase 4:** visualization overlay + Streamlit live view.
- **Phase 5:** PDF report generation.
