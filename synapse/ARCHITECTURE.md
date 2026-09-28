# ARCHITECTURE.md — Synapse 1.0 (90-Minute MVP)

## System Overview

```
Video/Frames → Preprocessor → Detector → Tracker → Visualizer → Streamlit UI
                                              ↓
                                     Confidence Engine
```

## Pipeline (Single Thread, Sequential)

1. **Preprocessor** — Extract frames from video or load image sequence. Grayscale + CLAHE + Gaussian blur.
2. **Detector** — OpenCV-based candidate region detection (heuristic, NOT a trained model). Adaptive threshold → morphological ops → contour analysis → filter by area/aspect ratio. Outputs list of `Detection(bbox, raw_score, frame_idx)`. The `raw_score` is a heuristic detection score, NOT an ML probability.
3. **Tracker** — IoU-based frame-to-frame association. Maintains `Lesion` objects with: `lesion_id, first_frame, last_frame, persistence_count, score_history[]`. Persistence ratio = `persistence_count / (last_frame - first_frame + 1)`.
4. **Confidence Engine** — `displayed_confidence = 0.5 * mean(score_history) + 0.5 * persistence_ratio`. Labeled: "Prototype confidence — combines image-based candidate score and temporal persistence."
5. **Visualizer** — Draw bboxes, lesion IDs, confidence bars on frames. Color-coded by confidence level.
6. **Streamlit UI** — Video upload OR sample data. Frame-by-frame playback with annotations. Side panel: lesion table, confidence chart, persistence stats.

## Key Data Structures

```python
@dataclass
class Detection:
    bbox: tuple[int,int,int,int]  # x,y,w,h
    raw_score: float              # heuristic detection score (NOT ML probability)
    frame_idx: int

@dataclass
class Lesion:
    lesion_id: int
    detections: list[Detection]
    first_frame: int
    last_frame: int
    persistence_count: int
    score_history: list[float]    # history of raw_score values
```

## File Structure

```
synapse/
├── app.py                    # Streamlit entry point
├── pipeline/
│   ├── __init__.py
│   ├── preprocessor.py       # Frame extraction + enhancement
│   ├── detector.py           # OpenCV candidate detector
│   ├── tracker.py            # IoU temporal tracker
│   └── confidence.py         # Displayed confidence formula
├── utils/
│   ├── __init__.py
│   ├── visualization.py      # Annotation rendering
│   └── sample_data.py        # Synthetic test frames
├── requirements.txt
├── AGENTS.md
├── ARCHITECTURE.md
├── IMPLEMENTATION_STATUS.md
└── TODO.md
```

## What is CUT (from 8-hour blueprint)

| Feature | Status | Reason |
|---|---|---|
| YOLO fine-tuning | CUT | No time, no labeled data ready |
| PDF report generation | DEFERRED to post-90min | Nice-to-have |
| CADICA dataset integration | DEFERRED | Use synthetic frames first |
| Multi-view analysis | CUT | Complexity |
| Clinical validation metrics | CUT | Cannot fabricate |
| Database/persistence layer | CUT | Unnecessary for demo |
| Authentication/multi-user | CUT | Unnecessary |
| Advanced analytics dashboard | DEFERRED | Post-90min polish |
| DICOM support | CUT | Complexity |

## Terminology & Honesty Rules

| Term in Code/UI | Meaning |
|---|---|
| **Detection Score** / **Candidate Region Score** | Heuristic value from OpenCV morphological analysis. NOT a trained ML probability. |
| **Suspicious Candidate** | A region flagged by the OpenCV heuristic. NOT a confirmed stenosis. |
| **Temporal Persistence** | How many consecutive frames a candidate region persists (IoU-tracked). |
| **Displayed Confidence** | `0.5 * mean(score_history) + 0.5 * persistence_ratio`. Must be labeled: "Prototype confidence — combines image-based candidate score and temporal persistence." |

**Prohibited terms in UI/code comments:**
- "AI confidence" (for OpenCV outputs)
- "Model probability"
- "Clinically validated"
- "Stenosis detection accuracy"

## Medical Disclaimer (Must appear in UI)

> ⚠️ RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE. Not clinically validated. Do not use for clinical decisions. Detection scores are heuristic, not trained ML outputs.
