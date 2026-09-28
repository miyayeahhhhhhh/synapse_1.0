"""
utils/visualization.py
-----------------------
OpenCV-based annotation overlay for Synapse 1.0.

Renders bounding boxes, lesion IDs, prototype confidence, and persistence
counts onto grayscale frames for display in the Streamlit app.

TERMINOLOGY:
- Boxes are labeled "Lesion #N" as shorthand for "tracked candidate #N".
  This is NOT a clinical diagnosis label.
- Confidence bands (color) reflect prototype displayed confidence only.
  Color does NOT imply clinical severity.
- The disclaimer "Prototype — Not for Clinical Use" must appear on every frame.

DESIGN:
- Output is always a BGR uint8 array suitable for cv2.imshow or st.image.
- Keeps OpenCV only — no matplotlib dependency.
- Lightweight: ~1 ms per frame for typical candidate counts.
"""

import cv2
import numpy as np
from typing import List, Optional

from pipeline.tracker import LesionTrack


# ------------------------------------------------------------------ #
# Visual style constants
# ------------------------------------------------------------------ #

# Confidence band colors (BGR) — for visual differentiation ONLY.
# Colors do NOT encode clinical severity.
# Low / mid / high refer only to the prototype confidence value.
_COLOR_HIGH  = (0,   200,  80)    # green  — displayed_confidence >= 0.65
_COLOR_MID   = (0,   180, 220)    # amber  — 0.40 <= displayed_confidence < 0.65
_COLOR_LOW   = (80,   80, 200)    # muted  — displayed_confidence < 0.40

_COLOR_DISCLAIMER = (180, 180, 180)  # gray disclaimer text
_COLOR_LABEL_BG   = (20,  20,  20)  # dark label background

# Font settings
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_LABEL  = 0.42
_FONT_SCALE_SMALL  = 0.36
_FONT_SCALE_DISC   = 0.38
_FONT_THICK = 1
_BOX_THICK  = 2

# Disclaimer string (required on every annotated frame)
_DISCLAIMER = "Prototype - Not for Clinical Use"


# ------------------------------------------------------------------ #
# Color helper
# ------------------------------------------------------------------ #

def _confidence_color(mean_ml_score: float) -> tuple:
    """
    Map AI model score to a BGR color for visualization.

    Color bands:
      >= 0.65  ->  green  (high prototype confidence)
      >= 0.40  ->  amber  (medium)
      < 0.40   ->  muted  (low)

    IMPORTANT: Color bands reflect the PROTOTYPE HEURISTIC SCORE only.
    They do NOT indicate clinical severity or stenosis risk.
    """
    if mean_ml_score >= 0.65:
        return _COLOR_HIGH
    elif mean_ml_score >= 0.40:
        return _COLOR_MID
    else:
        return _COLOR_LOW


# ------------------------------------------------------------------ #
# Main annotation function
# ------------------------------------------------------------------ #

def annotate_frame(
    frame: np.ndarray,
    tracks: List[LesionTrack],
    frame_idx: Optional[int] = None,
    show_disclaimer: bool = True,
) -> np.ndarray:
    """
    Render candidate region annotations onto a grayscale or BGR frame.

    Draws for each active track:
      - Bounding box (color-coded by AI score)
      - "LESION #N" label
      - AI model score percentage
      - Persistence count

    Also overlays a mandatory disclaimer banner.

    Parameters
    ----------
    frame : np.ndarray
        Input frame, uint8. Accepts grayscale (H,W) or BGR (H,W,3).
    tracks : List[LesionTrack]
        Tracks to annotate. Only active tracks with a current_bbox are drawn.
    frame_idx : int, optional
        Frame number to display in the corner.
    show_disclaimer : bool
        Whether to render the "Prototype — Not for Clinical Use" banner.
        Default True; set False only for unit tests.

    Returns
    -------
    np.ndarray
        Annotated BGR uint8 frame, same spatial dimensions as input.
    """
    # Ensure BGR output
    if frame.ndim == 2:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        vis = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        vis = frame.copy()

    h, w = vis.shape[:2]

    # ---- Draw each track ---- #
    for track in tracks:
        if track.current_bbox is None:
            continue
        if not track.active:
            continue
            
        conf  = track.mean_ml_score
        
        # UI Visualization filter: hide extremely low-confidence transient candidates
        # We compute the exact rounded percentage as displayed on screen
        # Retain if AI score >= 10% OR persistence >= 3 frames
        displayed_pct = int(round(conf * 100))
        if displayed_pct < 10 and track.persistence_count < 3:
            continue

        x, y, bw, bh = track.current_bbox
        color = _confidence_color(conf)

        # Bounding box
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, _BOX_THICK)

        # Label lines
        line1 = f"LESION #{track.lesion_id}"
        line2 = f"AI Score: {conf:.0%}  Persist: {track.persistence_count}f"

        # Measure text to size the background box
        (tw1, th1), bl1 = cv2.getTextSize(line1, _FONT, _FONT_SCALE_LABEL, _FONT_THICK)
        (tw2, th2), bl2 = cv2.getTextSize(line2, _FONT, _FONT_SCALE_SMALL, _FONT_THICK)

        pad = 4
        box_w = max(tw1, tw2) + 2 * pad
        box_h = th1 + th2 + 3 * pad

        # Position label above bbox, clamped to image bounds
        lx = int(np.clip(x, 0, w - box_w))
        ly = int(np.clip(y - box_h - 2, 0, h - box_h))

        # Dark background rectangle
        cv2.rectangle(
            vis,
            (lx, ly),
            (lx + box_w, ly + box_h),
            _COLOR_LABEL_BG,
            cv2.FILLED,
        )
        # Colored top-border stripe to match bbox color
        cv2.rectangle(
            vis,
            (lx, ly),
            (lx + box_w, ly + 3),
            color,
            cv2.FILLED,
        )

        # Text: line 1
        cv2.putText(
            vis, line1,
            (lx + pad, ly + pad + th1 + 2),
            _FONT, _FONT_SCALE_LABEL,
            color, _FONT_THICK, cv2.LINE_AA,
        )
        # Text: line 2
        cv2.putText(
            vis, line2,
            (lx + pad, ly + pad + th1 + th2 + pad + 1),
            _FONT, _FONT_SCALE_SMALL,
            (210, 210, 210), _FONT_THICK, cv2.LINE_AA,
        )

    # ---- Frame index ---- #
    if frame_idx is not None:
        fi_str = f"Frame {frame_idx}"
        cv2.putText(
            vis, fi_str,
            (8, h - 10),
            _FONT, _FONT_SCALE_SMALL,
            (140, 140, 140), _FONT_THICK, cv2.LINE_AA,
        )

    # ---- Mandatory disclaimer banner ---- #
    if show_disclaimer:
        (dw, dh), _ = cv2.getTextSize(_DISCLAIMER, _FONT, _FONT_SCALE_DISC, _FONT_THICK)
        dx = w - dw - 8
        dy = h - 10
        # Subtle dark backing
        cv2.rectangle(
            vis,
            (dx - 4, dy - dh - 4),
            (dx + dw + 4, dy + 4),
            (10, 10, 10),
            cv2.FILLED,
        )
        cv2.putText(
            vis, _DISCLAIMER,
            (dx, dy),
            _FONT, _FONT_SCALE_DISC,
            _COLOR_DISCLAIMER, _FONT_THICK, cv2.LINE_AA,
        )

    return vis


# ------------------------------------------------------------------ #
# Smoke test
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from utils.sample_data import generate_synthetic_sequence
    from pipeline.detector import detect_candidates
    from pipeline.tracker import TemporalTracker

    frames = generate_synthetic_sequence(n_frames=60)
    tracker = TemporalTracker(iou_threshold=0.15, max_missing_frames=3)

    annotated_count = 0
    annotated_with_tracks = 0

    for fi, frame in enumerate(frames):
        dets   = detect_candidates(frame, frame_idx=fi)
        active = tracker.update(fi, dets)

        ann = annotate_frame(frame, active, frame_idx=fi)

        # Validate output
        assert ann.ndim == 3, f"Output must be 3-channel BGR, got ndim={ann.ndim}"
        assert ann.shape[2] == 3, f"Output must be BGR (3 channels)"
        assert ann.shape[:2] == frame.shape[:2], "Spatial dims changed"
        assert ann.dtype == np.uint8, "Output must be uint8"

        annotated_count += 1
        if active:
            annotated_with_tracks += 1

    print(f"\n--- visualization.py smoke test ---")
    print(f"Annotated frames:            {annotated_count}")
    print(f"Frames with active tracks:   {annotated_with_tracks}")
    print(f"Output shape:                {ann.shape}")
    print(f"Output dtype:                {ann.dtype}")

    # TEST 7: Visualization renders without crashing
    assert annotated_count == 60
    print("\n[TEST 7 PASS] Visualization renders all 60 frames without crashing")
    print("\nvisualization.py smoke test: PASSED")
