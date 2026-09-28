"""
pipeline/detector.py
--------------------
Candidate region detector using OpenCV image processing heuristics.

IMPORTANT TERMINOLOGY:
- This is a heuristic image analysis pipeline, NOT a trained machine learning model.
- Outputs are "detection scores" or "candidate region scores".
- DO NOT refer to outputs as "clinical probability", "stenosis probability",
  or "validated confidence".
- Candidate regions are flagged based on observable image characteristics:
  regions of intermediate brightness embedded within brighter vessel channels.
- A flagged region is a "suspicious candidate", NOT a confirmed stenosis.

Detection pipeline:
  1. Vessel mask — threshold to isolate bright vessel channels (>100 intensity).
  2. Mask dilation — extend mask 5 iterations to cover the interior of
     narrowing regions (which appear as intermediate-brightness patches
     within the vessel). 
  3. Intermediate-brightness filter — within the dilated vessel mask, find
     pixels in a "narrowing range" [30, 130]: above background but clearly
     below vessel brightness. This targets the observable narrowing signal.
  4. Morphological filtering — close gaps, remove noise.
  5. Contour extraction — find connected candidate blobs.
  6. Area filter — reject blobs too small or too large.
  7. Aspect-ratio filter — reject geometrically implausible blobs.
  8. Score computation — heuristic score from contrast vs. vessel + size.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List


# ------------------------------------------------------------------ #
# Configuration constants (all tuned for synthetic angiography frames)
# ------------------------------------------------------------------ #

# Vessel channel threshold: pixels above this are considered vessel
# (synthetic vessels ~190-210; background ~18)
_VESSEL_MIN_INTENSITY = 100

# Dilation iterations to extend vessel mask into narrowing ellipse interiors
_VESSEL_DILATE_ITERS = 5
_VESSEL_DILATE_KERNEL = (5, 5)

# Intermediate-brightness range that indicates a narrowing within a vessel
# (synthetic narrowings painted at ~55-65; above background ~18, below vessel ~190)
_NARROWING_LOW = 30     # pixels below this are background — excluded
_NARROWING_HIGH = 130   # pixels above this are still-bright vessel — excluded

# Morphological ops on the candidate binary mask
_CLOSE_KERNEL_SIZE = (5, 5)
_OPEN_KERNEL_SIZE  = (3, 3)

# Contour area bounds (pixels²)
_MIN_CONTOUR_AREA = 60
_MAX_CONTOUR_AREA = 3000

# Aspect-ratio bound (long/short, where long >= short)
_MAX_ASPECT = 8.0

# Minimum raw score to keep
_MIN_RAW_SCORE = 0.25

# Reference values for score computation
_VESSEL_REFERENCE_INTENSITY = 190.0   # typical bright vessel pixel
_REFERENCE_AREA = 350.0               # reference contour area for size scoring


# ------------------------------------------------------------------ #
# Data structure
# ------------------------------------------------------------------ #

@dataclass
class Detection:
    """
    A single candidate region detected in one frame.

    Attributes
    ----------
    bbox : tuple[int, int, int, int]
        Bounding box as (x, y, w, h) in pixel coordinates.
    raw_score : float
        Heuristic detection score in [0.0, 1.0].
        Derived from brightness contrast within vessel region and region size.
        NOT an ML model probability. NOT a clinical measurement.
    frame_idx : int
        Index of the frame this detection was found in.
    """
    bbox: tuple        # (x, y, w, h)
    raw_score: float   # heuristic candidate region score — NOT ML probability
    ml_score: float    # AI model score from CADICA classifier
    frame_idx: int


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _build_vessel_mask(frame: np.ndarray) -> np.ndarray:
    """
    Identify vessel channel pixels using a fixed absolute threshold, then
    dilate to extend the mask into the interior of narrowing regions
    (which would otherwise be excluded by the brightness threshold).
    """
    _, bright_mask = cv2.threshold(
        frame, _VESSEL_MIN_INTENSITY, 255, cv2.THRESH_BINARY
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _VESSEL_DILATE_KERNEL)
    vessel_mask = cv2.dilate(bright_mask, kernel, iterations=_VESSEL_DILATE_ITERS)
    return vessel_mask


def _find_narrowing_candidates(
    frame: np.ndarray,
    vessel_mask: np.ndarray,
) -> np.ndarray:
    """
    Within the vessel mask, find pixels in the 'narrowing brightness range':
    above background (< _NARROWING_LOW excluded) but clearly below vessel
    brightness (>= _NARROWING_HIGH excluded).

    This range targets observable narrowing regions — areas where vessel
    brightness dips significantly without falling to background level.

    Returns a binary mask where candidate pixels = 255.
    """
    candidate_mask = np.zeros_like(frame)
    candidate_mask[
        (frame >= _NARROWING_LOW) &
        (frame < _NARROWING_HIGH) &
        (vessel_mask > 0)
    ] = 255

    # Morphological close: fill small gaps inside candidates
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _CLOSE_KERNEL_SIZE)
    closed = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, close_k)

    # Morphological open: remove isolated noise specks
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _OPEN_KERNEL_SIZE)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_k)

    return opened


def _compute_raw_score(roi_mean: float, contour_area: float) -> float:
    """
    Heuristic candidate region score based on two observable characteristics:

    1. Brightness contrast score: how far below vessel brightness is the
       candidate ROI? A larger drop from vessel reference indicates a more
       pronounced narrowing signal.
         contrast = clip((vessel_ref - roi_mean) / (vessel_ref - narrowing_low), 0, 1)

    2. Size score: reward regions of moderate size near _REFERENCE_AREA.
       Very small (noise) or very large (artefact) regions score lower.

    Final: 0.65 * contrast + 0.35 * size
    Clamped to [0.0, 1.0].

    This formula is transparent, purely image-based, and NOT clinically calibrated.
    """
    contrast = float(np.clip(
        (_VESSEL_REFERENCE_INTENSITY - roi_mean) /
        (_VESSEL_REFERENCE_INTENSITY - _NARROWING_LOW + 1e-6),
        0.0, 1.0
    ))
    if contour_area > 0:
        log_ratio  = abs(np.log(max(contour_area, 1.0) / _REFERENCE_AREA))
        max_log    = np.log(_MAX_CONTOUR_AREA / _REFERENCE_AREA + 1e-6)
        size_score = float(np.clip(1.0 - log_ratio / (max_log + 1e-6), 0.0, 1.0))
    else:
        size_score = 0.0

    raw_score = 0.65 * contrast + 0.35 * size_score
    return float(np.clip(raw_score, 0.0, 1.0))


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def detect_candidates(
    frame: np.ndarray,
    frame_idx: int,
    ml_model=None,
) -> List[Detection]:
    """
    Identify suspicious candidate regions in a grayscale frame.

    Expects the raw (or lightly preprocessed) frame WITHOUT aggressive
    CLAHE enhancement, which can wash out the vessel-narrowing contrast.

    Uses OpenCV vessel masking + intermediate-brightness detection + morphological ops.
    This is a heuristic image processing pipeline — NOT a trained AI model.

    Parameters
    ----------
    frame : np.ndarray
        Uint8 grayscale frame (H, W).
    frame_idx : int
        Index of this frame in the sequence (populates Detection.frame_idx).

    Returns
    -------
    List[Detection]
        Candidate regions passing all filters, sorted by raw_score descending.
        Each Detection carries a heuristic `raw_score` in [0, 1].
        NOT a clinical probability. May return empty list.
    """
    if frame.ndim != 2:
        raise ValueError(
            f"detect_candidates expects a 2D grayscale frame, got shape {frame.shape}"
        )

    h, w = frame.shape

    # ---- Steps 1–2: Vessel mask with dilation ----
    vessel_mask = _build_vessel_mask(frame)

    # ---- Steps 3–4: Find narrowing candidates ----
    candidate_mask = _find_narrowing_candidates(frame, vessel_mask)

    if candidate_mask.max() == 0:
        return []

    # ---- Step 5: Contour extraction ----
    contours, _ = cv2.findContours(
        candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections: List[Detection] = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # ---- Step 6: Area filter ----
        if not (_MIN_CONTOUR_AREA <= area <= _MAX_CONTOUR_AREA):
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        # Guard: bbox fully inside image
        x  = int(np.clip(x,      0, w - 1))
        y  = int(np.clip(y,      0, h - 1))
        bw = int(np.clip(bw, 1, w - x))
        bh = int(np.clip(bh, 1, h - y))

        # ---- Step 7: Aspect-ratio filter ----
        long_side  = max(bw, bh)
        short_side = min(bw, bh)
        if short_side == 0:
            continue
        aspect = long_side / short_side
        if aspect > _MAX_ASPECT:
            continue

        # ---- Step 8: Score ----
        roi = frame[y : y + bh, x : x + bw]
        roi_mean = float(roi.mean()) if roi.size > 0 else _VESSEL_REFERENCE_INTENSITY

        raw_score = _compute_raw_score(roi_mean, area)

        if raw_score < _MIN_RAW_SCORE:
            continue

        ml_score = 0.0
        if ml_model is not None and ml_model.is_loaded:
            ml_score = ml_model.predict_crop(roi)

        detections.append(Detection(
            bbox=(x, y, bw, bh),
            raw_score=raw_score,
            ml_score=ml_score,
            frame_idx=frame_idx,
        ))

    detections.sort(key=lambda d: d.raw_score, reverse=True)
    return detections


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from utils.sample_data import generate_synthetic_sequence
    from pipeline.preprocessor import preprocess_frame

    frames = generate_synthetic_sequence(n_frames=60)
    total_detections = 0
    frames_with_detections = 0
    all_scores = []

    for fi, frame in enumerate(frames):
        # NOTE: pass raw frame to detector (CLAHE-enhanced used for display only)
        dets = detect_candidates(frame, frame_idx=fi)
        total_detections += len(dets)
        if dets:
            frames_with_detections += 1
            all_scores.extend(d.raw_score for d in dets)
        for d in dets:
            x, y, bw, bh = d.bbox
            assert x >= 0 and y >= 0, f"bbox origin negative: {d.bbox}"
            assert x + bw <= frame.shape[1], f"bbox right overflow: {d.bbox}"
            assert y + bh <= frame.shape[0], f"bbox bottom overflow: {d.bbox}"
            assert 0.0 <= d.raw_score <= 1.0, f"raw_score out of [0,1]: {d.raw_score}"

    print(f"\n--- detector.py smoke test ---")
    print(f"Total frames:               60")
    print(f"Frames with detections:     {frames_with_detections}")
    print(f"Total candidate detections: {total_detections}")
    print(f"Avg detections/frame:       {total_detections/60:.2f}")
    if all_scores:
        print(f"Score range:                [{min(all_scores):.3f}, {max(all_scores):.3f}]")
        print(f"Score mean:                 {sum(all_scores)/len(all_scores):.3f}")

    print("\nSample detections per key frame:")
    for fi in [0, 5, 10, 15, 25, 30, 50, 55]:
        dets = detect_candidates(frames[fi], frame_idx=fi)
        active = []
        if 5 <= fi < 50: active.append("A")
        if 15 <= fi < 55: active.append("B")
        if fi < 30: active.append("C")
        det_str = f"{len(dets)} detection(s)" if dets else "0 detections"
        scores = [f"{d.raw_score:.3f}" for d in dets]
        print(f"  Frame {fi:2d} (active={active}): {det_str}, scores={scores}")

    assert frames_with_detections > 10, (
        f"Too few frames with detections: {frames_with_detections}"
    )
    # Check inactive frames have no detections (frames 55-59 all inactive)
    for fi in range(55, 60):
        dets = detect_candidates(frames[fi], frame_idx=fi)
        assert len(dets) == 0, f"False detections on fully-inactive frame {fi}: {len(dets)}"

    print("\ndetector.py smoke test: PASSED")
