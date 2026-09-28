"""
pipeline/tracker.py
-------------------
IoU-based temporal persistence tracker for Synapse 1.0.

CORE INNOVATION:
Tracks candidate regions across consecutive frames using Intersection-over-Union
(IoU) bounding-box matching. Assigns persistent lesion IDs so that the same
suspicious region is recognized across many frames, building a "persistence"
signal that contributes to the displayed confidence.

TERMINOLOGY:
- A "lesion" here is a tracked candidate region, NOT a confirmed clinical lesion.
- "Persistence count" = number of frames the candidate was actually detected.
- "Displayed confidence" combines detection score + persistence — see confidence.py.

ALGORITHM OVERVIEW:
  For each frame:
    1. Compute IoU between every active track and every new detection.
    2. Greedily match highest-IoU pairs (one detection per track, one track
       per detection). Accept match only if IoU >= IOU_THRESHOLD.
    3. Matched tracks: update bbox, append raw_score, increment persistence_count,
       reset missing_frames counter.
    4. Unmatched detections: create new LesionTrack.
    5. Unmatched tracks: increment missing_frames counter. Close track if
       missing_frames > MAX_MISSING_FRAMES.

No learned model. No re-identification network. Pure deterministic geometry.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pipeline.detector import Detection
from pipeline.confidence import (
    compute_displayed_confidence,
    compute_displayed_confidence_from_history,
    DISPLAYED_CONFIDENCE_LABEL,
)


# ------------------------------------------------------------------ #
# Tracker configuration defaults
# ------------------------------------------------------------------ #

DEFAULT_IOU_THRESHOLD     = 0.15   # minimum IoU to accept a match
DEFAULT_MAX_MISSING_FRAMES = 3     # frames a track survives without a detection
DEFAULT_MIN_PERSISTENCE    = 1     # minimum persistence_count to include in final output


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class LesionTrack:
    """
    A persistent candidate region track across multiple frames.

    Terminology reminder:
    - 'lesion' = tracked suspicious candidate, NOT a confirmed clinical lesion.
    - raw_score values are heuristic detection scores, NOT ML probabilities.
    - displayed_confidence is a prototype heuristic blend, NOT clinical probability.

    Attributes
    ----------
    lesion_id : int
        Unique integer ID for this track. Assigned sequentially.
    first_frame : int
        Frame index when this track was created.
    last_frame : int
        Frame index of the most recent matched detection.
    persistence_count : int
        Number of frames in which a detection was matched to this track.
    score_history : list[float]
        Heuristic raw_score values from each matched detection.
    current_bbox : tuple[int,int,int,int]
        Bounding box (x,y,w,h) from the most recent matched detection.
    missing_frames : int
        How many consecutive frames have passed without a match.
        Internal; not intended for UI display.
    active : bool
        True while the track is still being updated. False after closure.
    """
    lesion_id: int
    first_frame: int
    last_frame: int
    persistence_count: int        = 0
    score_history: List[float]    = field(default_factory=list)
    ml_score_history: List[float] = field(default_factory=list)
    current_bbox: Optional[tuple] = None
    missing_frames: int           = 0
    active: bool                  = True

    # ---- Derived properties (recomputed on each access) ---- #

    @property
    def mean_raw_score(self) -> float:
        """Mean of heuristic detection scores. NOT an ML probability."""
        if not self.score_history:
            return 0.0
        return float(np.mean(self.score_history))

    @property
    def mean_ml_score(self) -> float:
        """Mean of AI model scores (CADICA prototype classifier)."""
        if not self.ml_score_history:
            return 0.0
        return float(np.mean(self.ml_score_history))

    @property
    def persistence_ratio(self) -> float:
        """
        Fraction of frames in the track's lifespan with an actual detection.
        persistence_ratio = persistence_count / lifespan
        Range [0, 1].
        """
        lifespan = self.last_frame - self.first_frame + 1
        if lifespan <= 0:
            return 0.0
        return float(np.clip(self.persistence_count / lifespan, 0.0, 1.0))

    @property
    def displayed_confidence(self) -> float:
        """
        Prototype confidence for this track.
        = 0.5 * mean_raw_score + 0.5 * persistence_ratio
        Computed via pipeline/confidence.py.

        NOT a clinical probability. Label as:
        "Prototype confidence — combines image-based candidate score
         and temporal persistence."
        """
        return compute_displayed_confidence(
            mean_raw_score=self.mean_raw_score,
            persistence_ratio=self.persistence_ratio,
        )

    def summary_dict(self) -> dict:
        """Return a flat dict suitable for DataFrame / Streamlit display."""
        return {
            "lesion_id":            self.lesion_id,
            "first_frame":          self.first_frame,
            "last_frame":           self.last_frame,
            "persistence_count":    self.persistence_count,
            "lifespan":             self.last_frame - self.first_frame + 1,
            "persistence_ratio":    round(self.persistence_ratio, 3),
            "mean_raw_score":       round(self.mean_raw_score, 3),
            "mean_ml_score":        round(self.mean_ml_score, 3),
            "active":               self.active,
            "current_bbox":         self.current_bbox,
            "confidence_label":     DISPLAYED_CONFIDENCE_LABEL,
        }


# ------------------------------------------------------------------ #
# IoU utility
# ------------------------------------------------------------------ #

def _compute_iou(bbox_a: tuple, bbox_b: tuple) -> float:
    """
    Compute Intersection-over-Union between two bounding boxes.

    Parameters
    ----------
    bbox_a, bbox_b : tuple[int,int,int,int]
        Each as (x, y, w, h).

    Returns
    -------
    float
        IoU in [0.0, 1.0]. Returns 0.0 for zero-area boxes.
    """
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    # Convert to (x1,y1,x2,y2)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    # Intersection
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return float(inter_area / union_area)


# ------------------------------------------------------------------ #
# Tracker
# ------------------------------------------------------------------ #

class TemporalTracker:
    """
    IoU-based temporal persistence tracker.

    Tracks suspicious candidate regions across frames, maintaining persistent
    lesion IDs and building a persistence signal for the confidence engine.

    Usage
    -----
    tracker = TemporalTracker()

    for frame_idx, frame in enumerate(frames):
        detections = detect_candidates(frame, frame_idx)
        active_tracks = tracker.update(frame_idx, detections)

    all_tracks = tracker.get_all_tracks()
    summary = tracker.get_summary()

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to accept a detection-to-track match. Default 0.15.
        Lower values are more permissive (tolerate more jitter).
    max_missing_frames : int
        Number of consecutive frames a track may go unmatched before closure.
        Default 3 — a brief flicker does not destroy a track.
    min_persistence : int
        Minimum persistence_count for a track to appear in the summary output.
        Default 1 (all tracks shown).
    """

    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_missing_frames: int = DEFAULT_MAX_MISSING_FRAMES,
        min_persistence: int = DEFAULT_MIN_PERSISTENCE,
    ) -> None:
        self.iou_threshold     = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.min_persistence   = min_persistence

        self._next_id: int = 1
        self._active_tracks: Dict[int, LesionTrack] = {}  # id -> track
        self._closed_tracks: List[LesionTrack] = []

    # ---- Public API ---- #

    def update(
        self,
        frame_idx: int,
        detections: List[Detection],
    ) -> List[LesionTrack]:
        """
        Process detections from one frame and update all tracks.

        Algorithm:
        1. Build IoU matrix: shape (n_active_tracks, n_detections).
        2. Greedy match: repeatedly take the highest IoU cell, assign
           that (track, detection) pair, mask out both row and column.
           Accept only if IoU >= iou_threshold.
        3. Matched tracks: update with detection data.
        4. Unmatched detections: spawn new tracks.
        5. Unmatched active tracks: increment missing_frames; close if exceeded.

        Returns
        -------
        List[LesionTrack]
            Currently active tracks after this update.
        """
        active_ids   = list(self._active_tracks.keys())
        active_tracks = [self._active_tracks[i] for i in active_ids]

        n_tracks = len(active_tracks)
        n_dets   = len(detections)

        matched_track_ids = set()
        matched_det_idxs  = set()

        if n_tracks > 0 and n_dets > 0:
            # Build IoU matrix
            iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
            for ti, track in enumerate(active_tracks):
                if track.current_bbox is None:
                    continue
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = _compute_iou(track.current_bbox, det.bbox)

            # Greedy matching: repeatedly pick highest IoU
            remaining = iou_matrix.copy()
            while True:
                if remaining.max() < self.iou_threshold:
                    break
                # Find best unmatched pair
                flat_idx = int(np.argmax(remaining))
                ti, di   = divmod(flat_idx, n_dets)

                track_id = active_ids[ti]
                matched_track_ids.add(track_id)
                matched_det_idxs.add(di)

                # Update track
                det = detections[di]
                track = self._active_tracks[track_id]
                track.last_frame        = frame_idx
                track.current_bbox      = det.bbox
                track.persistence_count += 1
                track.score_history.append(det.raw_score)
                track.ml_score_history.append(det.ml_score)
                track.missing_frames    = 0

                # Mask this row and column so they can't be reused
                remaining[ti, :] = -1.0
                remaining[:, di] = -1.0

        # Spawn new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det_idxs:
                new_track = LesionTrack(
                    lesion_id=self._next_id,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    persistence_count=1,
                    score_history=[det.raw_score],
                    ml_score_history=[det.ml_score],
                    current_bbox=det.bbox,
                    missing_frames=0,
                    active=True,
                )
                self._active_tracks[self._next_id] = new_track
                self._next_id += 1

        # Handle unmatched active tracks
        to_close = []
        for track_id in active_ids:
            if track_id not in matched_track_ids:
                track = self._active_tracks[track_id]
                track.missing_frames += 1
                if track.missing_frames > self.max_missing_frames:
                    to_close.append(track_id)

        for track_id in to_close:
            track = self._active_tracks.pop(track_id)
            track.active = False
            self._closed_tracks.append(track)

        return list(self._active_tracks.values())

    def get_all_tracks(self, min_persistence: Optional[int] = None) -> List[LesionTrack]:
        """
        Return all tracks (active + closed), optionally filtered by
        minimum persistence_count.

        Parameters
        ----------
        min_persistence : int, optional
            Override the instance-level min_persistence for this call.
        """
        threshold = min_persistence if min_persistence is not None else self.min_persistence
        all_tracks = self._closed_tracks + list(self._active_tracks.values())
        return [t for t in all_tracks if t.persistence_count >= threshold]

    def get_active_tracks(self) -> List[LesionTrack]:
        """Return only currently active (not yet closed) tracks."""
        return list(self._active_tracks.values())

    def get_summary(self, min_persistence: Optional[int] = None) -> List[dict]:
        """
        Return a list of summary dicts for all tracks, suitable for DataFrame
        or Streamlit table display.
        """
        return [t.summary_dict() for t in self.get_all_tracks(min_persistence)]

    def reset(self) -> None:
        """Reset tracker state. Call between independent sequences."""
        self._next_id = 1
        self._active_tracks.clear()
        self._closed_tracks.clear()


# ------------------------------------------------------------------ #
# Smoke tests
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from utils.sample_data import generate_synthetic_sequence
    from pipeline.detector import detect_candidates

    frames = generate_synthetic_sequence(n_frames=60)
    tracker = TemporalTracker(
        iou_threshold=0.15,
        max_missing_frames=3,
        min_persistence=2,
    )

    print("--- Running tracker on 60-frame synthetic sequence ---")
    for fi, frame in enumerate(frames):
        dets = detect_candidates(frame, frame_idx=fi)
        tracker.update(fi, dets)

    all_tracks = tracker.get_all_tracks(min_persistence=2)
    print(f"\nTracks with persistence >= 2: {len(all_tracks)}")
    print()

    for t in sorted(all_tracks, key=lambda x: x.lesion_id):
        print(f"Lesion #{t.lesion_id}:")
        print(f"  Frames:               {t.first_frame}-{t.last_frame}")
        print(f"  Lifespan:             {t.last_frame - t.first_frame + 1} frames")
        print(f"  Persistence count:    {t.persistence_count}")
        print(f"  Persistence ratio:    {t.persistence_ratio:.2f}")
        print(f"  Mean detection score: {t.mean_raw_score:.3f}")
        print(f"  Displayed confidence: {t.displayed_confidence:.1%}")
        print(f"  (Label: Prototype confidence - not clinical)")
        print()

    # ---- TEST 1: Same lesion ID across consecutive frames ----
    tracker1 = TemporalTracker(iou_threshold=0.15, max_missing_frames=3)
    from pipeline.detector import Detection
    dets_A = [Detection(bbox=(100, 100, 30, 20), raw_score=0.7, ml_score=0.0, frame_idx=0)]
    dets_B = [Detection(bbox=(101, 101, 30, 20), raw_score=0.72, ml_score=0.0, frame_idx=1)]
    dets_C = [Detection(bbox=(102, 100, 30, 20), raw_score=0.68, ml_score=0.0, frame_idx=2)]

    tracker1.update(0, dets_A)
    tracker1.update(1, dets_B)
    tracker1.update(2, dets_C)
    tracks = tracker1.get_all_tracks()
    assert len(tracks) == 1, f"TEST 1 FAIL: expected 1 track, got {len(tracks)}"
    assert tracks[0].persistence_count == 3, f"TEST 1 FAIL: expected persistence=3"
    print("[TEST 1 PASS] Same lesion ID across consecutive frames")

    # ---- TEST 2: Brief missing detection does NOT create new ID ----
    tracker2 = TemporalTracker(iou_threshold=0.15, max_missing_frames=3)
    tracker2.update(1, [Detection(bbox=(100, 100, 30, 20), raw_score=0.7,  ml_score=0.0, frame_idx=1)])
    tracker2.update(2, [Detection(bbox=(101, 101, 30, 20), raw_score=0.71, ml_score=0.0, frame_idx=2)])
    tracker2.update(3, [])  # missing frame
    tracker2.update(4, [Detection(bbox=(100, 100, 30, 20), raw_score=0.69, ml_score=0.0, frame_idx=4)])
    tracker2.update(5, [Detection(bbox=(101, 101, 30, 20), raw_score=0.72, ml_score=0.0, frame_idx=5)])
    tracks2 = tracker2.get_all_tracks()
    assert len(tracks2) == 1, f"TEST 2 FAIL: expected 1 track after flicker, got {len(tracks2)}"
    assert tracks2[0].persistence_count == 4, f"TEST 2 FAIL: expected persistence=4, got {tracks2[0].persistence_count}"
    print("[TEST 2 PASS] Brief missing frame does not create new lesion ID")

    # ---- TEST 3: Two spatially separated candidates get different IDs ----
    tracker3 = TemporalTracker(iou_threshold=0.15, max_missing_frames=3)
    dets_two = [
        Detection(bbox=(50,  50,  30, 20), raw_score=0.7, ml_score=0.0, frame_idx=0),
        Detection(bbox=(400, 300, 30, 20), raw_score=0.65, ml_score=0.0, frame_idx=0),
    ]
    tracker3.update(0, dets_two)
    tracks3 = tracker3.get_all_tracks()
    assert len(tracks3) == 2, f"TEST 3 FAIL: expected 2 separate tracks, got {len(tracks3)}"
    ids3 = {t.lesion_id for t in tracks3}
    assert len(ids3) == 2, f"TEST 3 FAIL: IDs not unique: {ids3}"
    print("[TEST 3 PASS] Two spatially separated candidates get different lesion IDs")

    # ---- TEST 4: Genuinely new spatial candidate gets new ID ----
    tracker4 = TemporalTracker(iou_threshold=0.15, max_missing_frames=3)
    tracker4.update(0, [Detection(bbox=(100, 100, 30, 20), raw_score=0.7, ml_score=0.0, frame_idx=0)])
    tracker4.update(1, [Detection(bbox=(100, 100, 30, 20), raw_score=0.71, ml_score=0.0, frame_idx=1)])
    # Completely new location
    tracker4.update(2, [Detection(bbox=(400, 300, 30, 20), raw_score=0.65, ml_score=0.0, frame_idx=2)])
    all4 = tracker4.get_all_tracks()
    # Track 1 was closed (missing > threshold after 3 missing frames — but only 1 missing here)
    # Actually: track 1 will still be active with missing_frames=1; we'll have 2 active
    active4 = tracker4.get_active_tracks()
    ids4 = {t.lesion_id for t in all4}
    assert len(ids4) >= 2, f"TEST 4 FAIL: new location should create new ID, got ids={ids4}"
    print("[TEST 4 PASS] New spatial candidate creates a new lesion ID")

    # ---- TEST 5: Persistence count increases correctly ----
    tracker5 = TemporalTracker(iou_threshold=0.15, max_missing_frames=3)
    for fi in range(10):
        tracker5.update(fi, [Detection(bbox=(100, 100, 30, 20), raw_score=0.7, ml_score=0.0, frame_idx=fi)])
    tracks5 = tracker5.get_all_tracks()
    assert len(tracks5) == 1
    assert tracks5[0].persistence_count == 10, f"TEST 5 FAIL: expected 10, got {tracks5[0].persistence_count}"
    print("[TEST 5 PASS] Persistence count increases correctly (10 frames -> 10)")

    # ---- TEST 6: Displayed confidence uses pipeline/confidence.py ----
    tracker6 = TemporalTracker(iou_threshold=0.15, max_missing_frames=3)
    for fi in range(5):
        tracker6.update(fi, [Detection(bbox=(100, 100, 30, 20), raw_score=0.7, ml_score=0.0, frame_idx=fi)])
    t6 = tracker6.get_all_tracks()[0]
    expected = compute_displayed_confidence(t6.mean_raw_score, t6.persistence_ratio)
    assert abs(t6.displayed_confidence - expected) < 1e-6, "TEST 6 FAIL: confidence mismatch"
    assert 0.0 <= t6.displayed_confidence <= 1.0
    print(f"[TEST 6 PASS] Displayed confidence = {t6.displayed_confidence:.1%} (via confidence.py)")

    print("\ntracker.py smoke test: ALL TESTS PASSED")
