"""Temporal tracking — the project's core innovation.

Links per-frame stenosis detections across frames using IoU-based
association, maintaining a persistence count and a confidence curve per
tracked lesion so that transient false positives can be distinguished
from a persistently-detected blockage.

Not implemented yet (Phase 0: skeleton only).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection


@dataclass
class Track:
    """A stenosis candidate tracked across consecutive frames."""

    track_id: int
    detections: list[Detection] = field(default_factory=list)
    confidence_curve: list[float] = field(default_factory=list)
    persistence: int = 0


class IoUTracker:
    """Placeholder for the IoU-based temporal tracker."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        raise NotImplementedError("IoUTracker.update is not implemented yet.")


def compute_iou(box_a, box_b) -> float:
    """Compute intersection-over-union between two boxes.

    Not implemented yet.
    """
    raise NotImplementedError("compute_iou is not implemented yet.")
