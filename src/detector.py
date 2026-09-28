"""Stenosis detection.

Wraps a YOLO-family model (via ultralytics, or an exported ONNX model run
through onnxruntime) to detect suspicious arterial narrowing / blockages
in a single frame, and applies false-positive suppression on top of the
raw detections.

Not implemented yet (Phase 0: skeleton only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Detection:
    """A single detected region of suspected stenosis in one frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0


class StenosisDetector:
    """Placeholder for the YOLO-family stenosis detector."""

    def __init__(self, weights_path: str | Path) -> None:
        self.weights_path = Path(weights_path)

    def predict(self, frame) -> list[Detection]:
        raise NotImplementedError("StenosisDetector.predict is not implemented yet.")


def suppress_false_positives(detections: list[Detection]) -> list[Detection]:
    """Filter raw detections to remove likely false positives.

    Not implemented yet.
    """
    raise NotImplementedError("suppress_false_positives is not implemented yet.")
