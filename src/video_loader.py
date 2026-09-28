"""Video / frame loading — Phase 2.

Reads a coronary angiography sequence from any of three sources and
yields frames **in order**, each as a `FrameRecord` carrying the frame's
index, its pixel data (as a numpy array), and a basic image-quality
verdict:

- a Phase 1 video JSON file (see `scripts/prepare_data.py`) — reads each
  frame's `image_path` from the JSON, in the order listed, and attaches
  that frame's ground-truth `boxes` (if any) to the record;
- a raw directory of sequential frame images (`*.png`/`*.jpg`), sorted
  by filename;
- a video file (`.mp4`, `.avi`, `.mov`, `.mkv`), read via OpenCV.

Quality gate
------------
Each frame is checked with simple OpenCV heuristics — grayscale mean
brightness and standard deviation — to flag frames that are extremely
dark, blank/uniform, or low-contrast. This is intentionally simple
(three threshold checks, no ML): good enough to catch obviously bad
frames (a corrupted read, a blank/black frame) without trying to be a
general image-quality model. By default, flagged frames are skipped
from iteration; pass `skip_bad_frames=False` to keep and inspect them
instead (every `FrameRecord` carries `quality_ok` / `quality_reason`
either way).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Quality-gate defaults. Frames are graded on an 8-bit grayscale
# (0-255) version of themselves.
DEFAULT_DARK_MEAN_THRESHOLD = 20.0  # mean brightness below this -> "too dark"
DEFAULT_BLANK_STD_THRESHOLD = 3.0  # std-dev below this -> "blank / uniform"
DEFAULT_LOW_CONTRAST_STD_THRESHOLD = 8.0  # std-dev below this (but above blank) -> "low contrast"


# --------------------------------------------------------------------------
# Quality gate
# --------------------------------------------------------------------------


@dataclass
class QualityResult:
    """Verdict from the image-quality gate for a single frame."""

    ok: bool
    reason: str  # "" when ok
    mean_brightness: float
    std_dev: float


def assess_frame_quality(
    frame: np.ndarray,
    dark_mean_threshold: float = DEFAULT_DARK_MEAN_THRESHOLD,
    blank_std_threshold: float = DEFAULT_BLANK_STD_THRESHOLD,
    low_contrast_std_threshold: float = DEFAULT_LOW_CONTRAST_STD_THRESHOLD,
) -> QualityResult:
    """Grade one frame with simple brightness / contrast heuristics.

    - mean brightness below `dark_mean_threshold` -> flagged as too dark
    - std-dev below `blank_std_threshold` -> flagged as blank/uniform
    - std-dev below `low_contrast_std_threshold` (but not blank) -> flagged as low contrast
    - otherwise -> ok
    """
    if frame is None or frame.size == 0:
        return QualityResult(ok=False, reason="empty/unreadable frame", mean_brightness=0.0, std_dev=0.0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    mean_brightness = float(gray.mean())
    std_dev = float(gray.std())

    if mean_brightness < dark_mean_threshold:
        return QualityResult(
            ok=False,
            reason=f"too dark (mean brightness={mean_brightness:.1f} < {dark_mean_threshold})",
            mean_brightness=mean_brightness,
            std_dev=std_dev,
        )
    if std_dev < blank_std_threshold:
        return QualityResult(
            ok=False,
            reason=f"blank/uniform (std-dev={std_dev:.1f} < {blank_std_threshold})",
            mean_brightness=mean_brightness,
            std_dev=std_dev,
        )
    if std_dev < low_contrast_std_threshold:
        return QualityResult(
            ok=False,
            reason=f"low contrast (std-dev={std_dev:.1f} < {low_contrast_std_threshold})",
            mean_brightness=mean_brightness,
            std_dev=std_dev,
        )
    return QualityResult(ok=True, reason="", mean_brightness=mean_brightness, std_dev=std_dev)


# --------------------------------------------------------------------------
# Frame records
# --------------------------------------------------------------------------


@dataclass
class FrameRecord:
    """One frame handed to the rest of the pipeline."""

    index: int
    frame: np.ndarray
    quality: QualityResult
    frame_id: str | None = None
    image_path: str | None = None
    boxes: list[dict[str, Any]] | None = None  # ground-truth boxes, when loaded from a Phase 1 JSON

    @property
    def quality_ok(self) -> bool:
        return self.quality.ok

    @property
    def quality_reason(self) -> str:
        return self.quality.reason


# --------------------------------------------------------------------------
# FrameSource
# --------------------------------------------------------------------------


class FrameSource:
    """Yields `FrameRecord`s, in order, from a Phase 1 video JSON, a raw
    directory of sequential frame images, or a video file.

    Usage:
        for record in FrameSource("data/processed/p1_v1.json"):
            ...  # record.frame is a numpy array, record.index its position

    Bad frames (per `assess_frame_quality`) are skipped by default; set
    `skip_bad_frames=False` to receive them too (still marked via
    `record.quality_ok`). Either way, `self.n_seen` / `self.n_skipped`
    are updated as iteration proceeds, for a simple end-of-run summary.
    """

    def __init__(
        self,
        source_path: str | Path,
        skip_bad_frames: bool = True,
        dark_mean_threshold: float = DEFAULT_DARK_MEAN_THRESHOLD,
        blank_std_threshold: float = DEFAULT_BLANK_STD_THRESHOLD,
        low_contrast_std_threshold: float = DEFAULT_LOW_CONTRAST_STD_THRESHOLD,
    ) -> None:
        self.source_path = Path(source_path)
        self.skip_bad_frames = skip_bad_frames
        self.dark_mean_threshold = dark_mean_threshold
        self.blank_std_threshold = blank_std_threshold
        self.low_contrast_std_threshold = low_contrast_std_threshold

        self.mode = self._detect_mode(self.source_path)
        self.n_seen = 0
        self.n_skipped = 0

    @staticmethod
    def _detect_mode(path: Path) -> str:
        if path.is_file() and path.suffix.lower() == ".json":
            return "json"
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            return "video"
        if path.is_dir():
            return "directory"
        raise ValueError(
            f"Don't know how to load frames from '{path}': expected a .json video "
            f"file (from prepare_data.py), a directory of frame images, or a video "
            f"file ({sorted(VIDEO_EXTENSIONS)})."
        )

    def _grade(self, frame: np.ndarray) -> QualityResult:
        return assess_frame_quality(
            frame,
            dark_mean_threshold=self.dark_mean_threshold,
            blank_std_threshold=self.blank_std_threshold,
            low_contrast_std_threshold=self.low_contrast_std_threshold,
        )

    def __iter__(self) -> Iterator[FrameRecord]:
        self.n_seen = 0
        self.n_skipped = 0
        if self.mode == "json":
            yield from self._iter_json()
        elif self.mode == "directory":
            yield from self._iter_directory()
        elif self.mode == "video":
            yield from self._iter_video()
        else:  # pragma: no cover - _detect_mode already validates this
            raise ValueError(f"Unknown mode: {self.mode}")

    # -- source-specific generators ---------------------------------------

    def _iter_json(self) -> Iterator[FrameRecord]:
        data = json.loads(self.source_path.read_text())
        frames = data.get("frames", [])
        for index, frame_entry in enumerate(frames):
            image_path = Path(frame_entry["image_path"])
            image = cv2.imread(str(image_path))
            self.n_seen += 1
            quality = self._grade(image)
            if not quality.ok and self.skip_bad_frames:
                self.n_skipped += 1
                continue
            yield FrameRecord(
                index=index,
                frame=image,
                quality=quality,
                frame_id=frame_entry.get("frame_id"),
                image_path=str(image_path),
                boxes=frame_entry.get("boxes"),
            )

    def _iter_directory(self) -> Iterator[FrameRecord]:
        image_paths = sorted(
            p for p in self.source_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        for index, image_path in enumerate(image_paths):
            image = cv2.imread(str(image_path))
            self.n_seen += 1
            quality = self._grade(image)
            if not quality.ok and self.skip_bad_frames:
                self.n_skipped += 1
                continue
            yield FrameRecord(
                index=index,
                frame=image,
                quality=quality,
                frame_id=image_path.stem,
                image_path=str(image_path),
                boxes=None,
            )

    def _iter_video(self) -> Iterator[FrameRecord]:
        cap = cv2.VideoCapture(str(self.source_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {self.source_path}")
        try:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self.n_seen += 1
                quality = self._grade(frame)
                if not quality.ok and self.skip_bad_frames:
                    self.n_skipped += 1
                    index += 1
                    continue
                yield FrameRecord(
                    index=index,
                    frame=frame,
                    quality=quality,
                    frame_id=f"{index:05d}",
                    image_path=None,
                    boxes=None,
                )
                index += 1
        finally:
            cap.release()


def load_cadica_sequence(source_path: str | Path, **kwargs: Any) -> FrameSource:
    """Convenience wrapper: `FrameSource` over a Phase 1 video JSON file
    or a raw CADICA `.../input/` frame directory. Accepts the same
    keyword arguments as `FrameSource` (e.g. `skip_bad_frames`).
    """
    return FrameSource(source_path, **kwargs)
