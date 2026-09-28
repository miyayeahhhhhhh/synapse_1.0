#!/usr/bin/env python3
"""Phase 1 — CADICA data preparation.

Scans a CADICA-format data root for video folders, parses each video's
frames and (where present) per-frame lesion bounding boxes, and writes
one JSON file per video into an output directory. Later phases
(``src/video_loader.py`` onward) are meant to consume these JSON files
instead of re-parsing the raw CADICA layout each time.

CADICA on-disk format (per the dataset's own documentation —
https://data.mendeley.com/datasets/p9bpx9ctcv):

    <root>/[selectedVideos|nonselectedVideos]/pX/vY/
        input/
            pX_vY_00001.png
            pX_vY_00002.png
            ...
        pX_vY_selectedFrames.txt        # keyframe IDs, one per line
        groundtruth/                    # present only if the video has a labeled lesion
            pX_vY_00001.txt             # one row per box: "x y w h category"
            pX_vY_00002.txt
            ...
            pX_vY_groundTruthTable.mat  # ignored here (only the .txt files are parsed)

Box format: ``x y w h category`` per line, where (x, y) is the pixel
coordinate of the box's top-left corner, w/h are its width and height,
and category is one of the seven CADICA stenosis-severity labels:
p0_20, p20_50, p50_70, p70_90, p90_98, p99, p100.

This script only locates videos, parses that format, and writes the
internal JSON representation below. It does NOT do any detection,
tracking, or model training — that's later phases.

Output representation (one JSON file per video):

    {
      "video_id": "p1_v1",
      "patient_id": "p1",
      "source_dir": "/path/to/.../p1/v1",
      "has_lesion": true,
      "frames": [
        {
          "frame_id": "00001",
          "image_path": "/path/to/.../p1/v1/input/p1_v1_00001.png",
          "boxes": [
            {"x": 12, "y": 8, "w": 6, "h": 5, "category": "p70_90"}
          ]
        },
        ...
      ]
    }

Usage:
    # Real dataset, once downloaded into data/cadica/:
    python scripts/prepare_data.py --data-root data/cadica --output-dir data/processed

    # Synthetic fixtures (no real data needed) — good for a smoke test:
    python scripts/prepare_data.py --data-root tests/fixtures/cadica_synthetic --output-dir /tmp/synapse_prepared

    # (Re)generate the synthetic fixtures themselves:
    python scripts/prepare_data.py --make-synthetic-fixtures
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

VALID_CATEGORIES = {"p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "cadica"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "cadica_synthetic"


# --------------------------------------------------------------------------
# Internal representation
# --------------------------------------------------------------------------


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    category: str


@dataclass
class FramePrep:
    frame_id: str
    image_path: str
    boxes: list[Box] = field(default_factory=list)


@dataclass
class VideoPrep:
    video_id: str
    patient_id: str
    source_dir: str
    has_lesion: bool
    frames: list[FramePrep] = field(default_factory=list)


# --------------------------------------------------------------------------
# Locating + parsing
# --------------------------------------------------------------------------


def find_video_folders(root: str | Path) -> Iterator[Path]:
    """Yield every directory under `root` that looks like a CADICA video
    folder: one that directly contains an `input/` subdirectory holding
    at least one `.png` frame.

    Works regardless of how deep `root` nests videos (e.g. directly as
    `pX/vY/`, or under `selectedVideos/pX/vY/` / `nonselectedVideos/pX/vY/`).
    """
    root = Path(root)
    if not root.is_dir():
        return
    for input_dir in sorted(root.rglob("input")):
        if input_dir.is_dir() and next(input_dir.glob("*.png"), None) is not None:
            yield input_dir.parent


def parse_groundtruth_file(txt_path: Path) -> list[Box]:
    """Parse one groundtruth file: one box per line, `x y w h category`."""
    boxes: list[Box] = []
    for lineno, raw_line in enumerate(txt_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(
                f"{txt_path}:{lineno}: expected 'x y w h category', got {raw_line!r}"
            )
        x, y, w, h = (int(float(p)) for p in parts[:4])
        category = parts[4]
        if category not in VALID_CATEGORIES:
            print(
                f"warning: {txt_path}:{lineno}: unrecognized category "
                f"{category!r} (expected one of {sorted(VALID_CATEGORIES)})",
                file=sys.stderr,
            )
        boxes.append(Box(x=x, y=y, w=w, h=h, category=category))
    return boxes


def frame_id_from_filename(png_path: Path) -> str:
    """`pX_vY_00001.png` -> `00001`. Falls back to the file stem if the
    frame filename doesn't end in `_<digits>`."""
    match = re.match(r".+_(\d+)$", png_path.stem)
    return match.group(1) if match else png_path.stem


def prepare_video(video_dir: Path, root: Path) -> VideoPrep:
    """Parse one video folder into a `VideoPrep`."""
    input_dir = video_dir / "input"
    groundtruth_dir = video_dir / "groundtruth"
    has_lesion = groundtruth_dir.is_dir()

    rel_parts = video_dir.relative_to(root).parts
    video_id = "_".join(rel_parts) if rel_parts else video_dir.name
    patient_id = next(
        (part for part in rel_parts if re.fullmatch(r"p\d+", part)),
        rel_parts[0] if rel_parts else "unknown",
    )

    frames: list[FramePrep] = []
    for png_path in sorted(input_dir.glob("*.png")):
        frame_id = frame_id_from_filename(png_path)
        gt_path = groundtruth_dir / f"{png_path.stem}.txt"
        boxes = parse_groundtruth_file(gt_path) if gt_path.is_file() else []
        frames.append(FramePrep(frame_id=frame_id, image_path=str(png_path), boxes=boxes))

    return VideoPrep(
        video_id=video_id,
        patient_id=patient_id,
        source_dir=str(video_dir),
        has_lesion=has_lesion,
        frames=frames,
    )


def write_video_json(video_prep: VideoPrep, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_prep.video_id}.json"
    out_path.write_text(json.dumps(asdict(video_prep), indent=2))
    return out_path


# --------------------------------------------------------------------------
# Synthetic test fixtures (NOT real data — see fixtures README)
# --------------------------------------------------------------------------


def make_synthetic_fixtures(fixtures_dir: Path) -> Path:
    """(Re)generate small synthetic CADICA-shaped video folders so this
    script (and later phases) can be exercised end-to-end without the
    real ~dozens-of-GB CADICA dataset. Clearly not real angiography data.
    """
    import numpy as np
    import cv2

    fixtures_dir = Path(fixtures_dir)
    if fixtures_dir.exists():
        shutil.rmtree(fixtures_dir)
    fixtures_dir.mkdir(parents=True)

    (fixtures_dir / "README.md").write_text(
        "# SYNTHETIC TEST FIXTURES — NOT REAL CADICA DATA\n\n"
        "Generated by `scripts/prepare_data.py --make-synthetic-fixtures`.\n\n"
        "Tiny (32x32) random-noise placeholder frames and made-up bounding "
        "boxes, laid out to mirror the real CADICA folder structure "
        "(`pX/vY/input/`, `pX/vY/groundtruth/`) purely so `prepare_data.py` "
        "and later pipeline phases can be tested end-to-end without the "
        "real dataset. Do not use for anything clinical or diagnostic.\n"
    )

    rng = np.random.default_rng(42)

    # p1/v1 — a "lesion" video: has a groundtruth/ dir; one keyframe has
    # two overlapping-ish boxes, one keyframe has a single box, and one
    # selected keyframe has NO box (lesion not visible in that frame).
    _make_synthetic_video(
        fixtures_dir / "p1" / "v1",
        patient="p1",
        video="v1",
        n_frames=3,
        rng=rng,
        groundtruth={
            "00001": [(12, 8, 6, 5, "p70_90")],
            "00002": [(13, 9, 6, 5, "p70_90"), (2, 2, 3, 3, "p20_50")],
            # 00003: intentionally no groundtruth file for this keyframe
        },
    )

    # p2/v1 — a "non-lesion" video: no groundtruth/ directory at all.
    _make_synthetic_video(
        fixtures_dir / "p2" / "v1",
        patient="p2",
        video="v1",
        n_frames=2,
        rng=rng,
        groundtruth=None,
    )

    return fixtures_dir


def _make_synthetic_video(
    video_dir: Path,
    patient: str,
    video: str,
    n_frames: int,
    rng,
    groundtruth: dict[str, list[tuple[int, int, int, int, str]]] | None,
) -> None:
    import numpy as np
    import cv2

    input_dir = video_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = [f"{i:05d}" for i in range(1, n_frames + 1)]
    for frame_id in frame_ids:
        frame = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)
        cv2.imwrite(str(input_dir / f"{patient}_{video}_{frame_id}.png"), frame)

    (video_dir / f"{patient}_{video}_selectedFrames.txt").write_text(
        "\n".join(frame_ids) + "\n"
    )

    if groundtruth:
        gt_dir = video_dir / "groundtruth"
        gt_dir.mkdir(exist_ok=True)
        for frame_id, boxes in groundtruth.items():
            lines = [f"{x} {y} {w} {h} {category}" for (x, y, w, h, category) in boxes]
            (gt_dir / f"{patient}_{video}_{frame_id}.txt").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help=f"Root directory to scan for CADICA video folders (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Where to write one JSON file per video (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--make-synthetic-fixtures",
        action="store_true",
        help=f"(Re)generate the synthetic test fixtures under {DEFAULT_FIXTURES_DIR} and exit.",
    )
    args = parser.parse_args()

    if args.make_synthetic_fixtures:
        make_synthetic_fixtures(DEFAULT_FIXTURES_DIR)
        print(f"Synthetic test fixtures written to {DEFAULT_FIXTURES_DIR}")
        return

    data_root = Path(args.data_root)
    video_dirs = list(find_video_folders(data_root))

    if not video_dirs:
        print(
            f"No CADICA video folders found under '{data_root}' "
            f"(looking for */input/*.png). Nothing to do.\n\n"
            f"If the real CADICA dataset isn't downloaded yet, try this "
            f"against the synthetic fixtures instead:\n"
            f"  python {Path(__file__).relative_to(REPO_ROOT)} "
            f"--data-root {DEFAULT_FIXTURES_DIR.relative_to(REPO_ROOT)} --output-dir /tmp/synapse_prepared"
        )
        return

    written = []
    for video_dir in video_dirs:
        video_prep = prepare_video(video_dir, data_root)
        out_path = write_video_json(video_prep, Path(args.output_dir))
        written.append(out_path)
        n_boxes = sum(len(f.boxes) for f in video_prep.frames)
        print(
            f"  {video_prep.video_id}: {len(video_prep.frames)} frame(s), "
            f"{n_boxes} box(es), has_lesion={video_prep.has_lesion} -> {out_path}"
        )

    print(f"\nWrote {len(written)} video JSON file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
