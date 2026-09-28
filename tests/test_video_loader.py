"""Phase 2 tests: src/video_loader.py — the quality gate and all three
frame-loading modes (Phase 1 JSON, raw frame directory, video file).

Uses:
- tests/fixtures/quality_gate_synthetic/  — hand-built frames for the
  quality gate itself (normal / blank / dark / low-contrast / corrupted).
- tests/fixtures/cadica_synthetic/        — Phase 1's synthetic CADICA
  fixtures, to test JSON-mode and directory-mode loading against
  something that looks like real project data (read-only here; never
  modified by these tests).
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUALITY_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "quality_gate_synthetic"
CADICA_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "cadica_synthetic"

sys.path.insert(0, str(REPO_ROOT))
from src.video_loader import FrameSource, assess_frame_quality  # noqa: E402


def _write_quality_gate_fixtures(fixtures_dir: Path) -> None:
    """Regenerate the hand-built quality-gate fixture frames. Not called
    automatically by the tests (the checked-in files are already there);
    kept here so the fixture set can be reproduced if it's ever lost.
    See tests/fixtures/quality_gate_synthetic/README.md for what each
    frame is meant to exercise.
    """
    import numpy as np
    import cv2

    fixtures_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    cv2.imwrite(str(fixtures_dir / "frame_0000.png"), rng.integers(40, 220, size=(64, 64), dtype=np.uint8))
    cv2.imwrite(str(fixtures_dir / "frame_0001.png"), np.full((64, 64), 128, dtype=np.uint8))
    cv2.imwrite(str(fixtures_dir / "frame_0002.png"), rng.integers(0, 4, size=(64, 64), dtype=np.uint8))
    cv2.imwrite(str(fixtures_dir / "frame_0003.png"), rng.integers(30, 200, size=(64, 64), dtype=np.uint8))
    cv2.imwrite(str(fixtures_dir / "frame_0004.png"), rng.integers(120, 132, size=(64, 64), dtype=np.uint8))
    (fixtures_dir / "frame_0005_corrupted.png").write_text("not a valid png, just garbage bytes\n")


# --------------------------------------------------------------------------
# Quality gate, in isolation
# --------------------------------------------------------------------------


def test_assess_frame_quality_normal_frame_passes():
    import cv2

    img = cv2.imread(str(QUALITY_FIXTURES_DIR / "frame_0000.png"))
    result = assess_frame_quality(img)
    assert result.ok, result.reason


def test_assess_frame_quality_blank_frame_flagged():
    import cv2

    img = cv2.imread(str(QUALITY_FIXTURES_DIR / "frame_0001.png"))
    result = assess_frame_quality(img)
    assert not result.ok
    assert "blank" in result.reason


def test_assess_frame_quality_dark_frame_flagged():
    import cv2

    img = cv2.imread(str(QUALITY_FIXTURES_DIR / "frame_0002.png"))
    result = assess_frame_quality(img)
    assert not result.ok
    assert "dark" in result.reason


def test_assess_frame_quality_low_contrast_frame_flagged():
    import cv2

    img = cv2.imread(str(QUALITY_FIXTURES_DIR / "frame_0004.png"))
    result = assess_frame_quality(img)
    assert not result.ok
    assert "contrast" in result.reason


def test_assess_frame_quality_corrupted_frame_flagged():
    import cv2

    img = cv2.imread(str(QUALITY_FIXTURES_DIR / "frame_0005_corrupted.png"))
    assert img is None  # cv2 itself fails to decode it
    result = assess_frame_quality(img)
    assert not result.ok


# --------------------------------------------------------------------------
# FrameSource — directory mode, against the quality-gate fixtures
# --------------------------------------------------------------------------


def test_frame_source_directory_mode_skips_bad_frames_by_default():
    source = FrameSource(QUALITY_FIXTURES_DIR)
    records = list(source)

    # 6 files on disk; only frame_0000 and frame_0003 should pass the gate.
    good_ids = {r.frame_id for r in records}
    assert good_ids == {"frame_0000", "frame_0003"}
    assert all(r.quality_ok for r in records)
    assert source.n_seen == 6
    assert source.n_skipped == 4


def test_frame_source_directory_mode_keeps_bad_frames_when_asked():
    source = FrameSource(QUALITY_FIXTURES_DIR, skip_bad_frames=False)
    records = list(source)

    assert len(records) == 6
    ok_by_id = {r.frame_id: r.quality_ok for r in records}
    assert ok_by_id["frame_0000"] is True
    assert ok_by_id["frame_0001"] is False  # blank
    assert ok_by_id["frame_0002"] is False  # dark
    assert ok_by_id["frame_0003"] is True
    assert ok_by_id["frame_0004"] is False  # low contrast
    assert ok_by_id["frame_0005_corrupted"] is False  # unreadable


def test_frame_source_yields_frames_in_order():
    source = FrameSource(QUALITY_FIXTURES_DIR, skip_bad_frames=False)
    records = list(source)
    assert [r.index for r in records] == list(range(len(records)))
    assert [r.frame_id for r in records] == sorted(r.frame_id for r in records)


# --------------------------------------------------------------------------
# FrameSource — JSON mode, against Phase 1's prepared CADICA fixtures
# --------------------------------------------------------------------------


def _prepare_cadica_json(tmp_path: Path) -> Path:
    """Run Phase 1's prepare_data.py against the Phase 1 synthetic CADICA
    fixtures (read-only) to get a video JSON file to load frames from.
    """
    output_dir = tmp_path / "prepared"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "prepare_data.py"),
            "--data-root",
            str(CADICA_FIXTURES_DIR),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
    )
    return output_dir / "p1_v1.json"


def test_frame_source_json_mode_yields_frames_with_boxes(tmp_path):
    video_json = _prepare_cadica_json(tmp_path)
    assert video_json.exists()

    source = FrameSource(video_json, skip_bad_frames=False)
    records = list(source)

    assert len(records) == 3  # p1/v1 has 3 frames (see Phase 1 fixtures)
    assert [r.frame_id for r in records] == ["00001", "00002", "00003"]
    assert records[0].boxes and records[0].boxes[0]["category"] == "p70_90"
    assert len(records[1].boxes) == 2
    assert records[2].boxes == []  # frame with no groundtruth file


def test_frame_source_directory_mode_on_real_cadica_fixture_input_dir():
    # Point directory mode straight at a Phase 1 fixture's raw input/ dir
    # (no JSON involved) to confirm that loading path also works.
    input_dir = CADICA_FIXTURES_DIR / "p2" / "v1" / "input"
    source = FrameSource(input_dir, skip_bad_frames=False)
    records = list(source)
    assert len(records) == 2
    assert records[0].boxes is None  # no annotation info available in this mode


# --------------------------------------------------------------------------
# FrameSource — video-file mode
# --------------------------------------------------------------------------


def test_frame_source_video_mode(tmp_path):
    import numpy as np
    import cv2

    video_path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
    rng = np.random.default_rng(3)
    for _ in range(4):
        frame = cv2.cvtColor(rng.integers(40, 220, size=(64, 64), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        writer.write(frame)
    writer.release()

    source = FrameSource(video_path)
    records = list(source)
    assert len(records) == 4
    assert [r.index for r in records] == [0, 1, 2, 3]
    assert all(r.quality_ok for r in records)


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


def test_frame_source_rejects_unknown_path(tmp_path):
    missing = tmp_path / "does_not_exist.xyz"
    try:
        FrameSource(missing)
        assert False, "expected a ValueError for a nonexistent, non-directory path"
    except ValueError:
        pass
