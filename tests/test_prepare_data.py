"""Phase 1 tests: exercise scripts/prepare_data.py against the small
synthetic CADICA-shaped fixtures in tests/fixtures/cadica_synthetic/
(NOT real CADICA data — see that folder's README).
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "cadica_synthetic"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import prepare_data  # noqa: E402


def test_find_video_folders_discovers_both_synthetic_videos():
    video_dirs = sorted(prepare_data.find_video_folders(FIXTURES_DIR))
    rel = sorted(str(d.relative_to(FIXTURES_DIR)) for d in video_dirs)
    assert rel == [str(Path("p1") / "v1"), str(Path("p2") / "v1")]


def test_prepare_video_lesion_video_has_expected_boxes():
    video_dir = FIXTURES_DIR / "p1" / "v1"
    video_prep = prepare_data.prepare_video(video_dir, FIXTURES_DIR)

    assert video_prep.video_id == "p1_v1"
    assert video_prep.patient_id == "p1"
    assert video_prep.has_lesion is True
    assert len(video_prep.frames) == 3

    frame_1, frame_2, frame_3 = video_prep.frames
    assert frame_1.frame_id == "00001"
    assert len(frame_1.boxes) == 1
    assert frame_1.boxes[0].category == "p70_90"

    assert len(frame_2.boxes) == 2
    assert {b.category for b in frame_2.boxes} == {"p70_90", "p20_50"}

    # Keyframe with no groundtruth file for it -> no boxes, not an error.
    assert frame_3.boxes == []


def test_prepare_video_non_lesion_video_has_no_boxes():
    video_dir = FIXTURES_DIR / "p2" / "v1"
    video_prep = prepare_data.prepare_video(video_dir, FIXTURES_DIR)

    assert video_prep.has_lesion is False
    assert len(video_prep.frames) == 2
    assert all(f.boxes == [] for f in video_prep.frames)


def test_write_video_json_round_trips(tmp_path):
    video_dir = FIXTURES_DIR / "p1" / "v1"
    video_prep = prepare_data.prepare_video(video_dir, FIXTURES_DIR)
    out_path = prepare_data.write_video_json(video_prep, tmp_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["video_id"] == "p1_v1"
    assert data["has_lesion"] is True
    assert len(data["frames"]) == 3
    assert data["frames"][0]["boxes"][0]["category"] == "p70_90"


def test_parse_groundtruth_file_rejects_malformed_line(tmp_path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("not-enough-fields\n")
    try:
        prepare_data.parse_groundtruth_file(bad_file)
        assert False, "expected a ValueError for a malformed groundtruth line"
    except ValueError:
        pass


def test_no_video_folders_found_does_not_raise(tmp_path):
    empty_root = tmp_path / "nothing_here"
    empty_root.mkdir()
    assert list(prepare_data.find_video_folders(empty_root)) == []
