# SYNTHETIC TEST FIXTURES — NOT REAL CADICA DATA

Hand-generated frames for exercising `src/video_loader.py`'s image-quality
gate end-to-end (see `tests/test_video_loader.py`). This is a flat
directory of frames (not a full CADICA `pX/vY/` layout) — it's used to
test the "raw directory of sequential frames" loading mode.

| file | intent | measured mean / std-dev |
|---|---|---|
| `frame_0000.png` | normal frame — should pass the gate | mean=129.2, std=51.8 |
| `frame_0001.png` | blank/uniform frame — should be flagged/skipped | mean=128.0, std=0.0 |
| `frame_0002.png` | extremely dark frame — should be flagged/skipped | mean=1.5, std=1.1 |
| `frame_0003.png` | normal frame — should pass the gate | mean=114.4, std=48.3 |
| `frame_0004.png` | low-contrast (not blank) frame — should be flagged/skipped | mean=125.6, std=3.5 |
| `frame_0005_corrupted.png` | not a valid PNG at all (garbage bytes) — `cv2.imread` returns `None`; should be flagged/skipped as unreadable | n/a |

Regenerate with the snippet in `tests/test_video_loader.py`'s
`_write_quality_gate_fixtures` helper if these ever need to change.
