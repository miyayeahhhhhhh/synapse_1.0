"""
utils/sample_data.py
--------------------
Synthetic angiography-like frame generator for development and demonstration purposes.

IMPORTANT:
- These frames are procedurally generated test fixtures.
- They are NOT real patient data.
- They are NOT representative of clinical angiography.
- Do NOT use to claim medical detection accuracy.
"""

import numpy as np
from typing import List


def _draw_vessel(
    canvas: np.ndarray,
    start: tuple,
    end: tuple,
    width: int,
    intensity: int,
) -> None:
    """Draw a single vessel segment (line with thickness) onto the canvas."""
    import cv2
    cv2.line(canvas, start, end, intensity, width, lineType=cv2.LINE_AA)


def _apply_narrowing(
    canvas: np.ndarray,
    center: tuple,
    radius: int,
    dark_value: int,
) -> None:
    """
    Simulate a narrowing (stenosis-like) region by painting a dark elliptical
    patch at `center` with explicit pixel intensity `dark_value`.
    This produces a clear, detectable dip in vessel brightness.
    """
    import cv2
    h, w = canvas.shape[:2]
    x, y = center
    # Draw a solid dark ellipse directly — visually distinct from vessel
    cv2.ellipse(
        canvas,
        (x, y),
        (radius, max(3, int(radius * 0.45))),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=dark_value,
        thickness=-1,
    )


def generate_synthetic_sequence(
    n_frames: int = 60,
    height: int = 480,
    width: int = 640,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Generate a sequence of synthetic angiography-like grayscale frames.

    Design choices:
    - Dark background with lighter vessel channels (mimics X-ray contrast).
    - 3 main vessel paths drawn as thick lines.
    - 3 narrowing regions that persist for ~15-30 consecutive frames each,
      allowing the temporal tracker to demonstrate persistence.
    - Small frame-to-frame jitter to simulate realistic motion.
    - All pixel values uint8.

    Parameters
    ----------
    n_frames : int
        Number of frames to generate.
    height, width : int
        Frame dimensions in pixels.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    List[np.ndarray]
        List of (height, width) uint8 arrays. NOT clinical/patient data.
    """
    import cv2

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Static vessel layout (defined once, jittered per-frame)
    # ------------------------------------------------------------------ #
    # Three vessel paths: (start_xy, end_xy, width, vessel_intensity)
    vessels = [
        ((80,  60),  (560, 140), 16, 210),   # main left-to-right artery (brighter, wider)
        ((80,  60),  (200, 420), 12, 195),   # branching downward
        ((200, 420), (580, 380), 13, 188),   # horizontal branch
    ]

    # ------------------------------------------------------------------ #
    # Narrowing regions (lesion-like): (center_xy, radius, dark_fill,
    #   active_frames_start, active_frames_end)
    # dark_fill: pixel intensity of the narrowing patch (low = dark = occlusion)
    # ------------------------------------------------------------------ #
    narrowings = [
        # Center,     radius, dark_fill, frame_start, frame_end
        # dark_fill ~60: clearly below vessel (190-210) but above background (~18)
        # This creates a detectable dip in vessel intensity.
        # Verified vessel centerline positions from pixel inspection.
        ((220, 79),   22,     60,        5,  50),   # Candidate A on main artery — 45 frames
        ((158, 72),   18,     55,        15, 55),   # Candidate B on left branch — 40 frames
        ((388, 109),  20,     58,        0,  30),   # Candidate C on horizontal branch — 30 frames
    ]

    frames: List[np.ndarray] = []

    for fi in range(n_frames):
        # Dark background (angiography is typically dark background)
        canvas = np.full((height, width), fill_value=18, dtype=np.uint8)

        # Small global jitter per frame (±1 px) to simulate motion
        jitter_x = int(rng.integers(-1, 2))
        jitter_y = int(rng.integers(-1, 2))

        # Draw vessels
        for (sx, sy), (ex, ey), vw, intensity in vessels:
            start = (
                int(np.clip(sx + jitter_x, 0, width - 1)),
                int(np.clip(sy + jitter_y, 0, height - 1)),
            )
            end = (
                int(np.clip(ex + jitter_x, 0, width - 1)),
                int(np.clip(ey + jitter_y, 0, height - 1)),
            )
            _draw_vessel(canvas, start, end, vw, intensity)

        # Slight additive noise
        noise = rng.integers(-6, 7, size=(height, width), dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Apply narrowings only during their active frame windows
        for (cx, cy), radius, dark_fill, f_start, f_end in narrowings:
            if f_start <= fi < f_end:
                # Jitter the center slightly per frame for realism
                nx = int(np.clip(cx + jitter_x, radius, width - radius))
                ny = int(np.clip(cy + jitter_y, radius, height - radius))
                _apply_narrowing(canvas, (nx, ny), radius, dark_fill)

        frames.append(canvas)

    return frames


if __name__ == "__main__":
    # Quick smoke test
    frames = generate_synthetic_sequence(n_frames=60)
    print(f"Generated {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}, dtype: {frames[0].dtype}")
    print(f"Pixel range — min: {frames[0].min()}, max: {frames[0].max()}")
    assert len(frames) == 60
    assert frames[0].shape == (480, 640)
    assert frames[0].dtype == np.uint8
    print("sample_data.py smoke test: PASSED")
