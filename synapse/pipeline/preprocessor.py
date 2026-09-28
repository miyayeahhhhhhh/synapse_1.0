"""
pipeline/preprocessor.py
------------------------
Frame preprocessing: grayscale conversion, CLAHE enhancement, Gaussian blur.

Design goals:
- Fast and deterministic (no randomness).
- Improves contrast in low-intensity vessel regions to assist candidate detection.
- Does NOT apply any trained model or learned transform.
"""

import cv2
import numpy as np


# CLAHE parameters — tuned for low-contrast grayscale angiography-like images.
_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID_SIZE = (8, 8)

# Gaussian blur kernel — light smoothing to reduce noise before thresholding.
_BLUR_KERNEL_SIZE = (5, 5)
_BLUR_SIGMA = 1.0

# Shared CLAHE object (thread-safe for read-only usage after creation).
_clahe = cv2.createCLAHE(
    clipLimit=_CLAHE_CLIP_LIMIT,
    tileGridSize=_CLAHE_TILE_GRID_SIZE,
)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a single frame for candidate region detection.

    Steps:
    1. Convert to grayscale (no-op if already single-channel).
    2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
       to improve local contrast in vessel regions.
    3. Apply light Gaussian blur to suppress high-frequency noise.

    Parameters
    ----------
    frame : np.ndarray
        Input frame. Accepts uint8 grayscale (H, W) or BGR (H, W, 3).

    Returns
    -------
    np.ndarray
        Enhanced uint8 grayscale frame, same spatial dimensions as input.
    """
    # Step 1: Ensure grayscale
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        gray = frame[:, :, 0]
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError(
            f"Unsupported frame shape: {frame.shape}. "
            "Expected (H, W) or (H, W, 3) uint8 array."
        )

    # Step 2: CLAHE — local contrast enhancement
    enhanced = _clahe.apply(gray)

    # Step 3: Light Gaussian blur — noise suppression
    blurred = cv2.GaussianBlur(enhanced, _BLUR_KERNEL_SIZE, _BLUR_SIGMA)

    return blurred


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from utils.sample_data import generate_synthetic_sequence

    frames = generate_synthetic_sequence(n_frames=5)
    for i, frame in enumerate(frames):
        out = preprocess_frame(frame)
        assert out.shape == frame.shape, "Shape mismatch after preprocessing"
        assert out.dtype == np.uint8, "Output must be uint8"

    print(f"preprocessor.py smoke test: PASSED ({len(frames)} frames processed)")
