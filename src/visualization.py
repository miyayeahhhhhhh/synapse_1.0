"""Visualization overlay.

Draws tracked bounding boxes, confidence scores, and persistence/curve
indicators onto frames for the Streamlit live view.

Not implemented yet (Phase 0: skeleton only).
"""

from __future__ import annotations

from .tracker import Track


def draw_overlays(frame, tracks: list[Track]):
    """Return a copy of `frame` with track overlays drawn on it.

    Not implemented yet.
    """
    raise NotImplementedError("draw_overlays is not implemented yet.")
