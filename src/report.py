"""PDF report generation.

Summarizes a completed run (tracked lesions, persistence, confidence
curves, representative frames) into a shareable PDF report using fpdf2.

Not implemented yet (Phase 0: skeleton only).
"""

from __future__ import annotations

from pathlib import Path

from .tracker import Track


def generate_pdf_report(tracks: list[Track], output_path: str | Path) -> Path:
    """Generate a PDF summary report for a set of tracks.

    Not implemented yet.
    """
    raise NotImplementedError("generate_pdf_report is not implemented yet.")
