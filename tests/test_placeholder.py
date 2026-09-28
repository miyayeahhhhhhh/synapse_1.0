"""Phase 0 placeholder tests.

Only checks that the package skeleton imports cleanly. Real tests for
detection, tracking, and visualization logic land in later phases.
"""

import src.detector
import src.report
import src.tracker
import src.video_loader
import src.visualization


def test_modules_import():
    """All skeleton modules should import without error."""
    assert src.video_loader is not None
    assert src.detector is not None
    assert src.tracker is not None
    assert src.visualization is not None
    assert src.report is not None


def test_package_version():
    import src

    assert isinstance(src.__version__, str)
