"""
pipeline/confidence.py
----------------------
Displayed confidence formula for the Synapse 1.0 prototype.

PURPOSE:
Combine two independent heuristic signals — image-based detection score
and temporal persistence — into a single displayed value.

LABELING REQUIREMENT (from ARCHITECTURE.md):
The result MUST always be presented as:
  "Prototype confidence — combines image-based candidate score and temporal persistence."

This value is NOT:
  - A clinically validated probability
  - An output of a trained machine learning model
  - A measurement of stenosis severity
  - A calibrated statistical confidence interval

It is a prototype heuristic blend, transparent in its construction.
"""

import numpy as np
from typing import Sequence


# Blend weights — documented here so they are easy to audit/adjust.
_DETECTION_SCORE_WEIGHT = 0.5
_PERSISTENCE_RATIO_WEIGHT = 0.5

# Human-readable label for this value. Use this string in all UI elements.
DISPLAYED_CONFIDENCE_LABEL = (
    "AI model score is a prototype classification output, not a calibrated clinical probability."
)


def compute_displayed_confidence(
    mean_raw_score: float,
    persistence_ratio: float,
) -> float:
    """
    Compute a displayed confidence value for a tracked lesion candidate.

    Formula (transparent, not calibrated):
        displayed_confidence = 0.5 * mean_raw_score + 0.5 * persistence_ratio

    Where:
    - mean_raw_score   : mean of heuristic detection scores across frames
                         this candidate was observed. Range [0, 1].
    - persistence_ratio: fraction of frames in the candidate's lifespan
                         during which it was actually detected.
                         = persistence_count / (last_frame - first_frame + 1)
                         Range [0, 1].

    The result is clamped to [0.0, 1.0].

    UI label: "Prototype confidence — combines image-based candidate score
               and temporal persistence."

    Parameters
    ----------
    mean_raw_score : float
        Mean heuristic detection score (NOT ML probability). Must be in [0, 1].
    persistence_ratio : float
        Persistence ratio for this lesion candidate. Must be in [0, 1].

    Returns
    -------
    float
        Displayed confidence in [0.0, 1.0].
    """
    value = (
        _DETECTION_SCORE_WEIGHT * float(mean_raw_score)
        + _PERSISTENCE_RATIO_WEIGHT * float(persistence_ratio)
    )
    return float(np.clip(value, 0.0, 1.0))


def compute_displayed_confidence_from_history(
    score_history: Sequence[float],
    persistence_count: int,
    first_frame: int,
    last_frame: int,
) -> float:
    """
    Convenience wrapper: compute displayed confidence directly from a lesion's
    score_history and frame span.

    Parameters
    ----------
    score_history : Sequence[float]
        Raw detection scores from each frame this candidate was observed.
    persistence_count : int
        Number of frames the candidate was actually detected.
    first_frame : int
        Index of the first frame the candidate appeared in.
    last_frame : int
        Index of the most recent frame the candidate appeared in.

    Returns
    -------
    float
        Displayed confidence in [0.0, 1.0].
    """
    if not score_history:
        return 0.0

    mean_raw_score = float(np.mean(score_history))

    lifespan = last_frame - first_frame + 1
    persistence_ratio = persistence_count / lifespan if lifespan > 0 else 0.0
    persistence_ratio = float(np.clip(persistence_ratio, 0.0, 1.0))

    return compute_displayed_confidence(mean_raw_score, persistence_ratio)


if __name__ == "__main__":
    # Smoke tests
    test_cases = [
        # (mean_raw_score, persistence_ratio, expected_range)
        (0.8, 1.0, (0.8, 0.95)),    # high on both -> high confidence
        (0.5, 0.5, (0.45, 0.55)),   # mid on both -> mid confidence
        (0.0, 0.0, (0.0,  0.01)),   # zero on both -> zero
        (1.0, 1.0, (0.95, 1.0)),    # max on both -> 1.0
        (1.5, 1.5, (0.95, 1.0)),    # out-of-range inputs -> clamped to 1.0
        (-0.5, -0.5, (0.0, 0.01)),  # negative inputs -> clamped to 0.0
    ]

    print("--- confidence.py smoke test ---")
    all_passed = True
    for mean_s, pers_r, (lo, hi) in test_cases:
        result = compute_displayed_confidence(mean_s, pers_r)
        ok = lo <= result <= hi
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  [{status}] mean_raw={mean_s:.2f}, pers_ratio={pers_r:.2f} -> {result:.4f}  (expected [{lo},{hi}])")

    # Test convenience wrapper
    history = [0.6, 0.7, 0.65, 0.72]
    dc = compute_displayed_confidence_from_history(
        score_history=history,
        persistence_count=4,
        first_frame=5,
        last_frame=8,
    )
    assert 0.0 <= dc <= 1.0, f"Wrapper result out of range: {dc}"
    print(f"  [PASS] wrapper test -> displayed_confidence={dc:.4f}")
    print(f"\n  Label to use in UI:\n  \"{DISPLAYED_CONFIDENCE_LABEL}\"")

    if all_passed:
        print("\nconfidence.py smoke test: PASSED")
    else:
        print("\nconfidence.py smoke test: FAILED")
        raise SystemExit(1)
