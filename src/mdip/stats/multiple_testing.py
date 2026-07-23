"""Benjamini-Hochberg correction for running a family of tests.

Business question: we ran several tests (2 email arms x 3 outcomes = 6). Testing many
things inflates the chance that at least one looks "significant" purely by luck. How do
we keep our list of findings trustworthy?

Intuition: at alpha = 0.05, each test has a 5% false-positive chance. Across 6 tests the
probability of *at least one* false alarm is well above 5%. Benjamini-Hochberg raises
the bar in a graded way to control the **false discovery rate** (FDR) — the expected
share of our "significant" results that are actually false. It's less severe than
Bonferroni, which is why it's the modern default.

Assumptions: tests are independent or positively correlated (holds for our outcomes).

We implement the ~6-line algorithm by hand to show we understand it; a unit test
cross-checks it against statsmodels.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> dict:
    """Apply the Benjamini-Hochberg FDR procedure.

    Algorithm: sort the m p-values ascending. The largest k for which
    ``p_(k) <= (k/m) * alpha`` is the cutoff; reject all hypotheses up to it.

    Args:
        p_values: raw p-values from the family of tests.
        alpha:    target false discovery rate.

    Returns:
        Dict with, in the original input order: ``reject`` (bool per test) and
        ``adjusted`` (BH-adjusted p-values, capped at 1.0).
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)              # indices that sort p ascending
    ranks = np.arange(1, m + 1)

    p_sorted = p[order]

    # BH critical values and rejection: find the largest rank passing the threshold.
    passes = p_sorted <= (ranks / m) * alpha
    if passes.any():
        max_k = np.max(np.where(passes))
        reject_sorted = np.arange(m) <= max_k
    else:
        reject_sorted = np.zeros(m, dtype=bool)

    # Adjusted p-values: enforce monotonicity from the largest rank downward.
    adjusted_sorted = np.minimum.accumulate((p_sorted * m / ranks)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    # Map both back to the original input order.
    reject = np.empty(m, dtype=bool)
    adjusted = np.empty(m, dtype=float)
    reject[order] = reject_sorted
    adjusted[order] = adjusted_sorted

    return {"reject": reject.tolist(), "adjusted": adjusted.tolist()}
