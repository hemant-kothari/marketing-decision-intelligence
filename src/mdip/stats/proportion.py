"""Proportion tests: Sample Ratio Mismatch (SRM) and the two-proportion z-test.

These answer the two questions at the heart of an A/B readout:
  1. SRM  -> "Can I trust this experiment?" (did randomisation split the arms correctly?)
  2. z-test -> "Is the lift in a rate real, or noise?"

We implement the z-test from the standard formula rather than only calling a library,
because the point of the project is to *understand* the statistics. A unit test
cross-checks our result against statsmodels so we know the hand version is correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ProportionTestResult:
    """Result of a two-proportion z-test, with everything needed to report it."""

    label: str
    treated_rate: float
    control_rate: float
    lift: float               # treated_rate - control_rate (absolute, in proportion units)
    ci_low: float             # 95% CI on the lift (unpooled SE)
    ci_high: float
    z_statistic: float
    p_value: float
    n_treated: int
    n_control: int

    @property
    def significant(self) -> bool:
        """True if the 95% CI on the lift excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0


def srm_check(observed_counts: list[int], expected_ratios: list[float] | None = None) -> dict:
    """Chi-square goodness-of-fit test for Sample Ratio Mismatch.

    Business question: did customers land in the arms in the intended proportions?
    A *large* p-value is the good outcome here: it means no evidence of mismatch.

    Assumptions: independent observations, mutually exclusive arms, large expected
    counts per cell (here ~21k, so the chi-square approximation is excellent).

    Args:
        observed_counts: Number of customers in each arm.
        expected_ratios: Intended split (defaults to equal across arms).

    Returns:
        Dict with the chi-square statistic, p-value, and a plain-English verdict.
    """
    observed = np.asarray(observed_counts, dtype=float)
    n = observed.sum()
    if expected_ratios is None:
        expected_ratios = [1.0 / len(observed)] * len(observed)
    expected = np.asarray(expected_ratios, dtype=float) * n

    chi2 = float(((observed - expected) ** 2 / expected).sum())
    dof = len(observed) - 1
    p_value = float(stats.chi2.sf(chi2, dof))

    return {
        "observed": observed_counts,
        "expected": expected.round(1).tolist(),
        "chi2_statistic": chi2,
        "p_value": p_value,
        "srm_detected": p_value < 0.01,   # conventional SRM alarm threshold
        "verdict": (
            "No SRM: arm sizes are consistent with the intended split."
            if p_value >= 0.01
            else "SRM DETECTED: arm sizes deviate from the intended split - investigate."
        ),
    }


def two_proportion_ztest(
    x_treated: int,
    n_treated: int,
    x_control: int,
    n_control: int,
    label: str = "",
    alpha: float = 0.05,
) -> ProportionTestResult:
    """Two-proportion z-test for the difference in a binary rate between two groups.

    Business question: is the treated-vs-control gap in this rate beyond chance?

    Intuition: express the observed gap in units of its own standard error. Under the
    null "same rate in both groups", z ~ Normal(0, 1); a |z| beyond ~1.96 (p < 0.05)
    makes chance an unconvincing explanation.

    Method detail worth knowing:
      * The **test statistic** uses the *pooled* proportion, because under the null the
        two groups share one true rate.
      * The **confidence interval** uses the *unpooled* SE, because a CI should not
        assume the null is true.

    Args:
        x_treated:  successes (e.g. visits) in the treated group.
        n_treated:  size of the treated group.
        x_control:  successes in the control group.
        n_control:  size of the control group.
        label:      human-readable name for the comparison.
        alpha:      significance level (default 0.05 -> 95% CI).

    Returns:
        A :class:`ProportionTestResult`.
    """
    p_treated = x_treated / n_treated
    p_control = x_control / n_control
    lift = p_treated - p_control

    # Test statistic: pooled proportion under H0.
    p_pool = (x_treated + x_control) / (n_treated + n_control)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / n_treated + 1 / n_control))
    z = lift / se_pooled if se_pooled > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))   # two-sided

    # Confidence interval: unpooled SE (does not assume equal rates).
    se_unpooled = np.sqrt(
        p_treated * (1 - p_treated) / n_treated + p_control * (1 - p_control) / n_control
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = lift - z_crit * se_unpooled
    ci_high = lift + z_crit * se_unpooled

    return ProportionTestResult(
        label=label,
        treated_rate=p_treated,
        control_rate=p_control,
        lift=lift,
        ci_low=ci_low,
        ci_high=ci_high,
        z_statistic=float(z),
        p_value=p_value,
        n_treated=n_treated,
        n_control=n_control,
    )
