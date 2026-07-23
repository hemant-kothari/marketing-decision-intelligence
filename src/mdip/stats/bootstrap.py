"""Bootstrap confidence intervals for a difference in means.

Business question: what is the range of plausible effect sizes — especially for spend,
where the outcome is 99% zeros with a heavy tail and the Normal-approximation formulas
behave poorly?

Intuition: instead of trusting a formula, we simulate "running the experiment again"
thousands of times by resampling the data we have (with replacement), recompute the
lift each time, and read the middle 95% of those results as the confidence interval.

Assumptions: the sample represents the population (true here — it's an RCT) and
observations are independent. No Normality assumption — that is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    label: str
    lift: float          # observed difference in means (treated - control)
    ci_low: float
    ci_high: float
    n_boot: int

    @property
    def significant(self) -> bool:
        """True if the confidence interval excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0


def bootstrap_diff_means(
    treated_values: np.ndarray,
    control_values: np.ndarray,
    label: str = "",
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI for the difference in mean outcome between two groups.

    Works for any numeric outcome: a 0/1 rate (visit) or a skewed continuous one (spend).

    Args:
        treated_values: outcome values for the treated group.
        control_values: outcome values for the control group.
        label:          name for the comparison.
        n_boot:         number of bootstrap resamples.
        alpha:          1 - confidence level (0.05 -> 95% CI).
        seed:           RNG seed for reproducibility.

    Returns:
        A :class:`BootstrapResult`.
    """
    rng = np.random.default_rng(seed)
    treated_values = np.asarray(treated_values, dtype=float)
    control_values = np.asarray(control_values, dtype=float)
    n_t, n_c = len(treated_values), len(control_values)

    observed_lift = treated_values.mean() - control_values.mean()

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        # Resample each group independently, with replacement, to its own size.
        t_sample = treated_values[rng.integers(0, n_t, n_t)]
        c_sample = control_values[rng.integers(0, n_c, n_c)]
        diffs[i] = t_sample.mean() - c_sample.mean()

    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return BootstrapResult(
        label=label,
        lift=float(observed_lift),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_boot=n_boot,
    )
