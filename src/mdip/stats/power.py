"""Statistical power and Minimum Detectable Effect (MDE) for a two-proportion test.

Business question: if email had only a small true effect, could this experiment even
have detected it? And how many customers would a *future* test need to detect a target
lift? Power analysis is what turns "we found nothing" into either "there is nothing" or
"we were underpowered to see a small something".

Intuition: power is the probability of correctly detecting a real effect of a given
size. A small effect hides in the noise unless the sample is large. MDE is the smallest
effect our sample size could reliably catch (at the chosen power).

Assumptions: same Normal-approximation setup as the z-test; a chosen significance level
(alpha) and target power. We use statsmodels' NormalIndPower here — power math is fiddly
and this is the standard, trusted implementation.
"""

from __future__ import annotations

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

_ANALYSIS = NormalIndPower()


def required_sample_size(
    baseline_rate: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Customers needed **per group** to detect a given absolute lift.

    Args:
        baseline_rate: control-group rate (e.g. 0.106 for visits).
        mde_absolute:  absolute lift to detect (e.g. 0.01 for +1 percentage point).
        alpha:         significance level.
        power:         desired power (probability of detecting a true effect).

    Returns:
        Required sample size per group (rounded up).
    """
    effect = proportion_effectsize(baseline_rate + mde_absolute, baseline_rate)
    n = _ANALYSIS.solve_power(effect_size=effect, alpha=alpha, power=power, ratio=1.0)
    return int(-(-n // 1))  # ceil


def detectable_effect(
    baseline_rate: float,
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest absolute lift detectable with a given per-group sample size (the MDE).

    Returns:
        The minimum detectable absolute lift (in proportion units).
    """
    # Solve for the effect size (Cohen's h) achievable at this n, power, alpha.
    effect_h = _ANALYSIS.solve_power(
        effect_size=None, nobs1=n_per_group, alpha=alpha, power=power, ratio=1.0
    )
    # Convert Cohen's h back to an absolute rate difference at this baseline.
    # h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1)); invert for p2 given p1 and h.
    import numpy as np

    p1 = baseline_rate
    p2 = np.sin(effect_h / 2 + np.arcsin(np.sqrt(p1))) ** 2
    return float(p2 - p1)
