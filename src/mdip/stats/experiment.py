"""Compose the individual tests into one experiment readout.

This is the "answer Q1" module: given the campaign data, it runs the SRM check, the
proportion tests for the binary outcomes, a bootstrap CI for spend, applies a
Benjamini-Hochberg correction across the family of tests, and returns tidy results.
Keeping the orchestration here (not in a script) makes it testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mdip.stats.bootstrap import bootstrap_diff_means
from mdip.stats.multiple_testing import benjamini_hochberg
from mdip.stats.proportion import srm_check, two_proportion_ztest

# The two email arms we compare against the shared control, and the binary outcomes.
EMAIL_ARMS = ["Mens E-Mail", "Womens E-Mail"]
BINARY_OUTCOMES = ["visit", "conversion"]


def run_readout(df: pd.DataFrame, seed: int = 42, alpha: float = 0.05) -> dict:
    """Run the full experiment readout and return structured results.

    Returns a dict with:
        * ``srm``:          the SRM check result.
        * ``tests``:        a tidy DataFrame, one row per (arm, outcome) test, with lift,
                            CI, p-value, and BH-adjusted p-value / decision.
    """
    control = df[df["segment"] == "No E-Mail"]

    rows = []
    for arm in EMAIL_ARMS:
        treated = df[df["segment"] == arm]

        # Binary outcomes -> two-proportion z-test.
        for outcome in BINARY_OUTCOMES:
            res = two_proportion_ztest(
                x_treated=int(treated[outcome].sum()),
                n_treated=len(treated),
                x_control=int(control[outcome].sum()),
                n_control=len(control),
                label=f"{arm} | {outcome}",
                alpha=alpha,
            )
            rows.append(
                {
                    "arm": arm,
                    "outcome": outcome,
                    "test": "two-proportion z",
                    "treated_rate": res.treated_rate,
                    "control_rate": res.control_rate,
                    "lift": res.lift,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "p_value": res.p_value,
                }
            )

        # Spend (skewed, continuous) -> bootstrap CI (no Normality assumption).
        boot = bootstrap_diff_means(
            treated_values=treated["spend"].to_numpy(),
            control_values=control["spend"].to_numpy(),
            label=f"{arm} | spend",
            seed=seed,
            alpha=alpha,
        )
        # A bootstrap CI has no single p-value; we record a p from whether 0 is inside
        # the interval by mapping the CI to an approximate two-sided p via a normal SE.
        se = (boot.ci_high - boot.ci_low) / (2 * 1.959963985)
        z = boot.lift / se if se > 0 else 0.0
        from scipy import stats as _stats

        rows.append(
            {
                "arm": arm,
                "outcome": "spend",
                "test": "bootstrap diff-in-means",
                "treated_rate": treated["spend"].mean(),
                "control_rate": control["spend"].mean(),
                "lift": boot.lift,
                "ci_low": boot.ci_low,
                "ci_high": boot.ci_high,
                "p_value": float(2 * _stats.norm.sf(abs(z))),
            }
        )

    tests = pd.DataFrame(rows)

    # Multiple-testing correction across the whole family of tests.
    bh = benjamini_hochberg(tests["p_value"].tolist(), alpha=alpha)
    tests["p_adjusted"] = np.round(bh["adjusted"], 5)
    tests["significant_bh"] = bh["reject"]

    srm = srm_check(
        observed_counts=df["segment"].value_counts().reindex(
            ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
        ).tolist()
    )

    return {"srm": srm, "tests": tests}
