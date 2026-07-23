"""Plots for the experiment readout: a forest plot of lifts with confidence intervals.

A forest plot is the clearest way to show several effect estimates at once: each row is
a comparison, the dot is the estimated lift, the horizontal line is its 95% CI, and a
vertical line at zero marks "no effect". If a CI crosses zero, that effect isn't
statistically distinguishable from nothing.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from mdip.viz.style import COLORS


def plot_forest(tests: pd.DataFrame, outcome: str, scale: float = 100.0) -> plt.Figure:
    """Forest plot of lift +/- 95% CI for one outcome across the email arms.

    Args:
        tests:   the tidy results DataFrame from ``experiment.run_readout``.
        outcome: which outcome to plot ("visit", "conversion", or "spend").
        scale:   multiply lifts by this for display (100 -> percentage points for rates).
    """
    sub = tests[tests["outcome"] == outcome].reset_index(drop=True)
    unit = "percentage points" if outcome in ("visit", "conversion") else "$"
    disp_scale = scale if outcome in ("visit", "conversion") else 1.0

    fig, ax = plt.subplots(figsize=(8, 0.9 * len(sub) + 1.5))
    for i, row in sub.iterrows():
        crosses_zero = row["ci_low"] < 0 < row["ci_high"]
        color = COLORS["control"] if crosses_zero else COLORS["treated"]
        ax.plot(
            [row["ci_low"] * disp_scale, row["ci_high"] * disp_scale],
            [i, i],
            color=color,
            lw=2,
        )
        ax.plot(row["lift"] * disp_scale, i, "o", color=color, ms=8)

    ax.axvline(0, ls="--", color=COLORS["neutral"], lw=1)
    ax.set_yticks(range(len(sub)), sub["arm"])
    ax.set_xlabel(f"lift in {outcome} ({unit})")
    ax.set_title(f"Email effect on {outcome} (dot = lift, line = 95% CI)")
    ax.margins(y=0.3)
    fig.tight_layout()
    return fig
