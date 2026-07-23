"""EDA plots: each function draws one clear figure and returns the matplotlib Figure.

Design choices that make these read well:
  * treated = blue, control = red, consistently (see viz.style).
  * every chart has a title that states the takeaway, and axis labels in plain words.
  * functions return the Figure so a notebook can display it and a script can save it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mdip.eda.profile import response_rate_by
from mdip.viz.style import COLORS


def plot_arm_sizes(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of customers per experiment arm (the balance check, visualised)."""
    counts = df["segment"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.to_numpy(), color=COLORS["neutral"])
    ax.axhline(len(df) / 3, ls="--", color=COLORS["accent"], label="perfect 1/3 split")
    ax.set_title("Experiment arms are balanced (~1/3 each)")
    ax.set_ylabel("customers")
    ax.legend()
    return fig


def plot_history_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram of prior-year spend (`history`), raw vs log — shows right skew."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["history"], bins=50, color=COLORS["treated"])
    axes[0].set_title("history is heavily right-skewed")
    axes[0].set_xlabel("prior-year spend ($)")
    axes[0].set_ylabel("customers")

    axes[1].hist(np.log1p(df["history"]), bins=50, color=COLORS["accent"])
    axes[1].set_title("log(1 + history) is far more symmetric")
    axes[1].set_xlabel("log(1 + prior-year spend)")
    fig.tight_layout()
    return fig


def plot_response_by_arm(df: pd.DataFrame) -> plt.Figure:
    """Grouped bars: visit and conversion rate, treated vs control."""
    outcomes = ["visit", "conversion"]
    treated = [df.loc[df["treatment"] == 1, o].mean() * 100 for o in outcomes]
    control = [df.loc[df["treatment"] == 0, o].mean() * 100 for o in outcomes]

    x = np.arange(len(outcomes))
    w = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - w / 2, control, w, label="control (no email)", color=COLORS["control"])
    ax.bar(x + w / 2, treated, w, label="treated (emailed)", color=COLORS["treated"])
    ax.set_xticks(x, [o.capitalize() for o in outcomes])
    ax.set_ylabel("rate (%)")
    ax.set_title("Emailed customers visit and convert more (raw rates)")
    ax.legend()
    for i, (c, t) in enumerate(zip(control, treated)):
        ax.text(i - w / 2, c, f"{c:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, t, f"{t:.1f}", ha="center", va="bottom", fontsize=8)
    return fig


def plot_lift_by_attribute(df: pd.DataFrame, by: str) -> plt.Figure:
    """Descriptive visit-rate lift (treated - control) within each category of `by`."""
    tbl = response_rate_by(df, by=by, outcome="visit").sort_values("lift")
    fig, ax = plt.subplots()
    ax.barh(tbl[by].astype(str), tbl["lift"] * 100, color=COLORS["accent"])
    ax.set_xlabel("visit-rate lift (percentage points)")
    ax.set_title(f"Descriptive email lift by {by}\n(correlational, not yet causal)")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Correlation heatmap among numeric/binary features (spot redundancy & structure)."""
    cols = ["recency", "history", "mens", "womens", "newbie", "visit", "conversion"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Feature correlations")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig
