"""EDA profiling: small functions that return summary tables as DataFrames.

Kept as plain functions that each answer one question, so they are easy to call from
a notebook and easy to test. No plotting here — tables only. Plots live in ``plots.py``.
"""

from __future__ import annotations

import pandas as pd

# Customer attributes we profile throughout the project.
CATEGORICAL_FEATURES = ["history_segment", "zip_code", "channel", "newbie", "mens", "womens"]
NUMERIC_FEATURES = ["recency", "history"]


def numeric_summary(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Return count/mean/std/min/median/max plus skew for numeric columns.

    Skew is included because a highly skewed variable (like ``history``) may need a
    log transform later — spotting that is a core EDA judgement.
    """
    cols = cols or NUMERIC_FEATURES
    desc = df[cols].describe().T
    desc["skew"] = df[cols].skew()
    return desc[["count", "mean", "std", "min", "50%", "max", "skew"]].rename(
        columns={"50%": "median"}
    )


def categorical_summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return the count and share of each category in a column."""
    counts = df[col].value_counts(dropna=False)
    return pd.DataFrame(
        {
            col: counts.index,
            "customers": counts.to_numpy(),
            "share_pct": (counts.to_numpy() / len(df) * 100).round(2),
        }
    )


def response_rate_by(df: pd.DataFrame, by: str, outcome: str = "visit") -> pd.DataFrame:
    """Descriptive response rate for an outcome, split by treatment, within groups of ``by``.

    Returns one row per category of ``by`` with treated rate, control rate, and their
    difference (the *descriptive* per-group lift). This is a correlation, not a causal
    subgroup effect — that distinction is made explicit wherever this is used.

    Args:
        df: Data with a binary ``treatment`` column and the outcome column.
        by: Customer attribute to group by (e.g. ``"channel"``).
        outcome: Binary outcome column, default ``"visit"``.
    """
    g = df.groupby([by, "treatment"])[outcome].mean().unstack("treatment")
    g.columns = ["control_rate", "treated_rate"]
    g["lift"] = g["treated_rate"] - g["control_rate"]
    counts = df.groupby(by).size().rename("customers")
    out = g.join(counts).reset_index()
    return out[[by, "customers", "control_rate", "treated_rate", "lift"]]


def outcome_base_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Overall treated vs control rates for visit and conversion, and mean spend."""
    rows = []
    for outcome in ["visit", "conversion", "spend"]:
        treated = df.loc[df["treatment"] == 1, outcome].mean()
        control = df.loc[df["treatment"] == 0, outcome].mean()
        rows.append(
            {
                "outcome": outcome,
                "treated": treated,
                "control": control,
                "difference": treated - control,
            }
        )
    return pd.DataFrame(rows)
