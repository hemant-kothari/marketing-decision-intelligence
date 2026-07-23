"""Tests for the M2 EDA profiling functions.

We test the summary logic (rates and lifts) because a bug there would quietly mislead
the narrative. We do not test the plotting functions' pixels — only that they run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdip.config import load_config
from mdip.data.load import load_hillstrom
from mdip.eda import plots, profile

CFG = load_config()
skip_if_no_data = pytest.mark.skipif(
    not CFG.path("raw_data").exists(),
    reason="Run `python scripts/download_data.py` first.",
)


@pytest.fixture(scope="module")
def df():
    return load_hillstrom(CFG)


@skip_if_no_data
def test_response_rate_by_lift_equals_treated_minus_control(df):
    tbl = profile.response_rate_by(df, by="channel", outcome="visit")
    # lift must equal treated_rate - control_rate for every row.
    assert ((tbl["treated_rate"] - tbl["control_rate"] - tbl["lift"]).abs() < 1e-12).all()
    # customers per group must sum back to the full dataset.
    assert tbl["customers"].sum() == len(df)


@skip_if_no_data
def test_outcome_base_rates_match_direct_means(df):
    tbl = profile.outcome_base_rates(df).set_index("outcome")
    treated_visit = df.loc[df["treatment"] == 1, "visit"].mean()
    assert tbl.loc["visit", "treated"] == pytest.approx(treated_visit)


@skip_if_no_data
def test_numeric_summary_flags_history_skew(df):
    summ = profile.numeric_summary(df)
    # history is known to be strongly right-skewed; this guards the EDA claim.
    assert summ.loc["history", "skew"] > 1.0


@skip_if_no_data
@pytest.mark.parametrize(
    "plotter", ["plot_arm_sizes", "plot_response_by_arm", "plot_correlation_heatmap"]
)
def test_plots_run_without_error(df, plotter):
    fig = getattr(plots, plotter)(df)
    assert fig is not None
