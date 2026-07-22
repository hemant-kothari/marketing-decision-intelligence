"""Tests for the M1 data layer: loading, validation, and SQL execution.

We test the data layer because a silent bug here (wrong treatment coding, dropped
rows) would quietly corrupt every downstream statistic. These are the checks worth
having; we do not test third-party libraries themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdip.config import load_config
from mdip.data.database import make_connection, run_query
from mdip.data.load import DataQualityError, load_hillstrom

CFG = load_config()
DATA_MISSING = not CFG.path("raw_data").exists()
skip_if_no_data = pytest.mark.skipif(
    DATA_MISSING, reason="Run `python scripts/download_data.py` first."
)


@skip_if_no_data
def test_load_shape_and_columns():
    df = load_hillstrom(CFG)
    assert len(df) == 64_000
    assert "treatment" in df.columns


@skip_if_no_data
def test_treatment_is_binary_and_correctly_coded():
    df = load_hillstrom(CFG)
    # treatment must be exactly {0, 1}
    assert set(df["treatment"].unique()) == {0, 1}
    # every 'No E-Mail' row is control; every emailed row is treated.
    assert (df.loc[df["segment"] == "No E-Mail", "treatment"] == 0).all()
    assert (df.loc[df["segment"] != "No E-Mail", "treatment"] == 1).all()
    # roughly 2/3 treated (two email arms out of three).
    assert 0.66 < df["treatment"].mean() < 0.68


@skip_if_no_data
def test_quality_check_catches_corrupt_data():
    df = load_hillstrom(CFG)
    df_bad = df.iloc[:100].copy()  # wrong row count -> should fail validation
    from mdip.data.load import _check_quality

    with pytest.raises(DataQualityError):
        _check_quality(df_bad, CFG)


@skip_if_no_data
def test_raw_lift_query_matches_manual_computation():
    """The SQL raw-lift must equal the same number computed directly in pandas."""
    df = load_hillstrom(CFG)
    con = make_connection(df, table="campaign")
    sql_result = run_query(con, "03_raw_lift_binary")
    sql_lift = sql_result["visit_lift_pp"].iloc[0]

    treated = df.loc[df["treatment"] == 1, "visit"].mean()
    control = df.loc[df["treatment"] == 0, "visit"].mean()
    manual_lift = round((treated - control) * 100, 3)

    assert sql_lift == pytest.approx(manual_lift, abs=0.01)
