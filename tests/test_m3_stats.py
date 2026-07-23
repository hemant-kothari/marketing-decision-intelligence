"""Tests for the M3 statistics module.

The important pattern here: we cross-check our hand-written implementations (the
two-proportion z-test and Benjamini-Hochberg) against statsmodels. If ours matches the
trusted library, we know the from-scratch code is correct — and we still get to explain
the maths in an interview.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdip.stats.bootstrap import bootstrap_diff_means
from mdip.stats.multiple_testing import benjamini_hochberg
from mdip.stats.proportion import srm_check, two_proportion_ztest


def test_ztest_matches_statsmodels():
    """Our two-proportion z-test p-value must match statsmodels' pooled z-test."""
    x_t, n_t, x_c, n_c = 3200, 21307, 2260, 21306
    ours = two_proportion_ztest(x_t, n_t, x_c, n_c)
    sm_stat, sm_p = proportions_ztest([x_t, x_c], [n_t, n_c])  # pooled by default
    assert ours.z_statistic == pytest.approx(sm_stat, rel=1e-6)
    assert ours.p_value == pytest.approx(sm_p, rel=1e-6)


def test_ztest_ci_excludes_zero_for_clear_effect():
    res = two_proportion_ztest(3200, 21307, 2260, 21306)
    assert res.lift > 0
    assert res.ci_low > 0  # a clear positive effect -> CI above zero
    assert res.significant


def test_ztest_null_effect_is_not_significant():
    """Identical rates in both groups -> lift ~0, CI straddles zero."""
    res = two_proportion_ztest(1000, 10000, 1000, 10000)
    assert res.lift == pytest.approx(0.0, abs=1e-9)
    assert res.ci_low < 0 < res.ci_high
    assert not res.significant


def test_bh_matches_statsmodels():
    """Our Benjamini-Hochberg reject/adjusted must match statsmodels' fdr_bh."""
    p_values = [0.001, 0.008, 0.04, 0.2, 0.7, 0.9]
    ours = benjamini_hochberg(p_values, alpha=0.05)
    reject_sm, adj_sm, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    assert ours["reject"] == reject_sm.tolist()
    assert np.allclose(ours["adjusted"], adj_sm)


def test_srm_passes_on_balanced_arms():
    res = srm_check([21307, 21387, 21306])
    assert not res["srm_detected"]
    assert res["p_value"] > 0.05


def test_srm_flags_imbalanced_arms():
    res = srm_check([25000, 20000, 19000])
    assert res["srm_detected"]
    assert res["p_value"] < 0.01


def test_bootstrap_ci_brackets_known_difference():
    """Bootstrap CI should contain the true difference for a simple constructed case."""
    rng = np.random.default_rng(0)
    treated = rng.normal(1.0, 1.0, 5000)   # mean ~1
    control = rng.normal(0.0, 1.0, 5000)   # mean ~0, true diff = 1
    res = bootstrap_diff_means(treated, control, n_boot=2000, seed=1)
    assert res.ci_low < 1.0 < res.ci_high
    assert res.significant
