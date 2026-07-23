"""Tests for M4 feature engineering and the response model.

We test the feature logic (deterministic, leakage-free) and that the model pipeline
trains and produces sane probabilities. We do not assert exact metric values (those
depend on library internals) — only properties that must hold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdip.config import load_config
from mdip.data.load import load_hillstrom
from mdip.features import engineer_features, feature_columns
from mdip.response_model import _split, build_response_dataset, evaluate, train_logistic

CFG = load_config()
skip_if_no_data = pytest.mark.skipif(
    not CFG.path("raw_data").exists(),
    reason="Run `python scripts/download_data.py` first.",
)


@skip_if_no_data
def test_engineer_features_are_deterministic_and_correct():
    df = load_hillstrom(CFG).head(1000)
    a = engineer_features(df)
    b = engineer_features(df)
    # deterministic
    assert (a["log_history"] == b["log_history"]).all()
    # log_history = log1p(history)
    assert np.allclose(a["log_history"], np.log1p(df["history"]))
    # recent_buyer flag matches definition
    assert (a["recent_buyer"] == (df["recency"] <= 3).astype(int)).all()
    # no feature column is the dropped raw 'history' or 'history_segment'
    assert "history" not in feature_columns()
    assert "history_segment" not in feature_columns()


@skip_if_no_data
def test_response_dataset_is_treated_only():
    df = load_hillstrom(CFG)
    X, y = build_response_dataset(df)
    # only emailed customers, and label is binary
    assert len(X) == int(df["treatment"].sum())
    assert set(np.unique(y)) <= {0, 1}


@skip_if_no_data
def test_logistic_produces_valid_probabilities_and_beats_no_skill():
    df = load_hillstrom(CFG)
    X, y = build_response_dataset(df)
    X_train, X_test, y_train, y_test = _split(X, y, CFG.seed)
    model = train_logistic(X_train, y_train, CFG.seed)
    proba = model.predict_proba(X_test)[:, 1]
    # probabilities in [0, 1]
    assert proba.min() >= 0 and proba.max() <= 1
    # PR-AUC must beat the no-skill baseline (the positive rate)
    ev = evaluate(model, X_test, y_test, "logit")
    assert ev.pr_auc > ev.positive_rate
