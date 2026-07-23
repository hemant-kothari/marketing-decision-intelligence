"""Predictive response model: P(visit | customer attributes) among emailed customers.

This is the *conventional* marketing model — "who is likely to respond?" — and we build
it deliberately as the **baseline the causal work (M5-M6) must beat**. It ranks customers
by predicted response, but some high-response customers would have visited anyway, so
targeting on this alone wastes budget. That contrast is the whole point of the project.

Two models, same features, honest train/test split:
  * Logistic regression — interpretable, reported via odds ratios.
  * LightGBM — stronger if it actually wins on PR-AUC; explained with SHAP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mdip.features import (
    BINARY_FEATURES,
    NUMERIC_FEATURES,
    engineer_features,
    feature_columns,
    make_preprocessor,
)


@dataclass(frozen=True)
class ModelEvaluation:
    """Held-out evaluation metrics for a response model."""

    name: str
    pr_auc: float          # average precision — primary metric under class imbalance
    roc_auc: float
    brier: float           # calibration/accuracy of the probabilities (lower is better)
    positive_rate: float   # base rate of visitors in the test set (for context)


def build_response_dataset(df: pd.DataFrame, outcome: str = "visit"):
    """Return (X, y) for the response model: emailed customers only.

    We train on the treated group because the business question is "if we email this
    customer, how likely are they to visit?".
    """
    treated = df[df["treatment"] == 1].copy()
    treated = engineer_features(treated)
    X = treated[feature_columns()]
    y = treated[outcome].astype(int)
    return X, y


def _split(X, y, seed: int):
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)


def train_logistic(X_train, y_train, seed: int = 42) -> Pipeline:
    """Fit a logistic-regression pipeline (preprocess -> standardise-free linear model)."""
    model = Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, seed: int = 42) -> Pipeline:
    """Fit a LightGBM pipeline.

    These features carry weak signal for predicting a visit, so an unregularised GBM
    overfits. We use a deliberately conservative configuration (shallow trees, large
    ``min_child_samples``, L2 regularisation) to give it a *fair* comparison against
    logistic regression rather than letting it overfit and lose.
    """
    model = Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    num_leaves=7,
                    min_child_samples=300,
                    reg_lambda=2.0,
                    random_state=seed,
                    verbose=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: Pipeline, X_test, y_test, name: str) -> ModelEvaluation:
    """Compute held-out PR-AUC, ROC-AUC and Brier score for a fitted model."""
    proba = model.predict_proba(X_test)[:, 1]
    return ModelEvaluation(
        name=name,
        pr_auc=float(average_precision_score(y_test, proba)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        brier=float(brier_score_loss(y_test, proba)),
        positive_rate=float(y_test.mean()),
    )


def odds_ratios(logistic_model: Pipeline) -> pd.DataFrame:
    """Return odds ratios for the fitted logistic model, sorted by strength.

    Odds ratio = exp(coefficient). >1 means the feature increases the odds of a visit;
    <1 means it decreases them. This is the interpretable business output.
    """
    prep = logistic_model.named_steps["prep"]
    clf = logistic_model.named_steps["clf"]
    names = prep.get_feature_names_out()
    coefs = clf.coef_[0]
    out = pd.DataFrame(
        {"feature": names, "coefficient": coefs, "odds_ratio": np.exp(coefs)}
    )
    return out.reindex(out["odds_ratio"].sub(1).abs().sort_values(ascending=False).index)


def calibration_points(model: Pipeline, X_test, y_test, n_bins: int = 10):
    """Return (mean_predicted, fraction_positive) for a calibration curve."""
    proba = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


def numeric_feature_names_after_prep() -> list[str]:
    """Numeric+binary feature names that pass through the preprocessor unchanged."""
    return NUMERIC_FEATURES + BINARY_FEATURES


def shap_values_for_lightgbm(lgbm_model: Pipeline, X, sample: int = 2000, seed: int = 0):
    """Compute SHAP values for the LightGBM pipeline's classifier.

    SHAP explains each prediction as feature contributions, recovering the
    interpretability we trade away by using a boosted-tree model. Returns
    ``(shap_values, feature_matrix, feature_names)`` on a random sample for speed.
    """
    import shap

    prep = lgbm_model.named_steps["prep"]
    clf = lgbm_model.named_steps["clf"]
    X_prep = prep.transform(X)
    names = list(prep.get_feature_names_out())

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_prep), size=min(sample, len(X_prep)), replace=False)
    X_sample = X_prep[idx]

    values = shap.TreeExplainer(clf).shap_values(X_sample)
    # LightGBM binary classifiers return a list [class0, class1]; take the positive class.
    values = values[1] if isinstance(values, list) else values
    return values, X_sample, names


def train_and_evaluate_both(df: pd.DataFrame, seed: int = 42) -> dict:
    """Convenience: build data, split, train both models, evaluate. Returns everything."""
    X, y = build_response_dataset(df)
    X_train, X_test, y_train, y_test = _split(X, y, seed)

    logit = train_logistic(X_train, y_train, seed)
    gbm = train_lightgbm(X_train, y_train, seed)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "logistic": logit,
        "lightgbm": gbm,
        "eval_logistic": evaluate(logit, X_test, y_test, "Logistic Regression"),
        "eval_lightgbm": evaluate(gbm, X_test, y_test, "LightGBM"),
        "odds_ratios": odds_ratios(logit),
    }
