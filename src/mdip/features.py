"""Feature engineering for the predictive and causal models.

We keep feature engineering explicit and in one place so it's easy to explain and reuse.
The theme is the classic marketing **RFM** framing (Recency, Frequency, Monetary), plus
a couple of honest transforms:

  * ``log_history`` — prior-year spend is right-skewed (skew ~2.4), so we log it.
  * ``recent_buyer`` — a simple recency flag (bought in the last 3 months).
  * ``categories_bought`` — mens + womens, a light *frequency* proxy (Hillstrom has no
    purchase-count column, so true "F" is unavailable — we're honest about that).

We deliberately **drop ``history_segment``**: it is a coarse binning of ``history``, so
keeping both would be redundant/collinear and would muddy the logistic-regression
coefficients. ``history`` (continuous) carries strictly more information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Feature groups after engineering. We keep ``log_history`` (de-skewed) and drop raw
# ``history`` — keeping both would be redundant/collinear, the same reason we drop
# ``history_segment``.
NUMERIC_FEATURES = ["recency", "log_history", "categories_bought"]
BINARY_FEATURES = ["mens", "womens", "newbie", "recent_buyer"]
CATEGORICAL_FEATURES = ["zip_code", "channel"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered feature columns added.

    Pure and deterministic: no fitting happens here, so it can be applied to any split
    (train/test) or to new uploaded data without leakage.
    """
    out = df.copy()
    out["log_history"] = np.log1p(out["history"])          # Monetary, de-skewed
    out["recent_buyer"] = (out["recency"] <= 3).astype(int)  # Recency flag
    out["categories_bought"] = out["mens"] + out["womens"]   # light Frequency proxy
    return out


def feature_columns() -> list[str]:
    """The full list of engineered feature columns fed to the models."""
    return NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def make_preprocessor() -> ColumnTransformer:
    """One-hot encode the categorical features; pass the rest through unchanged.

    Wrapped in a ColumnTransformer so it can live inside a scikit-learn Pipeline and be
    fit on the training data only — the standard way to avoid data leakage.
    """
    return ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",  # numeric + binary features pass through
        verbose_feature_names_out=False,
    )
