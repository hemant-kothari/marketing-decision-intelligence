"""Load the cached Hillstrom dataset, validate it, and add analysis columns.

Two responsibilities, kept separate and simple:

1. ``load_hillstrom`` reads the cached CSV and runs a small set of **data-quality
   checks**. Checking data before analysis is a core Data-Science habit: it catches
   silent problems (missing values, unexpected categories, wrong row counts) that
   would otherwise corrupt every downstream number.

2. It adds two convenience columns used throughout the project:
     * ``treatment`` — the 3-arm ``segment`` collapsed to binary (1 = any email, 0 = control).
     * (that's it — we keep raw columns otherwise, so the data stays recognisable.)

There is no heavyweight schema framework here on purpose: a handful of explicit
``assert``-style checks is easy to read and easy to explain in an interview.
"""

from __future__ import annotations

import pandas as pd

from mdip.config import Config

# What we expect the raw file to look like. Stated explicitly so a mismatch is loud.
EXPECTED_COLUMNS = [
    "recency", "history_segment", "history", "mens", "womens",
    "zip_code", "newbie", "channel", "segment", "visit", "conversion", "spend",
]
EXPECTED_ROWS = 64_000
EXPECTED_SEGMENTS = {"Mens E-Mail", "Womens E-Mail", "No E-Mail"}


class DataQualityError(ValueError):
    """Raised when the loaded data fails a quality check."""


def _check_quality(df: pd.DataFrame, cfg: Config) -> None:
    """Run explicit data-quality checks; raise on the first failure.

    Each check maps to a question an interviewer might ask: "How did you know the
    data was clean?" — "I verified the row count, columns, missing values, treatment
    labels, and that the binary outcomes really are 0/1."
    """
    problems: list[str] = []

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        problems.append(f"missing columns: {sorted(missing_cols)}")

    if len(df) != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS} rows, got {len(df)}")

    null_counts = df.isnull().sum()
    if null_counts.any():
        problems.append(f"unexpected nulls: {null_counts[null_counts > 0].to_dict()}")

    actual_segments = set(df[cfg.data.treatment_col].unique())
    if actual_segments != EXPECTED_SEGMENTS:
        problems.append(f"unexpected treatment labels: {actual_segments}")

    for binary_col in ("visit", "conversion", "mens", "womens", "newbie"):
        bad = set(df[binary_col].unique()) - {0, 1}
        if bad:
            problems.append(f"column '{binary_col}' has non-binary values: {bad}")

    if (df["spend"] < 0).any():
        problems.append("negative spend values found")

    if problems:
        raise DataQualityError("Data quality checks failed:\n  - " + "\n  - ".join(problems))


def load_hillstrom(cfg: Config, validate: bool = True) -> pd.DataFrame:
    """Load the cached Hillstrom CSV, validate it, and add the binary ``treatment`` column.

    Args:
        cfg: Project configuration.
        validate: If True (default), run data-quality checks and raise on failure.

    Returns:
        The dataset as a DataFrame with an added integer ``treatment`` column
        (1 = received any email, 0 = control / no email).
    """
    path = cfg.path("raw_data")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )

    df = pd.read_csv(path)

    if validate:
        _check_quality(df, cfg)

    # Collapse the 3-arm experiment to a clean binary treatment for the main analysis.
    df["treatment"] = (
        df[cfg.data.treatment_col].isin(cfg.data.treated_labels).astype(int)
    )

    return df
