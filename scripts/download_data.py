"""Download the Hillstrom email-marketing dataset and cache it locally.

We fetch the dataset once via scikit-uplift's official loader and save a single
tidy CSV to ``data/raw/hillstrom.csv``. Every later stage reads that local file, so
the analysis is reproducible and never depends on the network again.

Run:
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make ``src/`` importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdip.config import load_config  # noqa: E402
from mdip.logging_setup import get_logger  # noqa: E402


def download() -> Path:
    """Fetch Hillstrom (features + all three outcomes) and cache it as one CSV."""
    cfg = load_config()
    log = get_logger("download_data", cfg.logging.level)

    out_path = cfg.path("raw_data")
    if out_path.exists():
        log.info("Dataset already cached at %s (delete it to re-download).", out_path)
        return out_path

    log.info("Fetching Hillstrom via scikit-uplift ...")
    from sklift.datasets import fetch_hillstrom

    bunch = fetch_hillstrom(target_col="all")
    features: pd.DataFrame = bunch["data"]
    outcomes: pd.DataFrame = bunch["target"]        # visit, conversion, spend
    treatment: pd.Series = bunch["treatment"]        # 'Mens E-Mail' / 'Womens E-Mail' / 'No E-Mail'

    # Assemble one flat table: features + treatment + outcomes.
    df = features.copy()
    df[bunch["treatment_name"]] = treatment.to_numpy()
    for col in outcomes.columns:
        df[col] = outcomes[col].to_numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Saved %d rows x %d cols to %s", df.shape[0], df.shape[1], out_path)
    return out_path


if __name__ == "__main__":
    download()
