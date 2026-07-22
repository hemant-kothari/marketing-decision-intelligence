"""M1 runner: load Hillstrom, run every committed SQL query, and print the results.

This is the "see M1 working" entry point. It answers the opening business questions
directly from SQL:
    01  arm sizes + balance check
    02  response rates by arm
    03  raw treated-vs-control lift (binary)
    04  visit lift by customer attribute (recency)
    05  spend distribution (why we don't model spend directly)

Run:
    python scripts/run_m1_sql.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdip.config import load_config  # noqa: E402
from mdip.data.database import list_queries, make_connection, run_query  # noqa: E402
from mdip.data.load import load_hillstrom  # noqa: E402
from mdip.logging_setup import get_logger  # noqa: E402


def main() -> None:
    cfg = load_config()
    log = get_logger("run_m1_sql", cfg.logging.level)

    log.info("Loading and validating Hillstrom ...")
    df = load_hillstrom(cfg)
    log.info("Loaded %d rows, %d columns. Data-quality checks passed.", *df.shape)

    con = make_connection(df, table="campaign")

    for name in list_queries():
        result = run_query(con, name)
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        print(tabulate(result, headers="keys", tablefmt="github", showindex=False))

    print(f"\n{'=' * 78}")
    log.info("M1 complete: %d SQL queries executed.", len(list_queries()))


if __name__ == "__main__":
    main()
