"""Load the dataset into DuckDB and run the committed SQL analysis queries.

Why DuckDB? It runs real SQL directly on a DataFrame with zero setup — no server,
no credentials. That lets the project *demonstrate SQL* (the skill recruiters filter
on) instead of doing every aggregation in pandas. The SQL lives in ``sql/*.sql`` as
plain, readable files so it can be reviewed and reused on its own.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SQL_DIR = ROOT / "sql"


def make_connection(df: pd.DataFrame, table: str = "campaign") -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection with ``df`` registered as a table.

    Args:
        df: The campaign data (typically the output of ``load_hillstrom``).
        table: Name the table is registered under (default ``"campaign"``).
    """
    con = duckdb.connect(database=":memory:")
    con.register(table, df)
    return con


def read_query(name: str) -> str:
    """Read a committed SQL file from ``sql/`` by name (with or without ``.sql``)."""
    filename = name if name.endswith(".sql") else f"{name}.sql"
    path = SQL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")


def run_query(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """Execute a committed SQL query by name and return the result as a DataFrame."""
    return con.execute(read_query(name)).df()


def list_queries() -> list[str]:
    """Return the names (without extension) of all committed SQL queries."""
    return sorted(p.stem for p in SQL_DIR.glob("*.sql"))
