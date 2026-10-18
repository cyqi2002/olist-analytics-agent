"""Database access. The connection is read-only, so even a query that slips past
the validator can't modify data."""

import threading
from functools import lru_cache

import duckdb
import pandas as pd

from agent.config import DB_PATH, QUERY_TIMEOUT_S


class QueryTimeout(Exception):
    pass


@lru_cache(maxsize=1)
def _connection() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found - run `python data/load.py` first")
    return duckdb.connect(str(DB_PATH), read_only=True)


def cursor() -> duckdb.DuckDBPyConnection:
    # a cursor per query keeps concurrent callers (Streamlit, eval threads) independent
    return _connection().cursor()


def run_query(sql: str, timeout_s: float = QUERY_TIMEOUT_S) -> pd.DataFrame:
    """Run a query, interrupting it if it takes longer than timeout_s."""
    cur = cursor()
    timer = threading.Timer(timeout_s, cur.interrupt)
    timer.start()
    try:
        return cur.execute(sql).df()
    except duckdb.InterruptException as e:
        raise QueryTimeout(f"query exceeded {timeout_s}s timeout") from e
    finally:
        timer.cancel()
        cur.close()


@lru_cache(maxsize=1)
def catalog() -> dict[str, dict[str, str]]:
    """{table: {column: type}} for every table in the database."""
    rows = cursor().execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
    ).fetchall()
    out: dict[str, dict[str, str]] = {}
    for table, column, dtype in rows:
        out.setdefault(table, {})[column] = dtype
    return out
