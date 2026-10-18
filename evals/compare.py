"""Execution-accuracy comparison: does the predicted result answer the question
the same way the gold result does?

Rules, chosen to forgive presentation and punish wrong answers:
- row order is ignored; the number of rows must match
- column names are ignored; extra predicted columns are allowed
- every gold column must be matched by some predicted column
- numbers match within a small tolerance (rounding to 2 decimals is fine), and a
  ratio may be expressed as a percentage (0.12 vs 12)
- dates match whether returned as timestamps or as 'YYYY-MM' / 'YYYY-MM-DD' text
- compare: numeric_only skips text/boolean gold columns (for free-form labels)
"""

import datetime as dt
import math
import re
from decimal import Decimal

import numpy as np
import pandas as pd

DATE_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?(?:[ T]00:00:00(?:\.0+)?)?$")


def _norm(value):
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, Decimal, np.integer, np.floating)):
        return float(value)
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        value = value.strftime("%Y-%m-%d %H:%M:%S") if isinstance(value, dt.datetime) else value.isoformat()
    text = str(value).strip()
    m = DATE_RE.match(text)
    if m:
        year, month, day = m.groups()
        return f"{year}-{month}" if day in (None, "01") else f"{year}-{month}-{day}"
    return text.lower()


def _equal(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-3, abs_tol=0.01)
    return a == b


def _column(df: pd.DataFrame, i: int) -> list:
    return [_norm(v) for v in df.iloc[:, i].tolist()]


def _is_numeric(col: list) -> bool:
    return all(v is None or isinstance(v, float) for v in col) and any(isinstance(v, float) for v in col)


def _scaled(col: list, factor: float) -> list:
    return [v * factor if isinstance(v, float) else v for v in col]


def _same_multiset(a: list, b: list) -> bool:
    remaining = list(b)
    for x in a:
        for j, y in enumerate(remaining):
            if _equal(x, y):
                del remaining[j]
                break
        else:
            return False
    return True


def results_match(pred: pd.DataFrame, gold: pd.DataFrame, numeric_only: bool = False) -> tuple[bool, str]:
    if len(pred) != len(gold):
        return False, f"row count {len(pred)} != {len(gold)}"

    gold_cols = [_column(gold, i) for i in range(gold.shape[1])]
    pred_cols = [_column(pred, i) for i in range(pred.shape[1])]
    if numeric_only:
        gold_cols = [c for c in gold_cols if _is_numeric(c)]

    # assign each gold column a distinct predicted column with the same values
    chosen: list[list] = []
    used: set[int] = set()
    for g_idx, g in enumerate(gold_cols):
        for p_idx, p in enumerate(pred_cols):
            if p_idx in used:
                continue
            variants = [p, _scaled(p, 0.01), _scaled(p, 100.0)] if _is_numeric(p) else [p]
            match = next((v for v in variants if _same_multiset(g, v)), None)
            if match is not None:
                used.add(p_idx)
                chosen.append(match)
                break
        else:
            return False, f"no predicted column matches gold column {gold.columns[g_idx]!r}"

    # columns match individually; now check they line up row by row
    gold_rows = list(zip(*gold_cols))
    pred_rows = list(zip(*chosen))
    remaining = list(pred_rows)
    for g_row in gold_rows:
        for j, p_row in enumerate(remaining):
            if all(_equal(a, b) for a, b in zip(g_row, p_row)):
                del remaining[j]
                break
        else:
            return False, f"no predicted row matches gold row {g_row}"
    return True, "match"
