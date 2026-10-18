from decimal import Decimal

import pandas as pd

from evals.compare import results_match


def test_ignores_column_names_and_row_order():
    gold = pd.DataFrame({"state": ["SP", "RJ"], "orders": [10, 5]})
    pred = pd.DataFrame({"n": [5, 10], "customer_state": ["RJ", "SP"]})
    assert results_match(pred, gold)[0]


def test_extra_predicted_columns_allowed():
    gold = pd.DataFrame({"payment_installments": [1]})
    pred = pd.DataFrame({"installments": [1], "payments": [25455]})
    assert results_match(pred, gold)[0]


def test_row_count_must_match():
    gold = pd.DataFrame({"x": [1, 2]})
    pred = pd.DataFrame({"x": [1, 2, 3]})
    assert not results_match(pred, gold)[0]


def test_rounding_and_decimal_tolerated():
    gold = pd.DataFrame({"avg": [4.086421]})
    assert results_match(pd.DataFrame({"a": [4.09]}), gold)[0]
    assert results_match(pd.DataFrame({"a": [Decimal("4.09")]}), gold)[0]
    assert not results_match(pd.DataFrame({"a": [4.2]}), gold)[0]


def test_ratio_vs_percentage():
    gold = pd.DataFrame({"pct": [3.04032]})
    assert results_match(pd.DataFrame({"ratio": [0.0304032]}), gold)[0]


def test_month_formats():
    gold = pd.DataFrame({"month": pd.to_datetime(["2018-01-01", "2018-02-01"]), "n": [7269, 6728]})
    pred = pd.DataFrame({"m": ["2018-01", "2018-02"], "orders": [7269, 6728]})
    assert results_match(pred, gold)[0]


def test_rows_must_line_up():
    gold = pd.DataFrame({"state": ["SP", "RJ"], "orders": [10, 5]})
    pred = pd.DataFrame({"state": ["SP", "RJ"], "orders": [5, 10]})
    assert not results_match(pred, gold)[0]


def test_numeric_only_ignores_labels():
    gold = pd.DataFrame({"is_late": [False, True], "score": [4.29, 2.57]})
    pred = pd.DataFrame({"status": ["on time", "late"], "score": [4.29, 2.57]})
    assert not results_match(pred, gold)[0]
    assert results_match(pred, gold, numeric_only=True)[0]
