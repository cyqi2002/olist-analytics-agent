import pytest

from agent.db import QueryTimeout, run_query
from agent.validator import validate


def test_valid_select_passes_and_gets_limit():
    r = validate("SELECT customer_state, count(*) AS n FROM customers GROUP BY 1")
    assert r.ok, r.errors
    assert r.limit_added
    assert "LIMIT 1000" in r.sql


def test_existing_limit_kept():
    r = validate("SELECT * FROM sellers LIMIT 5")
    assert r.ok and not r.limit_added


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "DROP TABLE orders",
        "INSERT INTO sellers VALUES ('a', 'b', 'c', 'd')",
        "UPDATE orders SET order_status = 'x'",
        "CREATE TABLE t AS SELECT 1",
        "SELECT 1; DROP TABLE orders",
        "ATTACH 'other.db'",
        "COPY orders TO 'out.csv'",
        "SELECT * FROM read_csv('/etc/passwd')",
    ],
)
def test_non_select_rejected(sql):
    assert not validate(sql).ok


def test_unknown_table():
    r = validate("SELECT * FROM order")
    assert not r.ok


def test_unknown_column_reports_available_columns():
    r = validate("SELECT p.product_name_lenght FROM products p")
    assert not r.ok
    assert "product_name_length" in r.errors[0]


def test_unqualified_unknown_column():
    r = validate("SELECT revenue FROM order_items")
    assert not r.ok


def test_aliases_and_ctes_allowed():
    sql = """
    WITH monthly AS (
      SELECT date_trunc('month', order_purchase_timestamp) AS month, count(*) AS n FROM orders GROUP BY 1
    )
    SELECT month, n, n - lag(n) OVER (ORDER BY month) AS diff FROM monthly ORDER BY month
    """
    r = validate(sql)
    assert r.ok, r.errors


def test_explain_catches_binder_errors():
    r = validate("SELECT sum(order_status) FROM orders")
    assert not r.ok


def test_syntax_error():
    assert not validate("SELEC * FRM orders").ok


def test_timeout():
    slow = "SELECT count(*) FROM geolocation a, geolocation b"
    with pytest.raises(QueryTimeout):
        run_query(slow, timeout_s=0.5)
