import pandas as pd

from agent.reporter import build_chart, check_numbers, choose_chart

CATEGORIES = pd.DataFrame({"category": ["health_beauty", "watches_gifts", "bed_bath_table"],
                           "revenue": [1255695.13, 1198185.21, 1035964.06]})


def test_chart_choice():
    assert choose_chart(pd.DataFrame({"orders": [99441]})) == "metric"
    assert choose_chart(CATEGORIES) == "bar"
    monthly = pd.DataFrame({"month": pd.date_range("2018-01-01", periods=8, freq="MS"), "orders": range(8)})
    assert choose_chart(monthly) == "line"
    text_months = pd.DataFrame({"month": ["2018-01", "2018-02", "2018-03"], "orders": [1, 2, 3]})
    assert choose_chart(text_months) == "line"
    wide = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"], "c": [1, 2], "d": [3, 4]})
    assert choose_chart(wide) == "table"


def test_build_chart():
    assert build_chart(CATEGORIES, "bar", "t") is not None
    assert build_chart(CATEGORIES, "table", "t") is None


def test_numbers_from_result_pass():
    text = ("Health & beauty led with R$1,255,695.13 in revenue, ahead of watches & gifts at R$1.2 million. "
            "The top 3 categories each exceeded R$1.0M.")
    assert check_numbers(text, CATEGORIES, "top 3 categories by revenue") == []


def test_invented_numbers_flagged():
    text = "Health & beauty beat watches & gifts by R$57,510, a 4.8% lead."
    bad = check_numbers(text, CATEGORIES)
    assert "57,510" in bad and "4.8" in bad


def test_percent_of_ratio_and_dates():
    df = pd.DataFrame({"month": pd.to_datetime(["2018-01-01", "2018-02-01"]), "late_share": [0.0656, 0.1599]})
    assert check_numbers("Late deliveries rose from 6.6% in January 2018 to 16.0% in 2018-02.", df) == []
    assert check_numbers("Late deliveries rose by 9.4 points.", df) == ["9.4"]
