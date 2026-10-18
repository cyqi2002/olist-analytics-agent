"""Exercise the graph's retry loop with the LLM steps stubbed out (no API calls)."""

import agent.graph as graph
from agent.planner import Plan
from agent.sql_writer import SQLDraft

USAGE = {"step": "stub", "cost_usd": 0.0}
PLAN = Plan(restated_question="q", tables=["orders"], glossary_terms=[], assumptions=[], clarifying_question="")


def stub_writer(drafts):
    calls = []

    def write_sql(question, plan, previous_sql=None, errors=None):
        calls.append({"previous_sql": previous_sql, "errors": errors})
        return SQLDraft(sql=drafts[len(calls) - 1], explanation=""), USAGE

    return write_sql, calls


def test_retry_fixes_bad_column(monkeypatch):
    write_sql, calls = stub_writer([
        "SELECT count(*) FROM orders WHERE status = 'canceled'",       # wrong column
        "SELECT count(*) AS n FROM orders WHERE order_status = 'canceled'",
    ])
    monkeypatch.setattr(graph, "plan", lambda q: (PLAN, USAGE))
    monkeypatch.setattr(graph, "write_sql", write_sql)

    state = graph.run("how many canceled orders?", report=False)

    assert state["error"] is None
    assert state["result"].iloc[0, 0] == 625
    assert state["attempts"] == 2
    assert "status" in calls[1]["errors"][0]  # the validator error was fed back
    assert state["error_history"][0]["stage"] == "validator"


def test_gives_up_after_max_retries(monkeypatch):
    write_sql, calls = stub_writer(["DROP TABLE orders"] * 5)
    monkeypatch.setattr(graph, "plan", lambda q: (PLAN, USAGE))
    monkeypatch.setattr(graph, "write_sql", write_sql)

    state = graph.run("delete everything", max_retries=2, report=False)

    assert len(calls) == 3  # first attempt + 2 retries
    assert state["error"] is not None
    assert state.get("result") is None


def test_empty_result_triggers_retry(monkeypatch):
    write_sql, calls = stub_writer([
        "SELECT count(*) AS n FROM orders WHERE order_status = 'Canceled' GROUP BY order_status",
        "SELECT count(*) AS n FROM orders WHERE order_status = 'canceled' GROUP BY order_status",
    ])
    monkeypatch.setattr(graph, "plan", lambda q: (PLAN, USAGE))
    monkeypatch.setattr(graph, "write_sql", write_sql)

    state = graph.run("how many canceled orders?", report=False)

    assert state["attempts"] == 2
    assert state["error_history"][0]["stage"] == "empty_result"
    assert state["result"].iloc[0, 0] == 625
