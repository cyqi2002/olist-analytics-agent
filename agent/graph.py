"""LangGraph wiring.

    plan -> write_sql -> validate --ok--> execute --ok--> report -> END
                ^           |                 |
                +--- error -+------ error ----+   (at most MAX_RETRIES times)

Each node reads the shared state and returns only the keys it changes.
`usage` and `error_history` use an add reducer, so every node's entries are
appended instead of overwritten.
"""

import json
import operator
import time
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agent.config import LOG_DIR, MAX_RETRIES
from agent.db import run_query
from agent.planner import Plan, plan
from agent.sql_writer import write_sql
from agent.validator import validate


class AgentState(TypedDict, total=False):
    question: str
    max_retries: int
    report: bool
    plan: Plan
    sql: str
    explanation: str
    attempts: int
    error: str | None
    error_history: Annotated[list[dict], operator.add]
    result: pd.DataFrame | None
    chart: Any
    chart_type: str
    insight: str
    insight_check: dict
    usage: Annotated[list[dict], operator.add]


def plan_node(state: AgentState) -> dict:
    p, usage = plan(state["question"])
    return {"plan": p, "attempts": 0, "usage": [usage]}


def write_sql_node(state: AgentState) -> dict:
    retrying = state.get("error") is not None
    draft, usage = write_sql(
        state["question"],
        state["plan"],
        previous_sql=state.get("sql") if retrying else None,
        errors=[state["error"]] if retrying else None,
    )
    return {"sql": draft.sql, "explanation": draft.explanation, "attempts": state["attempts"] + 1,
            "error": None, "usage": [usage]}


def validate_node(state: AgentState) -> dict:
    v = validate(state["sql"])
    if not v.ok:
        error = "; ".join(v.errors)
        return {"error": error,
                "error_history": [{"stage": "validator", "attempt": state["attempts"], "sql": state["sql"],
                                   "error": error}]}
    return {"sql": v.sql}


def execute_node(state: AgentState) -> dict:
    try:
        df = run_query(state["sql"])
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        return {"error": error, "result": None,
                "error_history": [{"stage": "execute", "attempt": state["attempts"], "sql": state["sql"],
                                   "error": error}]}
    if df.empty:
        # an empty result is usually a wrong filter value; let the writer try once more
        error = "query returned 0 rows - check filter values (exact spelling and case) and join keys"
        return {"error": error, "result": df,
                "error_history": [{"stage": "empty_result", "attempt": state["attempts"], "sql": state["sql"],
                                   "error": error}]}
    return {"result": df, "error": None}


def report_node(state: AgentState) -> dict:
    from agent.reporter import report

    return report(state["question"], state["plan"], state["result"])


def _retries_left(state: AgentState) -> bool:
    return state["attempts"] <= state.get("max_retries", MAX_RETRIES)


def after_validate(state: AgentState) -> str:
    if state.get("error") is None:
        return "execute"
    return "write_sql" if _retries_left(state) else END


def after_execute(state: AgentState) -> str:
    if state.get("error") is None:
        return "report" if state.get("report", True) else END
    if _retries_left(state):
        return "write_sql"
    # out of retries: an empty result is still a valid (if unhelpful) answer
    return "report" if state.get("result") is not None and state.get("report", True) else END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("write_sql", write_sql_node)
    g.add_node("validate", validate_node)
    g.add_node("execute", execute_node)
    g.add_node("report", report_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "write_sql")
    g.add_edge("write_sql", "validate")
    g.add_conditional_edges("validate", after_validate, ["execute", "write_sql", END])
    g.add_conditional_edges("execute", after_execute, ["report", "write_sql", END])
    g.add_edge("report", END)
    return g.compile()


GRAPH = build_graph()


def run(question: str, max_retries: int | None = None, report: bool = True) -> AgentState:
    start = time.perf_counter()
    state = GRAPH.invoke({
        "question": question,
        "max_retries": MAX_RETRIES if max_retries is None else max_retries,
        "report": report,
    })
    _log(state, time.perf_counter() - start)
    return state


def _log(state: AgentState, seconds: float) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": state["question"],
        "plan": state["plan"].model_dump() if state.get("plan") else None,
        "sql": state.get("sql"),
        "attempts": state.get("attempts"),
        "error": state.get("error"),
        "error_history": state.get("error_history", []),
        "rows": None if state.get("result") is None else len(state["result"]),
        "insight": state.get("insight"),
        "insight_check": state.get("insight_check"),
        "usage": state.get("usage", []),
        "cost_usd": round(sum(u["cost_usd"] for u in state.get("usage", [])), 5),
        "latency_s": round(seconds, 2),
    }
    with open(LOG_DIR / "runs.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
