"""SQL writer: turns the plan into one DuckDB query, seeing only the tables the
planner selected. On a retry it also sees its previous query and the error."""

from pydantic import BaseModel, Field

from agent.llm import structured
from agent.planner import Plan
from agent.schema import describe_glossary, describe_tables

SYSTEM = """You write DuckDB SQL for an e-commerce analytics database.
Return exactly one read-only SELECT query (CTEs are fine) that answers the question.

Rules:
- Use only the tables and columns in the schema provided.
- Follow the business definitions exactly when a term is used, and only then: do not carry a
  definition's filters (e.g. excluding canceled orders) over to metrics it doesn't define.
- Never sum money across a join of two one-to-many tables (e.g. order_items with order_payments);
  aggregate each to order level first.
- Return only the columns needed to answer, with readable aliases. Include the grouping column
  next to each metric.
- Return the number of rows the question asks for (e.g. LIMIT 5 for "top 5").
- For time series, order by the time column. Do not round numbers in SQL."""


class SQLDraft(BaseModel):
    sql: str = Field(description="A single DuckDB SELECT statement")
    explanation: str = Field(description="One sentence on how the query answers the question")


def write_sql(question: str, plan: Plan, previous_sql: str | None = None,
              errors: list[str] | None = None) -> tuple[SQLDraft, dict]:
    parts = [
        f"Schema (only these tables are available):\n\n{describe_tables(plan.tables)}",
        f"Business definitions:\n{describe_glossary(list(dict.fromkeys(plan.glossary_terms + ['status_filters'])))}",
        f"Question: {question}",
        f"Interpreted as: {plan.restated_question}",
    ]
    if plan.assumptions:
        parts.append("Assumptions:\n" + "\n".join(f"- {a}" for a in plan.assumptions))
    if previous_sql:
        parts.append(
            "Your previous query failed. Fix it.\n"
            f"Previous query:\n{previous_sql}\n\nErrors:\n" + "\n".join(f"- {e}" for e in errors or [])
        )
    return structured(SYSTEM, "\n\n".join(parts), SQLDraft, step="sql_writer")
