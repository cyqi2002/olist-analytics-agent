"""Phase 1 baseline: a single LLM call that sees the schema and returns SQL.
No planning, no validation, no retries. This is the number to beat.

with_context=False gives the model only table/column names and types.
with_context=True adds the hand-written schema notes and the business glossary,
which measures how much "context engineering" alone is worth."""

from pydantic import BaseModel, Field

from agent.llm import structured
from agent.schema import describe_glossary, describe_tables

SYSTEM = """You write DuckDB SQL for an e-commerce analytics database.
Return one SELECT query that answers the user's question."""


class SQLAnswer(BaseModel):
    sql: str = Field(description="A single DuckDB SELECT statement")


def generate_sql(question: str, with_context: bool = False) -> tuple[str, dict]:
    prompt = f"Database schema:\n\n{describe_tables(with_docs=with_context)}\n\n"
    if with_context:
        prompt += f"Business definitions:\n{describe_glossary()}\n\n"
    prompt += f"Question: {question}"
    answer, usage = structured(SYSTEM, prompt, SQLAnswer, step="baseline")
    return answer.sql, usage
