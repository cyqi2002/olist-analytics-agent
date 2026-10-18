"""Planner: restates the question, picks the tables and business terms it needs,
and writes down any assumption it has to make when the question is ambiguous."""

from pydantic import BaseModel, Field

from agent.db import catalog
from agent.llm import structured
from agent.schema import describe_glossary, table_summaries

SYSTEM = """You are the planning step of an analytics assistant for Olist, a Brazilian e-commerce marketplace.
You do not write SQL. You decide what the question means and which data is needed to answer it.

- Restate the question precisely, as an analyst would before writing a query: the metric, the
  grouping, the filters, the time range and the number of rows expected.
- Choose every table the query needs, including tables needed only for joins
  (e.g. category names need products and category_translation).
- When the question uses a business term defined in the glossary, list that term and follow its definition.
- If the question is ambiguous (unclear metric, time range, or definition), pick the most reasonable
  interpretation, record it in assumptions, and put the question you would ask the user in
  clarifying_question. Leave clarifying_question empty when the question is clear."""


class Plan(BaseModel):
    restated_question: str = Field(description="Precise version of the question")
    tables: list[str] = Field(description="Tables needed, including join tables")
    glossary_terms: list[str] = Field(description="Glossary terms the question relies on")
    assumptions: list[str] = Field(description="Interpretations chosen for ambiguous parts")
    clarifying_question: str = Field(description="Question to ask the user if ambiguous, else empty string")


def plan(question: str) -> tuple[Plan, dict]:
    prompt = (
        f"Tables:\n{table_summaries()}\n\n"
        f"Business glossary:\n{describe_glossary()}\n\n"
        f"Question: {question}"
    )
    result, usage = structured(SYSTEM, prompt, Plan, step="planner")
    # keep only real tables; an empty selection falls back to the full schema
    valid = [t for t in result.tables if t in catalog()]
    result.tables = valid or list(catalog())
    return result, usage
