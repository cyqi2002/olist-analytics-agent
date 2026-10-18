"""Render schema descriptions and the business glossary as prompt text."""

from functools import lru_cache
from pathlib import Path

import yaml

from agent.db import catalog

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def schema_docs() -> dict:
    return yaml.safe_load((HERE / "schema_docs.yaml").read_text())


@lru_cache(maxsize=1)
def glossary() -> dict:
    return yaml.safe_load((HERE / "glossary.yaml").read_text())


def describe_table(table: str, with_docs: bool = True) -> str:
    if not with_docs:
        cols = ", ".join(f"{c} {t}" for c, t in catalog()[table].items())
        return f"TABLE {table} ({cols})"
    doc = schema_docs().get(table, {})
    col_notes = doc.get("columns", {})
    lines = [f"TABLE {table}  -- {doc.get('description', '')}"]
    if doc.get("grain"):
        lines.append(f"  grain: {doc['grain']}")
    for col, dtype in catalog()[table].items():
        note = f"  -- {col_notes[col]}" if col in col_notes else ""
        lines.append(f"  {col} {dtype}{note}")
    for j in doc.get("joins", []):
        lines.append(f"  join: {j}")
    for n in doc.get("notes", []):
        lines.append(f"  NOTE: {n}")
    return "\n".join(lines)


def describe_tables(tables: list[str] | None = None, with_docs: bool = True) -> str:
    tables = tables or list(catalog())
    return "\n\n".join(describe_table(t, with_docs) for t in tables if t in catalog())


def table_summaries() -> str:
    """One line per table - enough for the planner to choose tables."""
    docs = schema_docs()
    return "\n".join(
        f"- {t}: {docs.get(t, {}).get('description', '')} Columns: {', '.join(cols)}"
        for t, cols in catalog().items()
    )


def describe_glossary(terms: list[str] | None = None) -> str:
    g = glossary()
    terms = terms if terms is not None else list(g)
    lines = []
    for term in terms:
        if term not in g:
            continue
        entry = g[term]
        aliases = f" (also: {', '.join(entry['aliases'])})" if entry.get("aliases") else ""
        lines.append(f"- {term}{aliases}: {entry['definition']}")
    return "\n".join(lines)
