"""Reporter: picks a chart from the shape of the result, builds it in Plotly, and
writes a 3-sentence insight that may only use numbers present in the result.

The number rule is enforced in code (check_numbers), not by asking another LLM:
every number in the insight must match a value in the result after rounding.
If the check fails, the insight is regenerated once with the offending numbers listed.
"""

import re

import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from agent.llm import structured
from agent.planner import Plan

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
MUTED_TEXT = "#52514e"
GRID = "#e6e5e0"
MAX_LINES = 3
MAX_BARS = 25


# ---------- chart choice ----------

def _is_time(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object:
        sample = s.dropna().astype(str).head(20)
        return len(sample) > 0 and sample.str.match(r"^\d{4}-\d{2}(-\d{2})?").all()
    return False


def choose_chart(df: pd.DataFrame) -> str:
    """metric | line | bar | table, from the shape of the result."""
    if df is None or df.empty:
        return "table"
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    time_cols = [c for c in df.columns if _is_time(df[c])]
    if len(df) == 1 and len(numeric) >= 1 and len(df.columns) <= 2:
        return "metric"
    if time_cols and numeric and len(numeric) <= MAX_LINES and len(df) >= 3:
        return "line"
    labels = [c for c in df.columns if c not in numeric]
    if len(numeric) == 1 and len(labels) == 1 and len(df) <= MAX_BARS:
        return "bar"
    if len(numeric) == 1 and not labels and len(df) <= MAX_BARS:
        return "bar"
    return "table"


def _layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15}},
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        font={"color": MUTED_TEXT},
        hovermode="closest",
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def build_chart(df: pd.DataFrame, kind: str, title: str) -> go.Figure | None:
    if kind in ("metric", "table") or df is None or df.empty:
        return None
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]

    if kind == "line":
        x = next(c for c in df.columns if _is_time(df[c]))
        data = df.sort_values(x)
        fig = go.Figure()
        for i, col in enumerate(c for c in numeric if c != x):
            fig.add_trace(go.Scatter(
                x=data[x], y=data[col], name=col, mode="lines+markers",
                line={"width": 2, "color": SERIES[i]}, marker={"size": 8},
            ))
        fig.update_layout(showlegend=len(numeric) > 1)
        return _layout(fig, title)

    # bar: horizontal, largest at the top, so long category names stay readable
    y_col = numeric[0]
    labels = [c for c in df.columns if c not in numeric]
    label_col = labels[0] if labels else None
    data = df.sort_values(y_col, ascending=True)
    names = data[label_col].astype(str) if label_col else data.index.astype(str)
    fig = go.Figure(go.Bar(
        x=data[y_col], y=names, orientation="h",
        marker={"color": SERIES[0], "cornerradius": 4},
        hovertemplate="%{y}: %{x:,.2f}<extra></extra>",
    ))
    fig.update_layout(height=max(300, 32 * len(data) + 80), showlegend=False)
    fig.update_xaxes(title=y_col)
    return _layout(fig, title)


# ---------- number check ----------

SUFFIX = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}
NUMBER_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(k|thousand|million|mn|m|bn|billion)?\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
MONTHS = "january february march april may june july august september october november december".split()


def _allowed_values(df: pd.DataFrame, question: str) -> tuple[list[float], set[int]]:
    values: list[float] = []
    date_parts: set[int] = set()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            values.extend(float(v) for v in s.dropna())
        elif pd.api.types.is_datetime64_any_dtype(s):
            for ts in s.dropna():
                date_parts.update({ts.year, ts.month, ts.day})
        else:
            for v in s.dropna().astype(str):
                date_parts.update(int(n) for n in re.findall(r"\d+", v) if len(n) <= 4)
                try:
                    values.append(float(v))  # numbers stored as text (e.g. Decimal)
                except ValueError:
                    pass
    # numbers the user wrote, and small counts/ranks ("3 of the 5 states")
    question_numbers = {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", question)}
    small = set(range(0, len(df) + 1))
    return values, date_parts | question_numbers | small


def check_numbers(text: str, df: pd.DataFrame, question: str = "") -> list[str]:
    """Return the numbers in `text` that don't match any value in `df`."""
    values, exact_ok = _allowed_values(df, question)
    text = DATE_RE.sub(" ", text)
    bad = []
    for m in NUMBER_RE.finditer(text):
        whole, frac, suffix = m.group(1), m.group(2) or "", (m.group(3) or "").lower()
        x = float(whole.replace(",", "") + frac)
        decimals = len(frac) - 1 if frac else 0
        mult = SUFFIX.get(suffix, 1.0)
        target = x * mult
        tol = 0.5 * 10 ** (-decimals) * mult + 1e-9
        if not suffix and not frac and x == int(x) and int(x) in exact_ok:
            continue
        candidates = values + [v * 100 for v in values]  # ratio written as a percentage
        if any(abs(v - target) <= tol for v in candidates):
            continue
        # also accept truncation-free rounding to whole numbers of large values (e.g. 1,255,695)
        if not frac and any(abs(round(v) - target) < 1 for v in candidates):
            continue
        bad.append(m.group(0).strip())
    return bad


# ---------- insight ----------

SYSTEM = """You are a retail analyst writing for an operations manager.
Write exactly 3 sentences about the query result: the headline answer, the most notable
pattern or comparison, and one practical implication.

Strict rule: every number you write must appear in the result table (you may round it,
format it with thousands separators, or show a ratio as a percentage). Do not calculate
new numbers such as differences, ratios or totals that are not in the table.
Money is in Brazilian reais (R$)."""


class Insight(BaseModel):
    insight: str = Field(description="Exactly 3 sentences")


def _result_as_text(df: pd.DataFrame, max_rows: int = 50) -> str:
    shown = df.head(max_rows)
    text = shown.to_csv(index=False)
    if len(df) > max_rows:
        text += f"... ({len(df)} rows in total, first {max_rows} shown)\n"
    return text


def write_insight(question: str, plan: Plan | None, df: pd.DataFrame) -> tuple[str, list[str], list[dict]]:
    base = f"Question: {question}\n"
    if plan and plan.assumptions:
        base += "Assumptions made: " + "; ".join(plan.assumptions) + "\n"
    base += f"\nResult:\n{_result_as_text(df)}"
    usages = []
    prompt = base
    for _ in range(2):
        out, usage = structured(SYSTEM, prompt, Insight, step="reporter", effort="low", max_tokens=4000)
        usages.append(usage)
        bad = check_numbers(out.insight, df, question)
        if not bad:
            return out.insight, [], usages
        prompt = (base + f"\n\nYour previous draft used numbers that are not in the result: {', '.join(bad)}. "
                  "Rewrite it using only numbers from the result.")
    return out.insight, bad, usages


def report(question: str, plan: Plan | None, df: pd.DataFrame) -> dict:
    kind = choose_chart(df)
    title = plan.restated_question if plan else question
    fig = build_chart(df, kind, title)
    if df is None or df.empty:
        return {"chart": None, "chart_type": "table", "insight": "The query returned no rows.",
                "insight_check": {"passed": True, "unsupported_numbers": []}}
    insight, bad, usages = write_insight(question, plan, df)
    return {
        "chart": fig,
        "chart_type": kind,
        "insight": insight,
        "insight_check": {"passed": not bad, "unsupported_numbers": bad},
        "usage": usages,
    }
