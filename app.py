"""Streamlit UI: ask a question, see the plan, SQL, chart and insight.

    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from agent.graph import run

st.set_page_config(page_title="Olist Analytics Agent", page_icon="📊", layout="wide")

EXAMPLES = [
    "Which 5 product categories bring in the most revenue?",
    "How have sales trended month by month?",
    "Do late deliveries get worse reviews?",
    "What percentage of customers are repeat customers?",
    "Which states have the longest delivery times?",
]

st.title("Olist Analytics Agent")
st.caption("Ask a business question about the Olist marketplace (100k orders, 2016-2018). "
           "The agent plans the query, writes and validates SQL, runs it, and summarises the result.")

with st.sidebar:
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            st.session_state["question"] = ex
    st.divider()
    st.markdown("**Pipeline:** planner → SQL writer → validator → DuckDB → reporter. "
                "Failed queries are sent back to the writer with the error (max 2 retries).")

question = st.text_input("Your question", key="question",
                         placeholder="e.g. Which seller states sold the most items?")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Planning, writing SQL and running it..."):
        state = run(question.strip())
    st.session_state["state"] = state

state = st.session_state.get("state")
if state:
    plan = state.get("plan")
    if plan:
        st.markdown(f"**Interpreted as:** {plan.restated_question}")
        if plan.assumptions:
            st.markdown("**Assumptions:** " + " · ".join(plan.assumptions))
        if plan.clarifying_question:
            st.info(f"This question is open to interpretation. The agent went ahead with the assumptions "
                    f"above, but you may want to clarify: *{plan.clarifying_question}*")

    result = state.get("result")
    if state.get("error") and (result is None or result.empty):
        st.error(f"Couldn't answer this one after {state.get('attempts', 0)} attempt(s): {state['error']}")
    elif result is not None:
        kind = state.get("chart_type")
        if kind == "metric":
            numeric = result.select_dtypes("number").columns
            col = numeric[0] if len(numeric) else result.columns[-1]
            value = result.iloc[0][col]
            st.metric(label=col.replace("_", " "), value=f"{value:,.2f}" if isinstance(value, float) else value)
        elif state.get("chart") is not None:
            st.plotly_chart(state["chart"], width="stretch")
        else:
            st.dataframe(result, width="stretch", hide_index=True)

        if state.get("insight"):
            st.subheader("Insight")
            st.write(state["insight"])
            check = state.get("insight_check", {})
            if check and not check.get("passed", True):
                st.warning("Some numbers in this summary could not be matched to the result: "
                           + ", ".join(check["unsupported_numbers"]))

        with st.expander(f"Data ({len(result)} rows)"):
            st.dataframe(result, width="stretch", hide_index=True)
            st.download_button("Download CSV", result.to_csv(index=False), "result.csv", "text/csv")

    with st.expander("SQL"):
        st.code(state.get("sql") or "", language="sql")
        if state.get("explanation"):
            st.caption(state["explanation"])
        history = state.get("error_history", [])
        if history:
            st.markdown(f"**Caught and fixed {len(history)} problem(s) before answering:**")
            st.dataframe(pd.DataFrame(history)[["attempt", "stage", "error"]], hide_index=True)

    usage = state.get("usage", [])
    if usage:
        cost = sum(u["cost_usd"] for u in usage)
        latency = sum(u["latency_s"] for u in usage)
        st.caption(f"{len(usage)} LLM calls · ${cost:.3f} · {latency:.1f}s model time · "
                   f"{state.get('attempts', 1)} SQL attempt(s)")
