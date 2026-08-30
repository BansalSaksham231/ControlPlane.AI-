"""
ControlPlane.ai monitoring dashboard (Streamlit).

Runs the real pipeline over synthetic traffic and visualises the actual
``DecisionTrace`` outputs — no simulated numbers.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.metrics import collect_traces, compute_dashboard_metrics

st.set_page_config(page_title="ControlPlane.ai — Monitoring", layout="wide")
st.title("ControlPlane.ai — Monitoring")
st.caption(
    "Live view over synthetic traffic. Heuristic, deterministic detectors — "
    "a system, not a production safety guarantee."
)

with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["evaluation", "traffic"], index=0)
    limit = st.slider("Interactions", min_value=50, max_value=600, value=150, step=50)

with st.spinner("Running the pipeline…"):
    traces, session_manager = collect_traces(limit=limit, source=source)
    metrics = compute_dashboard_metrics(traces, session_manager=session_manager)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Interactions", metrics.total_interactions)
c2.metric("Allowed", f"{metrics.allow_rate:.0%}")
c3.metric("Verify", f"{metrics.verify_rate:.0%}")
c4.metric("Human review", f"{metrics.human_review_rate:.0%}")
c5.metric("Blocked", f"{metrics.block_rate:.0%}")

c1, c2, c3 = st.columns(3)
c1.metric("Mean latency (ms)", f"{metrics.mean_latency_ms:.2f}")
c2.metric("p95 latency (ms)", f"{metrics.p95_latency_ms:.2f}")
c3.metric("Mean detector confidence", f"{metrics.mean_detector_confidence:.2f}")
c1.metric("Total est. cost (₹)", f"{metrics.total_estimated_cost_inr:.2f}")
c2.metric("Mean est. cost (₹)", f"{metrics.mean_estimated_cost_inr:.4f}")
c3.metric("Session escalations", metrics.session_escalations)

st.subheader("Intervention distribution")
st.bar_chart(pd.Series(metrics.decisions, name="count"))

st.subheader("Risk distributions")
risk_df = pd.DataFrame(
    {
        "performance": metrics.performance_risk_histogram,
        "responsibility": metrics.responsibility_risk_histogram,
        "cost": metrics.cost_risk_histogram,
        "overall": metrics.overall_risk_histogram,
    }
)
st.bar_chart(risk_df)

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Performance status")
    st.bar_chart(pd.Series(metrics.performance_status_distribution, name="count"))
    st.subheader("Top risk categories")
    st.dataframe(pd.DataFrame(metrics.top_risk_categories), hide_index=True)
with col_right:
    st.subheader("Triggered policy rules")
    st.dataframe(
        pd.DataFrame(
            metrics.triggered_rule_counts.items(), columns=["rule", "count"]
        ),
        hide_index=True,
    )
    st.subheader("Decisions by application")
    st.dataframe(pd.DataFrame(metrics.by_application).fillna(0).astype(int))

st.subheader("Sample decisions")
sample = pd.DataFrame(
    [
        {
            "interaction_id": t.interaction_id,
            "application": t.application,
            "decision": t.final_decision.decision.value,
            "overall_risk": t.final_decision.overall_risk,
            "perf": t.performance.status.value,
            "explanation": t.final_decision.explanation,
        }
        for t in traces[:40]
    ]
)
st.dataframe(sample, hide_index=True)
