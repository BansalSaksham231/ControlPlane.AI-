"""
ControlPlane.ai — interactive web UI.

    streamlit run streamlit_app.py

Runs the real decision pipeline in-process (no separate API server
needed). Every number shown comes from the actual engines.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from api.service import VERSION, ControlPlaneService
from data.schemas import (
    ActionType,
    Application,
    Interaction,
    InterventionTier,
    ModelName,
    UserType,
)
from decision.replay import build_replay
from explainability.builder import build_explanation
from monitoring.schemas import OperationalMonitoringReport
from tests import scenarios

st.set_page_config(
    page_title="ControlPlane.ai", page_icon="🛡️", layout="wide"
)

_TIER_STYLE = {
    "ALLOW": ("#1a7f37", "✅", "Response released as-is."),
    "ANNOTATE": ("#0969da", "📝", "Released with a caveat attached."),
    "VERIFY": ("#9a6700", "🔎", "Automated verification required before release."),
    "HUMAN_REVIEW": ("#bc4c00", "🧑‍⚖️", "Routed to a human reviewer."),
    "BLOCK": ("#cf222e", "⛔", "Response withheld."),
}

_SCENARIO_LABELS = {
    "A_clean": "A · Clean",
    "B_hallucination": "B · Hallucination",
    "C_pii": "C · PII leakage",
    "D_high_consequence": "D · High consequence",
    "E_multi_risk": "E · Multi-risk",
    "F_cost_anomaly": "F · Cost anomaly",
    "H_low_confidence": "H · Low confidence",
}


@st.cache_resource(show_spinner="Starting ControlPlane pipeline…")
def get_service() -> ControlPlaneService:
    return ControlPlaneService(fit_cost_baseline=True)


# ------------------------------------------------------------------ helpers


def interaction_to_form(interaction: Interaction) -> dict:
    return {
        "application": interaction.application.value,
        "user_type": interaction.user_type.value,
        "model": interaction.model.value,
        "session_id": interaction.session_id,
        "prompt": interaction.prompt,
        "context": interaction.context,
        "response": interaction.response,
        "tokens_in": interaction.tokens_in,
        "tokens_out": interaction.tokens_out,
        "latency_ms": float(interaction.latency_ms),
        "tool_calls": interaction.tool_calls,
        "retry_count": interaction.retry_count,
        "action_type": interaction.action_type.value,
        "action_amount_inr": float(interaction.action_amount_inr),
        "affected_entities": interaction.affected_entities,
    }


DEFAULT_FORM = interaction_to_form(scenarios.scenario_a_clean())


def build_interaction(form: dict, interaction_id: str | None = None) -> Interaction:
    return Interaction(
        interaction_id=interaction_id or f"INT-UI-{datetime.now(timezone.utc).strftime('%H%M%S%f')}",
        timestamp=datetime.now(timezone.utc),
        application=Application(form["application"]),
        user_type=UserType(form["user_type"]),
        model=ModelName(form["model"]),
        session_id=form["session_id"] or "SESSION-UI",
        prompt=form["prompt"],
        context=form["context"],
        response=form["response"],
        tokens_in=int(form["tokens_in"]),
        tokens_out=int(form["tokens_out"]),
        latency_ms=max(1.0, float(form["latency_ms"])),
        tool_calls=int(form["tool_calls"]),
        retry_count=int(form["retry_count"]),
        action_type=ActionType(form["action_type"]),
        action_amount_inr=float(form["action_amount_inr"]),
        affected_entities=max(1, int(form["affected_entities"])),
    )


def decision_banner(tier: str) -> None:
    color, icon, blurb = _TIER_STYLE[tier]
    st.markdown(
        f"""
        <div style="border-left:8px solid {color};background:{color}14;
        padding:16px 20px;border-radius:8px;margin:4px 0 12px 0;">
          <div style="font-size:13px;letter-spacing:.14em;color:{color};
          font-weight:700;">FINAL DECISION</div>
          <div style="font-size:34px;font-weight:800;color:{color};">{icon}&nbsp;{tier}</div>
          <div style="color:#57606a;">{blurb}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chips(items, color="#0969da") -> str:
    return " ".join(
        f"<span style='background:{color}18;color:{color};border-radius:6px;"
        f"padding:2px 8px;margin:2px;display:inline-block;font-size:12px;'>{x}</span>"
        for x in items
    )


def _fmt_opt(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_session_memory(mem) -> None:
    """
    'Multi-Turn Session Memory' — the ContextualSnapshot that fed this
    decision. Rendered only for genuinely multi-turn / critical sessions;
    a pure copy of ``summary.session_memory`` (nothing recomputed).
    """
    if mem is None or not (mem.turns_recorded > 1 or mem.has_critical_history):
        return

    st.markdown("### 🧵 Multi-Turn Session Memory")
    st.caption(
        "Verified state carried into this turn. The heavy detectors ran only on "
        "the newest turn (the delta); these are the accumulated prior-turn signals."
    )

    if mem.critical_floor_applied:
        st.warning(
            f"⚠️ **Non-decaying critical floor {mem.critical_floor:.2f} applied.** "
            "A critical violation earlier in this session forced elevated scrutiny "
            "on the current interaction — this floor is never decayed away."
        )
    elif mem.has_critical_history:
        st.info(
            f"This session has a critical history (non-decaying floor "
            f"{mem.critical_floor:.2f}) — not high enough to raise the current "
            "turn's scrutiny."
        )

    m = st.columns(4)
    m[0].metric("Turns in memory", mem.turns_recorded)
    m[1].metric("Peak performance", f"{mem.peak_performance_risk * 100:.0f}%")
    m[2].metric("Peak responsibility", f"{mem.peak_responsibility_risk * 100:.0f}%")
    m[3].metric("Peak cost", f"{mem.peak_cost_risk * 100:.0f}%")

    if mem.critical_events:
        st.markdown("**Critical events timeline**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "turn": e.turn_index,
                        "trigger": e.trigger,
                        "decision": e.decision,
                        "risk at event": round(e.risk_at_event, 2),
                    }
                    for e in mem.critical_events
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    if mem.pii_entity_keys:
        st.markdown("**Accumulated PII entities (redacted)**")
        st.markdown(_chips(mem.pii_entity_keys, "#cf222e"), unsafe_allow_html=True)

    if mem.reason_code_counts:
        top = sorted(mem.reason_code_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        st.caption(
            "Recurring reason codes: "
            + ", ".join(f"`{code}` ×{n}" for code, n in top)
        )

    if mem.explanation:
        st.caption(mem.explanation)


# ---- Command Center formatting helpers --------------------------------

_TREND_ARROW = {
    "increasing": "↑", "decreasing": "↓", "stable": "→",
    "up": "↑", "down": "↓", "flat": "→",
}


def _pctv(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _numv(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


_TIER_ARROW = "  →  "


def render_explanation(summary) -> None:
    """
    "Why did ControlPlane decide this?" — rendered entirely from the
    ExplainabilitySummary produced by ``build_explanation(trace)``. Every
    value here is a copy from that summary; nothing is recomputed.
    """
    st.markdown("## Why did ControlPlane decide this?")

    # --- Section 1: executive result -------------------------------------
    e = st.columns(5)
    e[0].metric("Decision", summary.decision.value)
    e[1].metric("Overall risk", f"{summary.overall_risk * 100:.0f}%")
    e[2].metric("Confidence", f"{summary.decision_confidence * 100:.0f}%")
    e[3].metric("Verification", summary.verification_path.value)
    e[4].metric("Human review", "YES" if summary.human_review_required else "NO")

    # --- Section 2: why -------------------------------------------------
    st.markdown("### Why")
    if summary.primary_reasons:
        st.markdown(_chips(summary.primary_reasons, "#bc4c00"), unsafe_allow_html=True)
    else:
        st.caption("No escalating reason codes — the risk band alone set the tier.")
    if summary.decision_drivers:
        st.markdown(
            "**Decision drivers:** "
            + ", ".join(f"`{r}`" for r in summary.decision_drivers)
        )

    # --- Section 3: risk breakdown ------------------------------------
    st.markdown("### Risk dimensions")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "dimension": d.dimension,
                    "risk": round(d.risk, 4),
                    "confidence": round(d.confidence, 4),
                    "weight": round(d.weight, 4),
                    "weighted contribution": round(d.weighted_contribution, 4),
                    "status": d.status or "—",
                    "dominant": "★" if d.is_dominant else "",
                }
                for d in summary.risk_dimensions
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Weighted contributions are copied from the fusion engine — not "
        "recomputed in the UI."
    )

    # --- Section 4: verification ------------------------------------
    v = summary.verification_summary
    st.markdown("### Verification")
    if v.used_deep:
        forced = " (forced despite a clean-looking response)" if v.deep_was_forced else ""
        st.markdown(f"**DEEP verification** ran{forced}.")
        if v.deep_trigger_reasons:
            st.markdown(
                _chips(v.deep_trigger_reasons, "#0969da"), unsafe_allow_html=True
            )
        if v.reason_for_deep_verification:
            st.caption(v.reason_for_deep_verification)
        vc = st.columns(5)
        vc[0].metric("Preliminary risk", _fmt_opt(v.preliminary_risk))
        vc[1].metric("Final risk", _fmt_opt(v.final_risk))
        vc[2].metric("Preliminary conf.", _fmt_opt(v.preliminary_confidence))
        vc[3].metric("Final conf.", _fmt_opt(v.final_confidence))
        vc[4].metric("Disagreement", _fmt_opt(v.disagreement_score))
    else:
        st.success("✓ FAST verification — no deep-verification trigger fired.")
    if v.explanation:
        st.caption(v.explanation)

    # --- Section 5: evidence & claims -------------------------------
    with st.expander(f"Evidence & Claims ({len(summary.evidence)})"):
        if summary.evidence:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "claim": c.claim,
                            "status": c.status,
                            "retrieval similarity": c.retrieval_similarity,
                            "NLI": c.nli_label or "—",
                            "NLI confidence": c.nli_confidence,
                            "evidence strength": c.evidence_strength,
                            "claim risk": c.claim_risk,
                            "top evidence (redacted)": c.supporting_evidence or "—",
                        }
                        for c in summary.evidence
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Claim and evidence text is redacted by the responsibility "
                "detector before it reaches this view."
            )
        else:
            st.caption("No verifiable factual claims were extracted.")

    # --- Section 6: consequence vs criticality --------------------
    st.markdown("### Probability of error  ≠  Consequence if wrong")
    cq, ck = st.columns(2)
    with cq:
        q = summary.consequence_summary
        st.markdown(f"**CONSEQUENCE — {q.consequence_score:.2f} ({q.severity_band})**")
        st.caption("How bad the outcome is *if* the AI is wrong.")
        if q.dominant_factors:
            st.markdown("Dominant: " + ", ".join(f"`{x}`" for x in q.dominant_factors))
        if q.factors:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "factor": f.factor,
                            "value": f.value,
                            "weight": f.weight,
                            "weighted": f.weighted_contribution,
                        }
                        for f in q.factors
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
        if q.explanation:
            st.caption(q.explanation)
    with ck:
        k = summary.criticality_summary
        st.markdown(
            f"**CRITICALITY — {k.action_criticality:.2f} ({k.band})**"
        )
        st.caption("How much it matters that this response is right.")
        if k.dominant_factors:
            st.markdown("Dominant: " + ", ".join(f"`{x}`" for x in k.dominant_factors))
        ck.metric("Max claim criticality", f"{k.max_claim_criticality:.2f}")
        if k.explanation:
            st.caption(k.explanation)

    # --- Section 7: decision path -----------------------------------
    st.markdown("### Decision path")
    if summary.decision_path:
        chain = [summary.decision_path[0].from_tier.value] + [
            s.to_tier.value for s in summary.decision_path
        ]
        st.markdown("#### " + _TIER_ARROW.join(chain))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "rule": s.rule,
                        "from": s.from_tier.value,
                        "to": s.to_tier.value,
                        "changed tier": s.from_tier != s.to_tier,
                        "reason": s.reason,
                    }
                    for s in summary.decision_path
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No policy rule trace available for this interaction.")
    with st.expander(f"All policy rules evaluated ({len(summary.policy_rules)})"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "rule": p.rule,
                        "fired": p.fired,
                        "tier before": p.tier_before.value if p.tier_before else "—",
                        "tier after": p.tier_after.value if p.tier_after else "—",
                        "changed tier": p.changed_tier,
                        "effect": p.effect,
                    }
                    for p in summary.policy_rules
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    # --- Section 7b: multi-turn session memory --------------------
    render_session_memory(summary.session_memory)

    # --- Section 8: human review ----------------------------------
    hr = summary.human_review
    if hr.required:
        st.error("🧑‍⚖️  HUMAN REVIEW REQUIRED")
        if hr.triggering_conditions:
            st.markdown(
                "**Triggered by:** "
                + ", ".join(f"`{r}`" for r in hr.triggering_conditions)
            )
    else:
        st.success("No human review required.")

    # --- Section 9: overall explanation --------------------------
    st.markdown("### ControlPlane explanation")
    st.info(summary.explanation)


def render_trace(trace) -> None:
    fd = trace.final_decision
    decision_banner(fd.decision.value)

    # Risk and confidence are shown side by side — they are different questions.
    left, right = st.columns(2)
    with left:
        st.markdown("###### RISK — how dangerous does this look?")
        st.markdown(f"<div style='font-size:30px;font-weight:800'>{fd.overall_risk:.2f}</div>",
                    unsafe_allow_html=True)
    with right:
        st.markdown("###### CONFIDENCE — how sure are we?")
        cc = fd.decision_confidence
        col = "#1a7f37" if cc >= 0.65 else "#bc4c00" if cc < 0.45 else "#9a6700"
        st.markdown(f"<div style='font-size:30px;font-weight:800;color:{col}'>{cc:.2f}</div>",
                    unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Performance", f"{fd.performance_risk:.2f}",
              help=f"criticality-weighted from {trace.performance.performance_risk:.2f}")
    c2.metric("Responsibility", f"{fd.responsibility_risk:.2f}")
    c3.metric("Cost", f"{fd.cost_risk:.2f}",
              help=f"efficiency {trace.cost.cost_efficiency_score:.2f}")
    c4.metric("Consequence", f"{trace.consequence.consequence_score:.2f}")
    c5.metric("Criticality", f"{trace.criticality.action_criticality:.2f}",
              help=f"{trace.criticality.band}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Human review", "YES" if fd.decision.value in ("HUMAN_REVIEW", "BLOCK") else "no")
    c2.metric("Uncertainty", f"{1 - fd.decision_confidence:.2f}")
    c3.metric("Pipeline latency", f"{trace.latency_ms:.1f} ms",
              help="detectors are independent and can run in parallel in production")

    # "Why did ControlPlane decide this?" — sourced entirely from the
    # ExplainabilitySummary (build_explanation is a pure view of the trace).
    summary = build_explanation(trace)
    st.session_state["result_explanation"] = summary
    render_explanation(summary)

    # The detector deep-dive shows raw internals; all of its free text /
    # claim / evidence strings are taken from the redacted incident replay
    # so no raw PII span is ever rendered.
    replay = build_replay(trace)

    st.markdown("#### Detector deep-dive")
    tabs = st.tabs(
        ["Performance", "Responsibility", "Cost", "Consequence", "Criticality", "Policy trace", "Fusion"]
    )

    with tabs[0]:
        p = trace.performance
        st.write(
            f"**Status:** `{p.status.value}` · grounding "
            f"{p.grounding_score if p.grounding_score is not None else 'n/a'} · "
            f"verification confidence {p.verification_confidence:.2f} · "
            f"method `{p.method}`"
        )
        st.caption(replay.risk_signals.performance_explanation)
        st.caption(
            f"evidence quality {p.evidence_quality:.2f} · detector confidence "
            f"{p.confidence:.2f} · uncertainty {p.uncertainty:.2f}"
        )
        if replay.claims:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "claim (redacted)": c.claim,
                            "status": c.status,
                            "claim risk": c.claim_risk,
                            "evidence strength": c.evidence_strength,
                            "retrieval similarity": c.retrieval_similarity,
                            "nli": c.nli_label or "—",
                            "top evidence (redacted)": c.top_evidence or "—",
                        }
                        for c in replay.claims
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No verifiable factual claims extracted.")

    with tabs[1]:
        r = trace.responsibility
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("PII risk", f"{r.pii_risk:.2f}")
        cc2.metric("Toxicity risk", f"{r.toxicity_risk:.2f}")
        cc3.metric("Bias signal", f"{r.bias_risk:.2f}")
        if r.contains_critical_pii:
            st.error("Critical PII: " + ", ".join(r.critical_pii_types))
        if r.findings:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "category": f.category.value,
                            "subtype": f.subtype,
                            "severity": f.severity.value,
                            "confidence": f.confidence,
                            "evidence (redacted)": f.redacted_text,
                        }
                        for f in r.findings
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.markdown("**Redacted response**")
            st.code(r.redacted_response, language=None)
        else:
            st.caption("No PII / toxicity / bias findings.")

    with tabs[2]:
        c = trace.cost
        b = c.cost_breakdown
        st.write(
            f"**Estimated cost:** ₹{c.estimated_cost_inr:.4f} "
            f"(input ₹{b.input_cost_inr:.4f} · output ₹{b.output_cost_inr:.4f} · "
            f"tools ₹{b.tool_cost_inr:.4f} · retries ₹{b.retry_cost_inr:.4f}) · "
            f"baseline `{c.baseline_source}`"
        )
        st.caption(replay.risk_signals.cost_explanation)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "dimension": i.dimension,
                        "observed": i.observed,
                        "baseline": i.baseline,
                        "ratio": i.ratio,
                        "flags at": i.threshold,
                        "triggered": i.triggered,
                    }
                    for i in c.anomaly_indicators
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    with tabs[3]:
        q = trace.consequence
        st.write(f"**Consequence score:** {q.consequence_score:.2f} ({q.severity_band})")
        st.caption(replay.consequence.explanation)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "factor": c.factor,
                        "value": c.value,
                        "weight": c.weight,
                        "weighted": c.weighted_contribution,
                    }
                    for c in q.contributions
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    with tabs[4]:
        q = trace.criticality
        st.write(
            f"**Action criticality:** {q.action_criticality:.2f} ({q.band}) — "
            "how much it matters if this response is wrong."
        )
        st.caption(replay.criticality.explanation)
        st.dataframe(
            pd.DataFrame(
                [
                    {"factor": f.factor, "value": f.value, "weight": f.weight,
                     "weighted": f.weighted_contribution, "band": f.band}
                    for f in q.factors
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        if replay.criticality.claim_criticalities:
            st.markdown("**Per-claim criticality**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"claim (redacted)": c.claim, "criticality": c.criticality,
                         "signals": ", ".join(c.signals)}
                        for c in replay.criticality.claim_criticalities
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

    with tabs[5]:
        pol = trace.policy
        st.write(
            f"Base tier `{pol.base_tier.value}` → proposed `{pol.proposed_tier.value}` "
            f"(profile: {pol.application})"
        )
        if pol.reason_codes:
            st.markdown(_chips(pol.reason_codes, "#0969da"), unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "rule": s.rule,
                        "fired": s.fired,
                        "tier before": s.tier_before,
                        "tier after": s.tier_after,
                        "effect": s.effect,
                        "detail (redacted)": s.detail,
                    }
                    for s in replay.decision_path
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    with tabs[6]:
        f = trace.fusion
        st.write(
            f"Weighted-only risk {f.weighted_only_risk:.2f} → fused {f.overall_risk:.2f} · "
            f"confidence {f.confidence:.2f} · uncertainty {f.uncertainty:.2f} · "
            f"dominant `{f.dominant_dimension}` · "
            f"multi-risk {'yes' if f.multi_risk else 'no'} · "
            f"severity rule {'applied' if f.severity_rule_applied else 'not applied'} · "
            f"floor {'applied' if f.severity_floor_applied else 'not applied'}"
        )
        st.caption(replay.risk_signals.fusion_explanation)


# ------------------------------------------------------------------ Command Center


def render_command_center(report: OperationalMonitoringReport, service) -> None:
    """
    Enterprise AI Risk Command Center. Every value is read from the
    OperationalMonitoringReport produced by the monitoring backend — the UI
    never recomputes a monitoring metric and never re-runs the pipeline.
    """
    r = report
    total = r.total_interactions

    st.markdown("## 🛰️ Enterprise AI Risk Command Center")
    st.caption(
        f"Demo / synthetic operational traffic · {total} recorded DecisionTrace "
        "record(s). Observational view — no detector, decision engine or "
        "verification pass is re-run here."
    )

    if total == 0:
        st.info(
            "**No operational traffic yet.** Use *Populate demo operational "
            "traffic* above to see the enterprise risk picture across "
            "applications."
        )
        return

    s = r.snapshot
    d = r.incident_digest

    # 1 — executive health
    st.markdown("### 1 · Executive health")
    e = st.columns(8)
    e[0].metric("Interactions", s.total_interactions)
    e[1].metric("Incident rate", _pctv(d.incident_rate))
    e[2].metric("Average risk", _numv(s.average_risk))
    e[3].metric("P95 risk", _numv(s.p95_risk))
    e[4].metric("Avg confidence", _numv(s.average_confidence))
    e[5].metric("Human review", _pctv(s.human_review_rate))
    e[6].metric("Block", _pctv(s.block_rate))
    e[7].metric("FAST-path", _pctv(s.fast_path_rate))
    st.caption(
        "RISK = estimated operational risk from ControlPlane.  CONFIDENCE = how "
        "sure ControlPlane is about that assessment — NOT the probability the AI "
        "response is correct.  **Risk ≠ confidence.**"
    )

    # 1b — routing & compute efficiency
    st.markdown("### 1b · Routing & compute efficiency")
    v = r.verification
    saved = v.estimated_bypass_compute_saved_ms
    re_ = st.columns(4)
    re_[0].metric("FAST path", _pctv(v.fast_rate),
                  help="Low-risk interactions that skipped DEEP verification.")
    re_[1].metric("DEEP path", _pctv(v.deep_rate))
    re_[2].metric(
        "Semantic bypasses", v.semantic_bypass_count,
        help="DEEP interactions where claim extraction / TF-IDF / NLI were "
             "skipped — a deterministic hard boundary blocks them regardless.",
    )
    re_[3].metric(
        "Compute cycles saved by bypass",
        "n/a" if saved is None else f"~{saved:,.0f} ms",
        help="Semantic-bypass count × the mean full-DEEP verification cost "
             "measured on non-bypassed DEEP interactions.",
    )
    if v.deep_count:
        st.caption(
            f"{v.semantic_bypass_count} of {v.deep_count} DEEP interactions "
            f"({_pctv(v.semantic_bypass_rate_of_deep)}) took the deterministic "
            "semantic bypass."
        )

    # 2 — enterprise risk distribution
    st.markdown("### 2 · Enterprise risk distribution")

    mt = r.multi_turn
    if mt.multi_turn_sessions:
        mc = st.columns(3)
        mc[0].metric("Multi-turn sessions", mt.multi_turn_sessions)
        mc[1].metric("Sessions hitting critical floor", mt.sessions_hitting_critical_floor)
        mc[2].metric(
            "Critical-floor session rate",
            _pctv(mt.critical_floor_session_rate),
            help="Multi-turn sessions where a BLOCK / critical-PII turn set the "
                 "non-decaying risk floor, forcing elevated scrutiny on every "
                 "later turn.",
        )
        if mt.sessions_hitting_critical_floor:
            st.warning(
                f"⚠️ **{mt.sessions_hitting_critical_floor} multi-turn "
                f"session(s) hit the non-decaying critical floor** "
                f"({mt.critical_floor_events} critical event(s) total). Later "
                "turns in those sessions inherited elevated scrutiny that does "
                "not decay."
            )

    buckets = r.risk_distribution.buckets
    st.bar_chart(
        pd.DataFrame(
            {"band": [b.bucket_name for b in buckets],
             "interactions": [b.count for b in buckets]}
        ).set_index("band")
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "band": b.bucket_name,
                    "range": f"{b.min_risk:.2f}–{min(b.max_risk, 1.0):.2f}",
                    "count": b.count,
                    "share": _pctv(None if b.percentage is None else b.percentage / 100),
                }
                for b in buckets
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    # 3 — application risk map
    st.markdown("### 3 · Application risk map")
    st.caption(
        "Different enterprise AI use cases show different **observed risk "
        "profiles**. A lower average risk is not a claim that an application is "
        "'safer'."
    )
    apps = r.applications
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "application": a.application,
                    "interactions": a.interaction_count,
                    "observed risk": round(a.average_risk, 3),
                    "p95": round(a.p95_risk, 3),
                    "confidence": round(a.average_confidence, 3),
                    "human review": _pctv(a.human_review_rate),
                    "block": _pctv(a.block_rate),
                    "FAST": _pctv(a.fast_path_rate),
                    "DEEP": _pctv(a.deep_path_rate),
                    "high-consequence": _pctv(a.high_consequence_rate),
                    "high-criticality": _pctv(a.high_criticality_rate),
                    "dominant dimension": a.dominant_risk_dimension or "—",
                }
                for a in apps
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    if apps:
        pick = st.selectbox(
            "Inspect an application", [a.application for a in apps], key="cc_app"
        )
        a = next(x for x in apps if x.application == pick)
        ac = st.columns(5)
        ac[0].metric("Interactions", a.interaction_count)
        ac[1].metric("Observed risk", _numv(a.average_risk))
        ac[2].metric("P95 risk", _numv(a.p95_risk))
        ac[3].metric("Confidence", _numv(a.average_confidence))
        ac[4].metric("Dominant dimension", a.dominant_risk_dimension or "—")
        dist = ", ".join(
            f"{k} {v}" for k, v in a.decision_distribution.items() if v
        )
        st.write(f"**Decision distribution:** {dist or '(none)'}")

    # 4 — incident command
    st.markdown("### 4 · 🚨 Incident command")
    ic = st.columns(4)
    ic[0].metric("Incidents", d.total)
    ic[1].metric("Critical", d.by_severity.get("CRITICAL", 0))
    ic[2].metric("High", d.by_severity.get("HIGH", 0))
    ic[3].metric("Medium", d.by_severity.get("MEDIUM", 0))
    st.caption(f"Incident rule: {d.incident_definition}")
    if r.incidents:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "severity": i.severity.value,
                        "interaction_id": i.interaction_id,
                        "application": i.application,
                        "decision": i.decision,
                        "risk": round(i.overall_risk, 2),
                        "confidence": round(i.confidence, 2),
                        "dominant": i.dominant_dimension or "—",
                        "reason codes": ", ".join(i.reason_codes) or "—",
                        "verification": i.verification_path,
                        "consequence": round(i.consequence_score, 2),
                        "criticality": round(i.criticality, 2),
                        "triggers": ", ".join(i.triggers),
                    }
                    for i in r.incidents
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        chosen = st.selectbox(
            "Select an incident to investigate",
            ["—"] + [i.interaction_id for i in r.incidents],
            key="cc_incident",
        )
        if chosen and chosen != "—":
            if st.button("🔎 Investigate Incident", type="primary", key="cc_investigate"):
                st.session_state["investigate_id"] = chosen
    else:
        st.success("No incidents in the current operational window.")

    inv_id = st.session_state.get("investigate_id")
    if inv_id:
        st.divider()
        investigation = service.investigate_incident(inv_id)
        if getattr(investigation, "found", False):
            render_investigation(investigation, service)
        else:
            st.info(getattr(investigation, "message", "Investigation not available."))

    # 5 — risk drivers
    st.markdown("### 5 · Why ControlPlane intervenes")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "reason code": rc.reason_code,
                    "frequency": rc.count,
                    "share of interventions": _pctv(rc.share_of_interventions),
                }
                for rc in r.reason_codes
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    # 6 — detector overview
    st.markdown("### 6 · Detector overview (observational)")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "detector": det.detector,
                    "coverage": det.interaction_coverage,
                    "average risk": round(det.average_risk, 3),
                    "high-risk rate": _pctv(det.high_risk_rate),
                    "mean weighted contribution": (
                        None
                        if det.mean_weighted_contribution is None
                        else round(det.mean_weighted_contribution, 3)
                    ),
                    "dominant-dimension count": det.dominant_dimension_count,
                }
                for det in r.detectors
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    # 7 — progressive verification
    st.markdown("### 7 · Progressive verification (FAST vs DEEP)")
    v = r.verification
    vc = st.columns(6)
    vc[0].metric("FAST", _pctv(v.fast_rate))
    vc[1].metric("DEEP", _pctv(v.deep_rate))
    vc[2].metric("Avg FAST latency", _numv(v.average_fast_latency_ms))
    vc[3].metric("Avg DEEP latency", _numv(v.average_deep_latency_ms))
    vc[4].metric("Avg verif. latency", _numv(v.average_total_verification_latency_ms))
    vc[5].metric("P95 verif. latency", _numv(v.p95_total_verification_latency_ms))
    st.caption(
        "Low-risk interactions take the cheaper FAST path; ambiguous or "
        "consequential interactions receive DEEP verification."
    )
    if v.semantic_bypass_count:
        st.caption(
            f"Of the {v.deep_count} DEEP interactions, "
            f"**{v.semantic_bypass_count}** ({_pctv(v.semantic_bypass_rate_of_deep)}) "
            "took the deterministic semantic bypass — no claim extraction / "
            "retrieval / NLI."
        )
    if v.deep_trigger_reason_counts:
        st.markdown("**Most common DEEP triggers**")
        st.dataframe(
            pd.DataFrame(
                sorted(
                    v.deep_trigger_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])
                ),
                columns=["trigger", "count"],
            ),
            hide_index=True,
            use_container_width=True,
        )

    # 8 — operational trends
    st.markdown("### 8 · Operational trends")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "metric": mt.metric,
                    "first half": _numv(mt.first_half_value),
                    "second half": _numv(mt.second_half_value),
                    "delta": _numv(mt.delta),
                    "direction": f"{_TREND_ARROW.get(mt.direction.value, '')} "
                    f"{mt.direction.value}",
                }
                for mt in r.trend.metrics
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(r.trend.method)

    # 9 — operational shifts
    st.markdown("### 9 · Operational shifts (recent vs historical)")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "metric": sh.metric,
                    "historical": _numv(sh.baseline_value),
                    "recent": _numv(sh.recent_value),
                    "delta": _numv(sh.delta),
                    "direction": f"{_TREND_ARROW.get(sh.direction, '')} {sh.direction}",
                }
                for sh in r.operational_shift.shifts
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(r.operational_shift.disclaimer)

    # 10 — governance feedback
    st.markdown("### 10 · Governance feedback")
    fb = r.feedback
    fc = st.columns(6)
    fc[0].metric("Feedback count", fb.feedback_count)
    fc[1].metric("Approved", fb.approved)
    fc[2].metric("Modified", fb.modified)
    fc[3].metric("Rejected", fb.rejected)
    fc[4].metric("Override rate", _pctv(fb.override_rate))
    fc[5].metric("Approval rate", _pctv(fb.approval_rate))
    st.caption(fb.note)

    dq = r.data_quality
    st.caption(
        f"Data basis: {dq.valid_records} valid trace(s), "
        f"{dq.invalid_records_skipped} skipped — operational view, not production "
        "traffic."
    )


# ------------------------------------------------------------------ Incident Investigation


def render_investigation(investigation, service) -> None:
    """
    Enterprise incident investigation workspace. Read-only reconstruction
    (replay + explainability) + a human governance workflow. The automated
    ControlPlane decision is never mutated here.
    """
    from investigation.schemas import (
        ACTIONS_REQUIRING_COMMENT,
        ACTIONS_REQUIRING_REVIEWER_DECISION,
        GovernanceActionType,
    )

    inv = investigation
    exp = inv.explanation
    iid = inv.interaction_id
    inc = inv.incident

    st.markdown("## 🔎 Incident Investigation")
    st.caption(
        "Reconstructed from the stored decision trace — no detector, decision "
        "engine, fusion, policy or verification pass is re-run."
    )

    # ---- incident header ------------------------------------------------
    h = st.columns(4)
    h[0].markdown(f"**Application**\n\n{inc.application if inc else exp.risk_summary.dominant_dimension}")
    h[1].markdown(f"**Interaction ID**\n\n`{iid}`")
    h[2].markdown(
        f"**Timestamp**\n\n{inc.timestamp:%Y-%m-%d %H:%M}" if inc else "**Timestamp**\n\n—"
    )
    h[3].markdown(
        f"**Severity**\n\n{inc.severity.value}" if inc else "**Severity**\n\nnot flagged"
    )
    if inc is not None:
        st.caption(f"Incident triggers: {', '.join(inc.triggers)} · {inc.severity_rationale}")

    # ---- Original vs Effective Governed Decision -----------------------
    og, eff = st.columns(2)
    og.metric("Original ControlPlane decision", inv.original_decision,
              help="The automated decision on the stored DecisionTrace. Immutable.")
    eff.metric(
        "Effective governed decision", inv.effective_governed_decision,
        delta=("human override" if inv.is_overridden else None),
        delta_color="off",
        help="The latest reviewer override if one exists, otherwise the "
             "original automated decision. A view over the append-only "
             "governance track — the DecisionTrace is never changed.",
    )
    if inv.is_overridden:
        st.warning(
            f"⚖️ A reviewer has overridden this interaction: "
            f"**{inv.original_decision} → {inv.effective_governed_decision}**. "
            f"The automated ControlPlane decision (`{inv.original_decision}`) "
            "remains immutable on the trace."
        )
    else:
        st.info(
            f"The automated ControlPlane decision is **{inv.original_decision}** "
            "and is immutable. Reviewer actions below record a human governance "
            "outcome — they never change this decision."
        )

    # ---- A. executive assessment -------------------------------------
    st.markdown("### A · Executive assessment")
    a = st.columns(5)
    a[0].metric("ControlPlane decision", exp.decision.value)
    a[1].metric("Risk", _pctv(exp.overall_risk))
    a[2].metric("Confidence", _pctv(exp.decision_confidence))
    a[3].metric("Verification", exp.verification_path.value)
    a[4].metric("Human review", "YES" if exp.human_review_required else "NO")

    # ---- B..G. reuse the existing explainability presentation --------
    render_explanation(exp)

    # ---- H. counterfactual (explicit SIMULATION) ---------------------
    st.markdown("### What if?  (counterfactual simulation)")
    st.caption(
        "Run the existing simulation engine with a production-visible field "
        "changed. This is a SIMULATION — it never modifies the stored decision "
        "or production policy."
    )
    cf_cols = st.columns([2, 2, 2])
    cf_field = cf_cols[0].selectbox(
        "Field to change",
        ["action_amount_inr", "affected_entities", "action_type"],
        key="inv_cf_field",
    )
    if cf_field == "action_type":
        cf_value = cf_cols[1].selectbox(
            "New value", [a.value for a in ActionType], key="inv_cf_value_at"
        )
    elif cf_field == "affected_entities":
        cf_value = cf_cols[1].number_input(
            "New value", 1, 1_000_000, 1, key="inv_cf_value_ent"
        )
    else:
        cf_value = cf_cols[1].number_input(
            "New value (₹)", 0.0, 100_000_000.0, 0.0, key="inv_cf_value_amt"
        )
    if cf_cols[2].button("Run counterfactual", key="inv_cf_run"):
        st.session_state["inv_cf_result"] = service.investigation_counterfactual(
            iid, {cf_field: cf_value}
        )
    cf = st.session_state.get("inv_cf_result")
    if cf is not None and getattr(cf, "interaction_id", None) == iid:
        cc = st.columns(2)
        cc[0].metric("Current decision", cf.current_decision)
        cc[1].metric("Counterfactual decision", cf.counterfactual_decision)
        st.write(
            f"Risk {cf.current_overall_risk:.2f} → {cf.counterfactual_overall_risk:.2f} · "
            f"decision changed: **{cf.decision_changed}**"
        )
        if cf.rules_removed:
            st.write("Rules that stopped firing: " + ", ".join(f"`{x}`" for x in cf.rules_removed))
        if cf.rules_added:
            st.write("Rules that started firing: " + ", ".join(f"`{x}`" for x in cf.rules_added))
        st.warning(cf.note)

    # ---- I. human governance & override ----------------------------
    st.markdown("### ⚖️ Human Governance & Override")
    st.caption(
        "Recorded on a separate, append-only governance track. The original "
        "automated DecisionTrace is never mutated."
    )
    st.write(f"**Governance status:** `{inv.investigation_status.value}`")
    if not inv.available_actions:
        st.success("This investigation is CLOSED — no further governance actions.")
    else:
        action = st.selectbox(
            "Governance action",
            [a.value for a in inv.available_actions],
            key="inv_action",
        )
        action_enum = GovernanceActionType(action)
        comment = ""
        if action_enum in ACTIONS_REQUIRING_COMMENT:
            comment = st.text_area("Comment (required)", key="inv_comment")
        else:
            comment = st.text_area("Comment (optional)", key="inv_comment_opt")
        reviewer_decision = None
        if action_enum in ACTIONS_REQUIRING_REVIEWER_DECISION:
            reviewer_decision = st.selectbox(
                "Reviewer decision (the tier you would have chosen)",
                [t.value for t in InterventionTier],
                key="inv_reviewer_decision",
            )
        actor = st.text_input("Reviewer", "demo-reviewer", key="inv_actor")
        if st.button("Submit governance action", type="primary", key="inv_submit"):
            try:
                service.record_governance_action(
                    iid,
                    action=action,
                    actor=actor,
                    comment=comment,
                    reviewer_decision=reviewer_decision,
                )
                st.success(f"Recorded governance action: {action}.")
            except Exception as exc:  # noqa: BLE001 — surface validation errors
                st.error(str(exc))

    # ---- J. governance history ------------------------------------
    st.markdown("### Governance history")
    history = service.governance_history(iid)
    if not history:
        st.caption("No governance actions recorded yet — status is OPEN.")
    else:
        st.markdown(
            f"**{inc.timestamp:%d %b %H:%M}** · Incident created · automated decision "
            f"`{inv.original_decision}`" if inc else "Incident created."
        )
        for act in history:
            line = (
                f"**{act.timestamp:%d %b %H:%M}** · {act.actor} · **{act.action.value}** "
                f"({act.previous_status.value} → {act.new_status.value})"
            )
            if act.reviewer_decision:
                line += (
                    f" · reviewer decision `{act.reviewer_decision}` "
                    f"(automated `{act.original_decision}` unchanged)"
                )
            st.markdown(line)
            if act.comment:
                st.caption(f"“{act.comment}”")
    st.divider()
    st.caption(
        "Automated ControlPlane decision remains immutable. Reviewer feedback is a "
        "governance signal, not ground truth."
    )
    if st.button("Close investigation view", key="inv_close_view"):
        st.session_state.pop("investigate_id", None)
        st.session_state.pop("inv_cf_result", None)


# ------------------------------------------------------------------ Governance Intelligence


def render_governance(report, service) -> None:
    """
    Governance Intelligence & Closed-Loop Monitoring. Reads a
    GovernanceIntelligenceReport (stored traces + governance actions +
    feedback). Recomputes nothing; recommends only.
    """
    r = report
    o = r.overview

    st.markdown("## 🧭 Governance Intelligence")
    st.caption(
        "Closed loop: traffic → detection → decision → monitoring → incident → "
        "human review → **governance signal** → analysis → **calibration "
        "recommendation** (never applied). Read-only; no pipeline is re-run."
    )

    if o.traffic.total_interactions == 0:
        st.info(
            "No operational traffic yet. Open the **🛰️ Command Center** tab and "
            "**Populate demo operational traffic**, then return here."
        )
        return

    # 1 — executive governance overview
    st.markdown("### 1 · Executive governance overview")
    d, rk, v = o.decisions, o.risk, o.verification
    e = st.columns(7)
    e[0].metric("Interactions", o.traffic.total_interactions)
    e[1].metric("High-risk", _pctv(rk.high_risk_rate))
    e[2].metric("Human oversight", _pctv(d.human_oversight_rate))
    e[3].metric("Block", _pctv(d.block_rate))
    e[4].metric("Reviewer override", _pctv(o.reviewer_disagreement.override_rate))
    e[5].metric("FAST", _pctv(v.fast_rate))
    e[6].metric("DEEP", _pctv(v.deep_rate))

    # 2 — application comparison
    st.markdown("### 2 · Application comparison")
    st.caption(
        "The application name is a **dimension of analysis** — there is no "
        "application-specific risk logic. A lower rate is an observed profile, "
        "not a claim that an app is 'safer'."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "application": a.application,
                    "volume": a.volume,
                    "risk": _numv(a.average_risk),
                    "p95": _numv(a.p95_risk),
                    "confidence": _numv(a.average_confidence),
                    "low-conf": _pctv(a.low_confidence_rate),
                    "human review": _pctv(a.human_review_rate),
                    "block": _pctv(a.block_rate),
                    "FAST": _pctv(a.fast_rate),
                    "DEEP": _pctv(a.deep_rate),
                    "incident rate": _pctv(a.incident_rate),
                    "reviewer override": _pctv(a.reviewer_override_rate),
                }
                for a in r.application_comparison.applications
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    hs = r.application_comparison
    st.caption(
        f"highest risk: **{hs.highest_risk}** · highest volume: **{hs.highest_volume}** "
        f"· lowest oversight: **{hs.lowest_intervention}**"
    )

    # 3 — policy intelligence
    st.markdown("### 3 · Policy intelligence")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "rule": p.rule,
                    "fire count": p.fire_count,
                    "fire rate": _pctv(p.fire_rate),
                    "tier changes": p.tier_changing_count,
                    "→ HUMAN_REVIEW": p.human_review_count,
                    "→ BLOCK": p.block_count,
                }
                for p in o.policy.rules[:12]
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    # 4 — reviewer signals
    st.markdown("### 4 · Reviewer signals")
    st.warning(
        "**automated decision ≠ reviewer signal ≠ ground truth.** Reviewer "
        "decisions are recorded *alongside* the immutable automated decision; "
        "they are a governance signal, not a correctness label."
    )
    rd = o.reviewer_disagreement
    sc = st.columns(4)
    sc[0].metric("Reviewed", rd.reviewed_count)
    sc[1].metric("Overrides", rd.override_count)
    sc[2].metric("Override rate", _pctv(rd.override_rate))
    sc[3].metric("Signals", r.signals.signal_count)
    if rd.automated_to_reviewer_transitions:
        st.markdown("**Automated → reviewer transitions**")
        st.dataframe(
            pd.DataFrame(
                rd.automated_to_reviewer_transitions.items(),
                columns=["automated → reviewer", "count"],
            ),
            hide_index=True,
            use_container_width=True,
        )
    st.markdown("**Governance signals by type**")
    st.dataframe(
        pd.DataFrame(r.signals.by_signal_type.items(), columns=["signal type", "count"]),
        hide_index=True,
        use_container_width=True,
    )

    # 5 — governance insights
    st.markdown("### 5 · Governance insights")
    if not r.insights:
        st.success("No governance threshold was crossed for the current traffic.")
    for ins in r.insights:
        with st.container(border=True):
            st.markdown(f"**[{ins.severity.value}] {ins.code}** — {ins.title}")
            st.caption("Why?  " + ins.explanation)
            st.write("Evidence: " + ", ".join(
                f"`{k}={v}`" for k, v in ins.supporting_metrics.items() if v is not None
            ))
            st.write(f"Recommended next action: **{ins.recommended_action.value}**")
            if ins.example_interaction_ids:
                st.caption(
                    "Affected incidents (inspect in the Command Center → Investigate): "
                    + ", ".join(f"`{x}`" for x in ins.example_interaction_ids)
                )

    # 6 — trend monitoring
    st.markdown("### 6 · Trend monitoring")
    st.caption(r.trends.basis + " — no statistical-significance claim.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "metric": s.metric,
                    "baseline": _numv(s.baseline),
                    "current": _numv(s.current),
                    "direction": f"{_TREND_ARROW.get(s.direction.value, '')} {s.direction.value}",
                    "magnitude": _numv(s.magnitude),
                    "label": s.label,
                }
                for s in r.trends.signals
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    # 7 — calibration recommendations
    st.markdown("### 7 · Calibration recommendations")
    st.error("**RECOMMENDATION ONLY — NOT APPLIED TO PRODUCTION.**")
    for rec in r.recommendations:
        with st.container(border=True):
            head = f"**{rec.recommendation_type.value}**"
            if rec.application:
                head += f" · application: `{rec.application}`"
            head += f" · disposition: **{rec.disposition.value}**"
            st.markdown(head)
            st.caption(rec.rationale)
            if rec.current_configuration and rec.candidate_configuration:
                cc = st.columns(2)
                cc[0].markdown("**Current configuration**")
                cc[0].json(rec.current_configuration)
                cc[1].markdown("**Recommended candidate**")
                cc[1].json(rec.candidate_configuration)
                if rec.expected_tradeoff:
                    st.write("Expected trade-off: " + rec.expected_tradeoff)
                if rec.safety_constraints:
                    st.caption("Safety constraints: " + str(rec.safety_constraints))
            if rec.points_to:
                st.caption("Points to: " + ", ".join(rec.points_to))
            st.caption(rec.disclaimer)

    st.divider()
    st.caption(
        "Governance analytics is read-only. It never re-runs the pipeline, never "
        "reads ground truth, and never writes production configuration. Reviewer "
        "feedback is a governance signal, not ground truth."
    )


# ------------------------------------------------------------------ Adaptive Intelligence


def render_adaptive(report, service) -> None:
    """
    🧠 Adaptive Intelligence — incident patterns → drift → attribution →
    adaptive recommendation → counterfactual → HUMAN APPROVAL →
    APPROVED_FOR_EVALUATION. Production configuration is never modified.
    """
    r = report
    st.markdown("## 🧠 Adaptive Intelligence")
    st.caption(
        "Closed loop: detect → decide → explain → record → observe patterns → "
        "detect drift → attribute → **propose a safe intervention** → simulate → "
        "check safety → **human approval** → APPROVED_FOR_EVALUATION."
    )
    st.error(
        "**Recommendations are RECOMMENDATION ONLY.** Approval means "
        "`APPROVED_FOR_EVALUATION` — there is **no auto-deployment path** and "
        "`config/settings.yaml` is never modified. "
        f"Production configuration: **{r.production_configuration_status}**."
    )

    # A — adaptive executive summary
    st.markdown("### A · Adaptive executive summary")
    a = st.columns(5)
    a[0].metric("Active patterns", r.pattern_count)
    a[1].metric("Drift signals", r.drift_signal_count)
    a[2].metric("Potential drift", r.potential_drift_count)
    a[3].metric("Reviewer overrides", r.reviewer_override_count)
    a[4].metric("Approved for eval", r.approved_for_evaluation_count)
    st.info(f"**Observation:** {r.observation}")

    # B — incident patterns
    st.markdown("### B · Incident patterns")
    if r.top_patterns:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "pattern": p["type"],
                        "application": ", ".join(p["applications"]) or "-",
                        "severity": p["severity"],
                        "count": p["incident_count"],
                        "detection confidence": p["detection_confidence"],
                        "dominant rule": p["affected_policy_rule"] or "-",
                        "dominant dimension": p["affected_dimension"] or "-",
                        "next step": p["recommended_next_step"],
                    }
                    for p in r.top_patterns
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "'detection confidence' = confidence that the operational pattern is real "
            "given the sample — NOT a statement about whether any AI response or "
            "decision was correct."
        )
    else:
        st.success("No recurring incident pattern in the current window.")

    # C — drift
    st.markdown("### C · Drift")
    if r.drift_signals:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "metric": s["metric"],
                        "scope": s["scope"],
                        "historical": _numv(s["baseline"]),
                        "recent": _numv(s["recent"]),
                        "delta": _numv(s["delta"]),
                        "signal": s["signal"],
                        "explanation": s["explanation"],
                    }
                    for s in r.drift_signals
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.success("All monitored metrics are STABLE between the historical and recent windows.")
    st.caption("Operational drift signal — **not proof of model degradation**. No statistical-significance test is performed.")

    # D — root-cause / attribution
    st.markdown("### D · Root-cause / attribution")
    intel = service.incident_intelligence()
    if intel.attributions:
        at = intel.attributions[0]
        st.write(f"**Dominant dimension:** {at.dominant_dimension or '—'}")
        for code, share in list(at.reason_code_shares.items())[:5]:
            bar = "█" * max(1, int(round(share * 20)))
            st.text(f"{code:28s} {bar} {share:.0%}")
        for tier, share in list(at.decision_shares.items())[:3]:
            bar = "█" * max(1, int(round(share * 20)))
            st.text(f"{tier:28s} {bar} {share:.0%}")
        st.caption(at.narrative)
    st.caption("**Observed association / attribution, not causal proof.**")

    # E — recommendations
    st.markdown("### E · Recommendations")
    for rec in r.recommendations:
        with st.container(border=True):
            head = f"**{rec.type.value}**"
            if rec.application:
                head += f" · `{rec.application}`"
            head += f" · {rec.severity.value} · status **{rec.status.value}**"
            st.markdown(head)
            st.caption(rec.rationale)
            st.write("Proposed change: " + rec.proposed_change)
            if rec.expected_tradeoff:
                st.write("Expected trade-off: " + rec.expected_tradeoff)

            # F — counterfactual
            sr = rec.simulation_result
            if sr is not None:
                st.markdown("**F · Counterfactual (CURRENT → CANDIDATE)**")
                rows = [
                    ("recall", sr.current_recall, sr.candidate_recall),
                    ("precision", sr.current_precision, sr.candidate_precision),
                    ("false-positive rate", sr.current_false_positive_rate, sr.candidate_false_positive_rate),
                    ("missed-risk rate", sr.current_missed_risk_rate, sr.candidate_missed_risk_rate),
                    ("FAST rate", sr.current_fast_rate, sr.candidate_fast_rate),
                    ("DEEP rate", sr.current_deep_rate, sr.candidate_deep_rate),
                    ("human-review rate", sr.current_human_review_rate, sr.candidate_human_review_rate),
                    ("avg latency (ms)", sr.current_average_latency_ms, sr.candidate_average_latency_ms),
                ]
                st.dataframe(
                    pd.DataFrame(
                        [{"metric": m, "current": _numv(c), "candidate": _numv(k)} for m, c, k in rows]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                if sr.safety_passed:
                    st.success("SAFETY: PASS — candidate satisfies the configured safety constraints (evaluated FIRST).")
                else:
                    st.error(
                        "SAFETY: no safe candidate — "
                        + (sr.selection_reason.split(".")[0] + ".")
                    )
                st.caption("Safety constraints: " + str(sr.safety_constraints))

            # G — approval gate
            if rec.type.value != "NO_ACTION":
                st.markdown("**G · Approval**")
                if rec.approval is not None:
                    st.info(
                        f"{rec.approval.decision} by {rec.approval.actor}. "
                        + rec.approval.disclaimer
                    )
                else:
                    ac = st.columns([2, 2, 4])
                    if ac[0].button("APPROVE FOR EVALUATION", key=f"ad_approve_{rec.recommendation_id}"):
                        service.adaptive_approve(
                            rec.recommendation_id, actor="demo-reviewer",
                            comment="Approved for evaluation.",
                        )
                        st.success(
                            "Approved for evaluation only. Production configuration remains unchanged."
                        )
                    if ac[1].button("REJECT", key=f"ad_reject_{rec.recommendation_id}"):
                        service.adaptive_reject(
                            rec.recommendation_id, actor="demo-reviewer", comment="Rejected.",
                        )
                        st.warning("Rejected.")
            st.caption(rec.disclaimer)

    st.divider()
    st.caption(
        "Adaptive intelligence is read-only w.r.t. the pipeline and production "
        "configuration. Counterfactuals reuse calibration.sweep / calibration.select. "
        "Reviewer feedback is a governance signal, not ground truth."
    )


# ------------------------------------------------------------------ Executive Command Center


_POSTURE_COLOR = {"LOW": "#1a7f37", "MODERATE": "#9a6700", "HIGH": "#cf222e"}


def render_executive(view, service) -> None:
    """
    🏢 Executive — judge-facing consolidated view. Read-only presentation of
    already-computed reports; expensive simulation runs only on request.
    """
    cc = view
    es = cc.executive_summary
    k = cc.kpi
    rp = cc.risk_posture

    st.markdown("## 🏢 ControlPlane.ai — Enterprise AI Risk Control Plane")
    st.caption(
        "*We don't just detect AI risk — we control what happens next.* "
        "This view is read-only; no detector / decision engine / verification is re-run."
    )

    if not k.has_data:
        st.info("**No operational data available.**")
        if st.button("Populate with demo traffic", type="primary", key="exec_populate"):
            n = service.populate_operational_demo(250)
            st.success(f"Recorded {n} interactions — click any control to refresh.")
        return

    # ---- executive story mode ----
    st.markdown("### Executive summary")
    e = st.columns(6)
    e[0].metric("AI systems monitored", es.ai_systems_monitored)
    e[1].metric("Interactions evaluated", es.interactions_evaluated)
    e[2].metric("High-risk interactions", es.high_risk_interactions)
    e[3].metric("Human oversight", es.human_oversight_count)
    e[4].metric("Potential drift signals", es.potential_drift_signals)
    e[5].metric("Open recommendations", es.open_governance_recommendations)

    badge = " · ".join(
        f"{b.application}: **{b.posture}**" for b in es.application_posture
    )
    st.markdown("**Application posture:** " + badge)
    if es.top_risk_dimension:
        st.markdown(f"**Top risk dimension:** {es.top_risk_dimension}")
    if es.top_governance_issue:
        st.markdown(f"**Top governance issue:** {es.top_governance_issue}")
    if es.recommended_action:
        st.markdown(f"**Recommended action:** {es.recommended_action}")
    st.success(f"**Safety status:** {es.safety_status}")

    # ---- A. executive KPI strip ----
    st.markdown("### A · Executive KPIs")
    a = st.columns(7)
    a[0].metric("Interactions", k.total_interactions)
    a[1].metric("Allow", _pctv(k.allow_rate))
    a[2].metric("Verify", _pctv(k.verify_rate))
    a[3].metric("Human review", _pctv(k.human_review_rate))
    a[4].metric("Block", _pctv(k.block_rate))
    a[5].metric("Incident rate", _pctv(k.incident_rate))
    a[6].metric("Override rate", _pctv(k.override_rate))
    a2 = st.columns(7)
    a2[0].metric("Avg risk", _numv(k.average_risk))
    a2[1].metric("Avg confidence", _numv(k.average_confidence))
    a2[2].metric("FAST", _pctv(k.fast_rate))
    a2[3].metric("DEEP", _pctv(k.deep_rate))
    a2[4].metric("Annotate", _pctv(k.annotate_rate))
    a2[5].metric("Potential drift", k.potential_drift_count)
    a2[6].metric("Active recs", k.active_recommendations)
    st.caption("Every number is derived from actual stored / generated traces.")

    # ---- B. risk posture ----
    st.markdown("### B · Risk posture")
    b = st.columns(4)
    b[0].metric("Performance", _numv(rp.performance_average),
                help=f"{rp.performance_high_risk_count} high-risk")
    b[1].metric("Responsibility", _numv(rp.responsibility_average),
                help=f"{rp.responsibility_high_risk_count} high-risk")
    b[2].metric("Cost", _numv(rp.cost_average), help=f"{rp.cost_high_risk_count} high-risk")
    b[3].metric("Overall", _numv(rp.overall_average),
                help=f"{rp.overall_high_risk_count} high-risk · trend {rp.risk_trend or 'n/a'}")
    st.caption(f"Dominant risk dimension: **{rp.dominant_dimension or '—'}** "
               f"(threshold for 'high risk': {rp.high_risk_threshold})")

    # ---- C. application risk matrix ----
    st.markdown("### C · Application risk matrix")
    st.caption("Different AI applications have different risk profiles → different governance.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "application": r.application,
                    "posture": r.posture,
                    "interactions": r.interactions,
                    "avg risk": _numv(r.average_risk),
                    "high-risk": _pctv(r.high_risk_rate),
                    "human review": _pctv(r.human_review_rate),
                    "block": _pctv(r.block_rate),
                    "FAST": _pctv(r.fast_rate),
                    "DEEP": _pctv(r.deep_rate),
                    "dominant dim": r.dominant_risk_dimension or "—",
                    "incidents": r.incident_count,
                    "recommended posture": r.recommended_posture or "—",
                }
                for r in cc.application_posture
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    # ---- D. risk heatmap ----
    st.markdown("### D · Risk heatmap")
    hm = cc.heatmap
    grid = {app: {} for app in hm.applications}
    for cell in hm.cells:
        grid[cell.application][cell.dimension] = (
            "N/A" if cell.value is None else round(cell.value, 3)
        )
    st.dataframe(
        pd.DataFrame(
            [{"application": app, **grid[app]} for app in hm.applications]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(hm.note)

    # ---- live control feed ----
    st.markdown("### Live control feed (newest first · STORED TRACE)")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "interaction_id": d.interaction_id,
                    "application": d.application,
                    "decision": d.decision,
                    "risk": round(d.overall_risk, 2),
                    "confidence": round(d.confidence, 2),
                    "verification": d.verification_path,
                    "dominant": d.dominant_dimension or "—",
                    "reason codes": ", ".join(d.reason_codes) or "—",
                    "human review": d.human_review_required,
                    "timestamp": d.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "source": d.source,
                }
                for d in cc.recent_decisions
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    _feed_ids = ["—"] + [d.interaction_id for d in cc.recent_decisions]
    _pick = st.selectbox("Inspect a decision (stored trace → explainability → replay)",
                         _feed_ids, key="exec_feed_pick")
    if _pick and _pick != "—":
        _t = service.get_audit_trace(_pick)
        if _t is not None:
            st.caption("LIVE / STORED TRACE — not a simulation.")
            render_trace(_t)

    # ---- governance audit timeline ----
    st.markdown("### Governance audit timeline")
    tl = service.governance_timeline()
    for ev in tl.events:
        ts = ev.timestamp.strftime("%d %b %H:%M") if ev.timestamp else "(order only)"
        st.markdown(f"**{ev.order}. {ev.event_type}** · `{ts}` · `{ev.entity}`")
        st.caption(ev.description + ("  " + ev.timestamp_note if ev.timestamp_note else ""))
    st.caption(tl.note)

    # ---- one-click enterprise demo ----
    st.markdown("### ▶ Run enterprise demo")
    st.caption(
        "Deterministic, bounded demo over the REAL pipeline. Runs the counterfactual "
        "(calibration.sweep + calibration.select) — takes ~1–2 minutes."
    )
    if st.button("▶ RUN ENTERPRISE DEMO", type="primary", key="exec_run_demo"):
        with st.spinner("Running the full governance loop over the real pipeline…"):
            st.session_state["exec_demo"] = service.enterprise_demo(with_counterfactual=True)
    demo = st.session_state.get("exec_demo")
    if demo is not None:
        for s in demo.steps:
            st.markdown(f"**STEP {s.step} · {s.title}**")
            st.write(s.detail)
        st.success(
            f"Counterfactual safety: **{demo.counterfactual_safety}** · "
            f"Approval: **{demo.approval_status or '—'}** · "
            f"Production configuration: **{demo.production_configuration_status}**"
        )

    # ---- what-if / policy playground ----
    st.markdown("### 🔬 What-If — policy playground")
    st.caption(
        "Explore an EXISTING calibration control. Runs calibration.sweep + "
        "calibration.select on request. Safety is evaluated FIRST."
    )
    wc = st.columns([2, 2, 2, 2])
    _wapp = wc[0].selectbox(
        "Application (context)", ["(all)"] + [r.application for r in cc.application_posture],
        key="exec_wi_app",
    )
    _wctrl = wc[1].selectbox(
        "Control",
        ["deep_verification_risk_threshold", "fast_path_min_confidence"],
        key="exec_wi_ctrl",
    )
    _wrecall = wc[2].slider("minimum_recall", 0.50, 1.0, 0.90, 0.01, key="exec_wi_recall")
    if wc[3].button("Run What-If", key="exec_wi_run"):
        with st.spinner("Simulating…"):
            st.session_state["exec_wi"] = service.enterprise_whatif(
                application=None if _wapp == "(all)" else _wapp,
                control=_wctrl,
                minimum_recall=float(_wrecall),
            )
    wi = st.session_state.get("exec_wi")
    if wi is not None:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "metric": m.metric,
                        "CURRENT": _numv(m.current),
                        "CANDIDATE": _numv(m.candidate),
                        "": _TREND_ARROW.get(m.direction, ""),
                    }
                    for m in wi.metrics
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        if wi.safety_status == "PASS":
            st.success("SAFETY ✓ PASS — candidate satisfies the configured safety constraints.")
        elif wi.safety_status == "NO_CANDIDATE":
            st.error("SAFETY — no safe configuration recommended. Constraints were NOT relaxed.")
        else:
            st.warning(f"SAFETY: {wi.safety_status}")
        st.info("INTERPRETATION: " + wi.interpretation)
        st.caption("Safety constraints: " + str(wi.safety_constraints))
        st.caption(wi.disclaimer)

    # ---- technical architecture ----
    with st.expander("🛠️ Technical architecture (visual only — nothing here is executed)"):
        arch = service.enterprise.technical_architecture()
        st.dataframe(
            pd.DataFrame(
                [{"stage": s.stage, "module": s.module, "what it does": s.description}
                 for s in arch.stages]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(arch.note)

    st.divider()
    st.caption(
        "Enterprise view is read-only. Recommendations are RECOMMENDATION ONLY; "
        "approval means APPROVED_FOR_EVALUATION; production configuration is never "
        "modified and there is no deployment path."
    )


# ------------------------------------------------------------------ header

svc = get_service()

st.title("🛡️ ControlPlane.ai")
st.caption(
    "Real-time AI risk **governance control plane** — not just a content classifier. "
    "Heuristic, deterministic system on synthetic data."
)
flow = st.columns(5)
for col, (label, sub) in zip(
    flow,
    [
        ("DETECT", "performance · responsibility · cost"),
        ("RISK + CONFIDENCE", "how bad · how sure"),
        ("CRITICALITY", "does it matter if wrong?"),
        ("POLICY", "per-application rules"),
        ("INTERVENE", "ALLOW → BLOCK + human review"),
    ],
):
    col.markdown(
        f"<div style='text-align:center;padding:8px;border:1px solid #d0d7de;"
        f"border-radius:8px;'><b>{label}</b><br><span style='font-size:11px;"
        f"color:#57606a;'>{sub}</span></div>",
        unsafe_allow_html=True,
    )

st.divider()

page = st.tabs(
    [
        "🔍 Check a response",
        "🎛️ Policy simulation",
        "🔀 Counterfactual",
        "🔁 Multi-turn session",
        "📋 Audit",
        "💬 Feedback",
        "🛰️ Command Center",
        "🧭 Governance",
        "🧠 Adaptive Intelligence",
        "🏢 Executive",
    ]
)

# ------------------------------------------------------------------ CHECK tab

with page[0]:
    # Seed each field's session_state key once (widgets below are keyed and
    # read from it, so "Try a scenario" can repopulate the form reliably).
    for _k, _v in DEFAULT_FORM.items():
        st.session_state.setdefault(f"f_{_k}", _v)

    def _load_scenario(scenario_key: str) -> None:
        for name, value in interaction_to_form(scenarios.ALL_SINGLE_TURN[scenario_key]()).items():
            st.session_state[f"f_{name}"] = value
        st.session_state.pop("result_trace", None)

    st.subheader("Try a scenario")
    cols = st.columns(len(_SCENARIO_LABELS))
    for col, (key, label) in zip(cols, _SCENARIO_LABELS.items()):
        col.button(
            label, use_container_width=True, key=f"scn-{key}",
            on_click=_load_scenario, args=(key,),
        )

    with st.form("check_form"):
        r1 = st.columns(4)
        r1[0].selectbox("Application", [a.value for a in Application], key="f_application")
        r1[1].selectbox("User type", [u.value for u in UserType], key="f_user_type")
        r1[2].selectbox("Model", [m.value for m in ModelName], key="f_model")
        r1[3].text_input("Session ID", key="f_session_id")

        st.text_area("Prompt", key="f_prompt", height=68)
        st.text_area("Context / evidence", key="f_context", height=110)
        st.text_area("AI response", key="f_response", height=140)

        r2 = st.columns(4)
        r2[0].number_input("Input tokens", 0, 100000, key="f_tokens_in")
        r2[1].number_input("Output tokens", 0, 100000, key="f_tokens_out")
        r2[2].number_input("Latency (ms)", 1.0, 120000.0, key="f_latency_ms")
        r2[3].number_input("Retries", 0, 100, key="f_retry_count")

        r3 = st.columns(4)
        r3[0].number_input("Tool calls", 0, 100, key="f_tool_calls")
        r3[1].selectbox("Action type", [a.value for a in ActionType], key="f_action_type")
        r3[2].number_input("Action amount (₹)", 0.0, 100_000_000.0, key="f_action_amount_inr")
        r3[3].number_input("Affected entities", 1, 1_000_000, key="f_affected_entities")

        submitted = st.form_submit_button(
            "CHECK AI RESPONSE", type="primary", use_container_width=True
        )

    form = {k: st.session_state[f"f_{k}"] for k in DEFAULT_FORM}
    st.session_state["form"] = form  # shared with policy-sim / counterfactual tabs

    if submitted:
        if not str(form["response"]).strip():
            st.warning("Enter an AI response to check.")
        else:
            interaction = build_interaction(form)
            trace = svc.check(interaction, timestamp=interaction.timestamp)
            st.session_state["result_trace_id"] = interaction.interaction_id
            st.session_state["result_decision"] = trace.final_decision.decision.value
            st.session_state["result_trace"] = trace

    if "result_trace" in st.session_state:
        st.divider()
        render_trace(st.session_state["result_trace"])
        st.caption(f"interaction_id: `{st.session_state['result_trace_id']}` — "
                   "use it in the Audit / Feedback / Counterfactual tabs.")


# ------------------------------------------------------------------ POLICY SIMULATION tab

with page[1]:
    st.subheader("Policy simulation — why per-application governance exists")
    st.caption(
        "Run the *same* interaction through every application policy profile. "
        "Stricter profiles (e.g. decision-support) escalate the same risk further."
    )
    if "result_trace" not in st.session_state:
        st.info("Run a check first (or load a scenario), then simulate it here.")
    else:
        base_form = st.session_state.get("form", DEFAULT_FORM)
        profiles = st.multiselect(
            "Policy profiles",
            [a.value for a in Application],
            default=[a.value for a in Application],
        )
        if st.button("Simulate", type="primary") and profiles:
            interaction = build_interaction(base_form, interaction_id="INT-UI-SIM")
            from simulation.engine import simulate_policies

            sim = simulate_policies(svc.engine, interaction, profiles)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "profile": o.profile,
                            "decision": o.decision,
                            "overall risk": o.overall_risk,
                            "confidence": o.decision_confidence,
                            "human review": o.requires_human_review,
                            "reason codes": ", ".join(o.reason_codes),
                        }
                        for o in sim.outcomes
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
            (st.success if sim.differs else st.info)(sim.summary)


# ------------------------------------------------------------------ COUNTERFACTUAL tab

with page[2]:
    st.subheader("Counterfactual — “what if?”")
    st.caption(
        "Re-run the last checked interaction with a few production-visible fields "
        "changed. Ground-truth fields can never be touched."
    )
    if "result_trace" not in st.session_state:
        st.info("Run a check first, then explore counterfactuals here.")
    else:
        base_form = dict(st.session_state.get("form", DEFAULT_FORM))
        cc = st.columns(3)
        new_amount = cc[0].number_input(
            "action_amount_inr", 0.0, 100_000_000.0, float(base_form["action_amount_inr"])
        )
        new_action = cc[1].selectbox(
            "action_type", [a.value for a in ActionType],
            index=[a.value for a in ActionType].index(base_form["action_type"]),
        )
        new_entities = cc[2].number_input(
            "affected_entities", 1, 1_000_000, int(base_form["affected_entities"])
        )
        if st.button("Compare", type="primary"):
            interaction = build_interaction(base_form, interaction_id="INT-UI-CF")
            from simulation.engine import compare_decisions

            modified = {}
            if new_amount != base_form["action_amount_inr"]:
                modified["action_amount_inr"] = new_amount
            if new_action != base_form["action_type"]:
                modified["action_type"] = new_action
            if new_entities != base_form["affected_entities"]:
                modified["affected_entities"] = new_entities
            if not modified:
                st.warning("Change at least one field.")
            else:
                result = compare_decisions(svc.engine, interaction, modified)
                a, b = st.columns(2)
                a.metric("Original", result.original_decision,
                         help=f"risk {result.original_overall_risk:.2f}")
                b.metric("Counterfactual", result.counterfactual_decision,
                         help=f"risk {result.counterfactual_overall_risk:.2f}")
                st.info(result.summary)
                if result.rules_removed:
                    st.markdown("**Rules that stopped firing:** " +
                                ", ".join(f"`{r}`" for r in result.rules_removed))
                if result.rules_added:
                    st.markdown("**Rules that started firing:** " +
                                ", ".join(f"`{r}`" for r in result.rules_added))
                if result.reason_codes_removed:
                    st.markdown("**Reason codes removed:** " +
                                ", ".join(result.reason_codes_removed))


# ------------------------------------------------------------------ MULTI-TURN tab

with page[3]:
    st.subheader("Multi-turn session risk accumulation")
    st.caption(
        "Repeatedly check responses under one session ID. Borderline turns "
        "compound into a bounded, decaying session risk that can escalate the tier."
    )
    sid = st.text_input("Session ID", "SESSION-DEMO-MT", key="mt_sid")

    cc = st.columns(3)
    if cc[0].button("Add a borderline (hallucination) turn", use_container_width=True):
        turn = scenarios.scenario_b_hallucination().model_copy(
            update={"session_id": sid, "interaction_id": f"INT-MT-{datetime.now(timezone.utc).strftime('%H%M%S%f')}"}
        )
        svc.check(turn, timestamp=datetime.now(timezone.utc))
    if cc[1].button("Add a clean turn", use_container_width=True):
        turn = scenarios.scenario_a_clean().model_copy(
            update={"session_id": sid, "interaction_id": f"INT-MT-{datetime.now(timezone.utc).strftime('%H%M%S%f')}"}
        )
        svc.check(turn, timestamp=datetime.now(timezone.utc))
    if cc[2].button("Reset session", use_container_width=True):
        svc.reset_session(sid)

    state = svc.get_session(sid)
    m = st.columns(4)
    m[0].metric("Turns", state["interaction_count"])
    m[1].metric("High-risk turns", state["high_risk_event_count"])
    m[2].metric("Cumulative risk", f"{state['cumulative_risk']:.2f}")
    m[3].metric("Escalated", "YES" if state["escalated"] else "no")

    if state["recent_decisions"]:
        st.dataframe(
            pd.DataFrame(
                {
                    "turn": list(range(1, len(state["recent_decisions"]) + 1)),
                    "decision": state["recent_decisions"],
                    "risk": state["recent_risks"],
                    "interaction_id": state["recent_interaction_ids"],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No turns recorded for this session yet.")


# ------------------------------------------------------------------ AUDIT tab

with page[4]:
    st.subheader("Decision audit trail")
    default_id = st.session_state.get("result_trace_id", "")
    aid = st.text_input("Interaction ID", default_id, key="audit_id")
    if st.button("Look up", key="audit_lookup") or aid:
        summary = svc.get_audit(aid) if aid else None
        if summary is None:
            st.info("No audit record for that interaction ID (run a check first).")
        else:
            st.markdown(f"### {summary['decision']} · overall risk {summary['overall_risk']:.2f}")
            a = st.columns(4)
            a[0].metric("Performance", f"{summary['performance_risk']:.2f}")
            a[1].metric("Responsibility", f"{summary['responsibility_risk']:.2f}")
            a[2].metric("Cost", f"{summary['cost_risk']:.2f}")
            a[3].metric("Consequence", f"{summary['consequence_score']:.2f}")
            st.write(f"**Timestamp:** {summary['timestamp']}  ·  "
                     f"**Application:** {summary['application']}  ·  "
                     f"**Action:** {summary['action_type']}")
            st.write(f"**Triggered rules:** {', '.join(summary['triggered_rules']) or '(none)'}")
            st.info(summary["explanation"])
            if summary["responsibility_findings"]:
                st.markdown("**Responsibility findings (redacted):**")
                st.dataframe(pd.DataFrame(summary["responsibility_findings"]), hide_index=True)
            if summary["cost_anomalies"]:
                st.markdown(f"**Cost anomalies:** {', '.join(summary['cost_anomalies'])}")


# ------------------------------------------------------------------ FEEDBACK tab

with page[5]:
    st.subheader("Reviewer feedback")
    if "result_decision" not in st.session_state:
        st.info("Run a check first, then leave feedback on that decision.")
    else:
        iid = st.session_state["result_trace_id"]
        sysd = st.session_state["result_decision"]
        st.write(f"Interaction `{iid}` — system decision **{sysd}**")
        outcome = st.radio("Was this decision correct?", ["approved", "modified", "rejected"], horizontal=True)
        reviewer_tier = None
        if outcome != "approved":
            reviewer_tier = st.selectbox(
                "What should it have been?", [t.value for t in InterventionTier]
            )
        reason = st.text_input("Reviewer reason (optional)")
        reviewer = st.text_input("Reviewer", "demo-reviewer")
        if st.button("Submit feedback", type="primary"):
            record = svc.submit_feedback(
                interaction_id=iid,
                system_decision=None,
                reviewer_decision=reviewer_tier,
                outcome=outcome,
                reason=reason,
                reviewer=reviewer,
            )
            st.success(f"Recorded {record.feedback_id} — outcome `{record.outcome.value}`, "
                       f"override={record.human_override}")

    agg = svc.feedback_summary()
    if agg["total"]:
        st.divider()
        st.markdown("#### Feedback summary")
        f = st.columns(4)
        f[0].metric("Total", agg["total"])
        f[1].metric("Approval rate", f"{agg['approval_rate']:.0%}")
        f[2].metric("Override rate", f"{agg['override_rate']:.0%}")
        f[3].metric("Escalations / de-escalations", f"{agg['escalations']} / {agg['de_escalations']}")


# ------------------------------------------------------------------ COMMAND CENTER tab

with page[6]:
    _cc_tools = st.columns([2, 1, 3])
    if _cc_tools[0].button(
        "Populate demo operational traffic", type="primary", key="cc_populate"
    ):
        _n = svc.populate_operational_demo(250)
        st.success(f"Recorded {_n} operational interactions.")
    _cc_tools[1].button("Refresh", key="cc_refresh")

    cc_report = svc.get_operational_monitoring()
    st.session_state["cc_report"] = cc_report
    render_command_center(cc_report, svc)

# ------------------------------------------------------------------ GOVERNANCE tab

with page[7]:
    _g_tools = st.columns([3, 2, 3])
    if _g_tools[0].button(
        "Populate demo operational traffic", type="primary", key="gov_populate"
    ):
        _n = svc.populate_operational_demo(250)
        st.success(f"Recorded {_n} operational interactions.")
    _g_calib = _g_tools[1].checkbox(
        "Run calibration bridge (slower)", key="gov_calibration"
    )
    _g_tools[2].button("Refresh", key="gov_refresh")

    gov_report = svc.governance_report(with_calibration=bool(_g_calib))
    st.session_state["gov_report"] = gov_report
    render_governance(gov_report, svc)

# ------------------------------------------------------------------ ADAPTIVE tab

with page[8]:
    _a_tools = st.columns([3, 2, 3])
    if _a_tools[0].button(
        "Populate demo operational traffic", type="primary", key="ad_populate"
    ):
        _n = svc.populate_operational_demo(250)
        st.success(f"Recorded {_n} operational interactions.")
    _a_cf = _a_tools[1].checkbox(
        "Run counterfactual bridge (slower)", key="ad_counterfactual"
    )
    _a_tools[2].button("Refresh", key="ad_refresh")

    adaptive_report = svc.adaptive_report(with_counterfactual=bool(_a_cf))
    st.session_state["adaptive_report"] = adaptive_report
    render_adaptive(adaptive_report, svc)

# ------------------------------------------------------------------ EXECUTIVE tab

with page[9]:
    _e_tools = st.columns([3, 5])
    if _e_tools[0].button(
        "Populate demo operational traffic", type="primary", key="exec_populate_top"
    ):
        _n = svc.populate_operational_demo(250)
        st.success(f"Recorded {_n} operational interactions.")
    _cc_view = svc.command_center_view()
    st.session_state["command_center_view"] = _cc_view
    render_executive(_cc_view, svc)

st.divider()
st.caption(
    f"ControlPlane.ai v{VERSION} · deterministic heuristic system · "
    "synthetic data · not a production safety guarantee."
)
