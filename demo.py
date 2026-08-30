"""
ControlPlane.ai — end-to-end demonstration.

    python demo.py                 # scenarios A-G
    python demo.py --evaluation    # + evaluation report (150 synthetic cases)
    python demo.py --ablation      # + ablation experiment
    python demo.py --all           # everything
    python demo.py --all --save demo_output.txt

Every number below is produced by the real pipeline at runtime. Nothing
is hard-coded.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from datetime import datetime

# The demo prints ₹ / unicode; make Windows consoles (cp1252) behave.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8")

from decision.engine import DecisionEngine
from evaluation.evaluation import build_engine
from session.manager import SessionManager
from simulation.engine import compare_decisions, simulate_policies
from tests import scenarios

_TS = datetime(2026, 8, 21, 12, 0, 0)
_RULE = "=" * 72
_SUB = "-" * 72

# name -> (letter, headline, what it demonstrates)
_SCENARIO_META = {
    "A_clean": ("A", "CLEAN RESPONSE", "Context fully supports the answer."),
    "B_hallucination": ("B", "HALLUCINATION", "Answer contradicts the supplied policy."),
    "C_pii": ("C", "PII LEAKAGE", "Response exposes synthetic customer contact data."),
    "D_high_consequence": (
        "D",
        "HIGH-CONSEQUENCE ACTION",
        "Plausible answer, but a large automated financial action.",
    ),
    "E_multi_risk": (
        "E",
        "MULTI-RISK",
        "Unsupported claim + PII + irreversible consequential action.",
    ),
    "F_cost_anomaly": (
        "F",
        "COST ANOMALY",
        "Huge token usage + retries + tool calls, content is fine.",
    ),
    "H_low_confidence": (
        "H",
        "HIGH STAKES, LOW CONFIDENCE",
        "Unverifiable consequential claim - not ALLOWed, not BLOCKed.",
    ),
}


def _risk_findings(trace) -> list[str]:
    lines: list[str] = []
    perf = trace.performance
    lines.append(
        f"  Performance    : status={perf.status.value}, "
        f"{len(perf.claim_results)} claim(s), "
        f"grounding={perf.grounding_score if perf.grounding_score is not None else 'n/a'}, "
        f"confidence={perf.confidence:.2f}"
    )
    resp = trace.responsibility
    if resp.findings:
        by_cat: dict[str, int] = {}
        for f in resp.findings:
            by_cat[f.category.value] = by_cat.get(f.category.value, 0) + 1
        detail = ", ".join(f"{k.lower()} x{v}" for k, v in by_cat.items())
        crit = " [CRITICAL PII]" if resp.contains_critical_pii else ""
        lines.append(f"  Responsibility : {detail}{crit}")
    else:
        lines.append("  Responsibility : no findings")
    cost = trace.cost
    if cost.triggered_dimensions:
        lines.append(
            f"  Cost           : ₹{cost.estimated_cost_inr:.4f}, "
            f"anomalies: {', '.join(cost.triggered_dimensions)}"
        )
    else:
        lines.append(f"  Cost           : ₹{cost.estimated_cost_inr:.4f}, no anomalies")
    cons = trace.consequence
    top = ", ".join(cons.dominant_factors) or "none"
    lines.append(
        f"  Consequence    : score={cons.consequence_score:.2f} ({cons.severity_band}), "
        f"drivers: {top}"
    )
    return lines


def _print_decision(letter: str, headline: str, note: str, trace) -> None:
    fd = trace.final_decision
    print(f"\n{_SUB}\nSCENARIO {letter} — {headline}\n{_SUB}")
    print(f"  {note}")
    print(f"  Application          : {trace.application}")
    print()
    print("  RISK FINDINGS")
    for line in _risk_findings(trace):
        print(line)
    print()
    print(f"  Performance Risk     : {fd.performance_risk:.2f}"
          f"   (criticality-weighted from {trace.performance.performance_risk:.2f})")
    print(f"  Responsibility Risk  : {fd.responsibility_risk:.2f}")
    print(f"  Cost Risk            : {fd.cost_risk:.2f}   "
          f"efficiency {trace.cost.cost_efficiency_score:.2f}")
    print(f"  Consequence Score    : {trace.consequence.consequence_score:.2f}")
    print(f"  Action Criticality   : {trace.criticality.action_criticality:.2f} "
          f"({trace.criticality.band})")
    print(f"  Overall RISK         : {fd.overall_risk:.2f}   "
          f"(pre-session {trace.pre_session_overall_risk:.2f})")
    print(f"  Overall CONFIDENCE   : {fd.decision_confidence:.2f}   "
          f"(uncertainty {1 - fd.decision_confidence:.2f})")
    print()
    print(f"  DECISION            :  {fd.decision.value}")
    human = fd.decision.value in ("HUMAN_REVIEW", "BLOCK")
    print(f"  Human review        :  {'YES' if human else 'no'}")
    print()
    print("  REASON CODES")
    if fd.reason_codes:
        for code in fd.reason_codes:
            print(f"    - {code}")
    else:
        print("    (none)")
    print()
    print("  WHAT MOVED THE DECISION (policy drivers)")
    if trace.decision_drivers:
        for d in trace.decision_drivers:
            print(f"    - {d.rule}: {d.effect}")
    else:
        print(f"    - risk band alone ({fd.decision.value})")
    print()
    print("  EXPLANATION")
    print(f"    {fd.explanation}")
    print(f"\n  (pipeline latency: {trace.latency_ms:.2f} ms; "
          f"detectors run independently and can be parallelized in production)")


def run_scenarios() -> None:
    print(_RULE)
    print("CONTROLPLANE.AI — END-TO-END DEMO")
    print(_RULE)
    print("Real pipeline: detect -> assess -> consequence -> policy -> decision")

    engine = build_engine()  # cost baseline fitted from synthetic traffic
    for name, factory in scenarios.ALL_SINGLE_TURN.items():
        letter, headline, note = _SCENARIO_META[name]
        trace = engine.evaluate(factory(), timestamp=_TS, record_session=False)
        _print_decision(letter, headline, note, trace)

    print(f"\n{_SUB}\nSCENARIO G — MULTI-TURN SESSION ESCALATION\n{_SUB}")
    print("  One session; each borderline turn adds to a bounded, decaying risk memory.")
    session_manager = SessionManager()
    session_engine = DecisionEngine(session_manager=session_manager)
    for i, interaction in enumerate(scenarios.scenario_g_multi_turn(), start=1):
        trace = session_engine.evaluate(interaction, timestamp=_TS)
        s = trace.session or {}
        print(
            f"  turn {i}: decision={trace.final_decision.decision.value:12s} "
            f"raw_risk={trace.pre_session_overall_risk:.2f} "
            f"session_adjusted={trace.final_decision.overall_risk:.2f} "
            f"session_risk={s.get('session_risk', 0):.2f} "
            f"high_risk_turns={s.get('high_risk_events', 0)} "
            f"escalated={s.get('escalated')}"
        )
    state = session_manager.get_state("SESSION-SCEN-G")
    print(
        f"\n  final session: {state.interaction_count} turns, "
        f"cumulative_risk={state.cumulative_risk:.2f}, escalated={state.escalated}"
    )

    # --- Scenario I: policy counterfactual ---
    print(f"\n{_SUB}\nSCENARIO I - POLICY COUNTERFACTUAL\n{_SUB}")
    print("  The same contradicted response, governed by different applications:")
    interaction, profiles = scenarios.scenario_i_policy_counterfactual()
    sim = simulate_policies(engine, interaction, profiles)
    for o in sim.outcomes:
        print(f"    {o.profile:30s} -> {o.decision:12s} (risk {o.overall_risk:.2f})")
    print(f"  {sim.summary}")

    # --- Scenario J: consequence counterfactual ---
    print(f"\n{_SUB}\nSCENARIO J - CONSEQUENCE COUNTERFACTUAL\n{_SUB}")
    interaction, modified = scenarios.scenario_j_consequence_counterfactual()
    cf = compare_decisions(engine, interaction, modified)
    print(f"  Original (₹{interaction.action_amount_inr:,.0f}): {cf.original_decision}")
    print(f"  Counterfactual (₹100):        {cf.counterfactual_decision}")
    print(f"  Rules that stopped firing: {', '.join(cf.rules_removed) or '(none)'}")
    print(f"  Reason codes removed:      {', '.join(cf.reason_codes_removed) or '(none)'}")
    print(f"  {cf.summary}")


def run_investigation() -> None:
    """
    Command Center -> incident -> investigate -> replay -> explainability ->
    counterfactual -> human governance. Every value comes from a real trace;
    the automated decision is never mutated.
    """
    from api.service import ControlPlaneService

    print(f"\n\n{_RULE}\nCONTROLPLANE INCIDENT INVESTIGATION\n{_RULE}")

    svc = ControlPlaneService(fit_cost_baseline=False)
    pii = scenarios.scenario_c_pii()
    svc.check(pii, timestamp=pii.timestamp)
    high_cons = scenarios.scenario_d_high_consequence()
    svc.check(high_cons, timestamp=high_cons.timestamp)

    inv = svc.investigate_incident(pii.interaction_id)
    exp = inv.explanation
    inc = inv.incident

    print(f"\n  Incident   : {inv.interaction_id}")
    print(f"  Application: {inc.application}")
    print(f"  Severity   : {inc.severity.value}  ({inc.severity_rationale})")
    print(f"\n  DECISION       {exp.decision.value}")
    print(f"  RISK           {exp.overall_risk:.2f}")
    print(f"  CONFIDENCE     {exp.decision_confidence:.2f}")
    print(f"  VERIFICATION   {exp.verification_path.value}")
    print("\n  WHY")
    for code in exp.primary_reasons:
        print(f"    {code}")
    chain = [exp.decision_path[0].from_tier.value] + [
        s.to_tier.value for s in exp.decision_path
    ]
    print("\n  DECISION PATH")
    print("    " + "  ->  ".join(chain))

    print("\n  COUNTERFACTUAL (simulation — stored decision unchanged)")
    cf = svc.investigation_counterfactual(
        high_cons.interaction_id, {"action_amount_inr": 1.0}
    )
    print(
        f"    High-consequence incident {high_cons.interaction_id}: with "
        f"action_amount_inr -> ₹1, {cf.current_decision} -> "
        f"{cf.counterfactual_decision}"
    )
    if cf.rules_removed:
        print(f"    Rules that stop firing: {', '.join(cf.rules_removed)}")

    print("\n  GOVERNANCE")
    print(f"    status: {inv.investigation_status.value}")
    inv2 = svc.record_governance_action(
        pii.interaction_id, action="ACKNOWLEDGE", actor="demo-reviewer"
    )
    inv3 = svc.record_governance_action(
        pii.interaction_id,
        action="MODIFY_DECISION",
        actor="demo-reviewer",
        comment="Route to human review rather than a hard block.",
        reviewer_decision="HUMAN_REVIEW",
    )
    for act in inv3.governance_history:
        line = f"    {act.action.value} ({act.previous_status.value} -> {act.new_status.value})"
        if act.reviewer_decision:
            line += (
                f"  reviewer_decision={act.reviewer_decision} "
                f"(automated {act.original_decision} unchanged)"
            )
        print(line)

    print(f"\n{_SUB}")
    print(
        f"  Original ControlPlane decision remains immutable: "
        f"{svc.get_audit_trace(pii.interaction_id).final_decision.decision.value}."
    )
    print(f"{_SUB}")


def run_enterprise(with_counterfactual: bool = True) -> None:
    """Judge-ready enterprise transcript — every number from the real system."""
    from api.service import ControlPlaneService

    print(f"\n\n{_RULE}\nCONTROLPLANE.AI — ENTERPRISE DEMONSTRATION\n{_RULE}")

    svc = ControlPlaneService(fit_cost_baseline=True)
    result = svc.enterprise_demo(with_counterfactual=with_counterfactual)
    cc = svc.command_center_view()

    print("\nAI TRAFFIC")
    print(f"  Interactions: {result.interactions}")

    rp = cc.risk_posture
    print("\nRISK POSTURE")
    print(f"  Average risk           {rp.overall_average:.3f}")
    print(f"  High-risk interactions {rp.overall_high_risk_count}")
    print(f"  Dominant dimension     {rp.dominant_dimension}")
    print(f"  Performance / Responsibility / Cost avg: "
          f"{rp.performance_average:.3f} / {rp.responsibility_average:.3f} / {rp.cost_average:.3f}")

    print("\nAPPLICATIONS")
    for a in cc.application_posture:
        print(f"  {a.application:30s} posture {a.posture:8s} "
              f"risk {a.average_risk:.2f}  human-review {a.human_review_rate:.0%}  "
              f"DEEP {a.deep_rate:.0%}  incidents {a.incident_count}")

    print("\nINCIDENT INTELLIGENCE")
    print(f"  Incidents: {result.incidents}")
    print(f"  Top pattern: {result.top_pattern or '(none)'}")

    print("\nDRIFT")
    print(f"  Potential drift signals: {result.potential_drift}")

    print("\nGOVERNANCE")
    print(f"  Top recommendation: {result.top_recommendation or '(none — NO_ACTION)'}")

    print("\nCOUNTERFACTUAL")
    print(f"  Safety: {result.counterfactual_safety}")
    step8 = next((s for s in result.steps if s.step == 8), None)
    if step8 and step8.metrics.get("candidate_recall") is not None:
        m = step8.metrics
        print(f"    recall     {m['current_recall']:.2f} -> {m['candidate_recall']:.2f}")
        print(f"    FAST-path  {m['current_fast_rate']:.0%} -> {m['candidate_fast_rate']:.0%}")

    print("\nHUMAN GOVERNANCE")
    print(f"  Approval: {result.approval_status or '(nothing actionable)'}")

    print(f"\n{_SUB}")
    print(f"  PRODUCTION: {result.production_configuration_status}  "
          "(no config change; no deployment path; approval means APPROVED_FOR_EVALUATION only)")
    print(f"{_SUB}")

    print("\nWORKFLOW STEPS")
    for s in result.steps:
        print(f"  STEP {s.step} · {s.title}")
        print(f"    {s.detail}")


def run_adaptive() -> None:
    """
    Closed-loop adaptive guardrails (Phase 10): incident patterns -> drift ->
    attribution -> governance signals -> adaptive recommendation -> counterfactual
    simulation -> safety check -> HUMAN APPROVAL -> APPROVED_FOR_EVALUATION.
    Production configuration is never touched.
    """
    from datetime import timedelta

    from api.service import ControlPlaneService

    print(f"\n\n{_RULE}\nCONTROLPLANE ADAPTIVE GOVERNANCE\n{_RULE}")

    svc = ControlPlaneService(fit_cost_baseline=True)
    svc.populate_operational_demo(250)
    traces = svc.all_traces()

    # K — recurring reviewer disagreement: repeated BLOCK -> HUMAN_REVIEW
    for t in [t for t in traces if t.final_decision.decision.value == "BLOCK"][:4]:
        svc.record_governance_action(t.interaction_id, action="ACKNOWLEDGE")
        svc.record_governance_action(
            t.interaction_id, action="MODIFY_DECISION",
            comment="Route to human review rather than a hard block.",
            reviewer_decision="HUMAN_REVIEW",
        )
    # M — drift: a batch of higher-risk recent traffic (later timestamps)
    base_ts = max(t.timestamp for t in traces)
    for i, scn in enumerate(
        [scenarios.scenario_e_multi_risk, scenarios.scenario_c_pii,
         scenarios.scenario_b_hallucination] * 6
    ):
        it = scn().model_copy(
            update={
                "interaction_id": f"OPS-DRIFT-{i:02d}",
                "session_id": f"OPS-DRIFT-{i:02d}",
                "timestamp": base_ts + timedelta(minutes=i + 1),
            }
        )
        svc.check(it, timestamp=it.timestamp)

    intel = svc.incident_intelligence()
    print(f"\nPATTERNS ({len(intel.patterns)})")
    for p in intel.patterns[:5]:
        apps = ", ".join(p.applications) or "-"
        print(f"  [{p.severity.value}] {p.type.value} · {apps} · {p.incident_count} incidents "
              f"· detection_confidence {p.detection_confidence}")

    print(f"\nDRIFT ({intel.drift.potential_drift_count} POTENTIAL_DRIFT / "
          f"{len(intel.drift.signals)} signals)")
    for s in intel.drift.signals:
        if s.signal != "STABLE":
            print(f"  {s.scope} {s.metric}: {s.baseline:.3f} -> {s.recent:.3f} "
                  f"(delta {s.delta:+.3f}) [{s.signal}]")

    if intel.attributions:
        a = intel.attributions[0]
        print("\nATTRIBUTION (observed association, not causal proof)")
        print(f"  {a.narrative}")

    print("\nGOVERNANCE SIGNAL")
    for op in intel.reviewer_override_patterns:
        print(f"  {op.transition} · {op.application} · overrides: {op.count} "
              f"· reason codes: {', '.join(op.affected_reason_codes) or '-'}")
    print("  (reviewer disagreement is a governance signal, NOT ground truth)")

    print("\nRunning counterfactual bridge (calibration.sweep + calibration.select) …")
    report = svc.adaptive_report(with_counterfactual=True)

    print("\nRECOMMENDATIONS")
    for r in report.recommendations:
        line = f"  {r.type.value}" + (f" · {r.application}" if r.application else "")
        line += f" · {r.severity.value} · status {r.status.value}"
        print(line)
        if r.simulation_result is not None:
            sr = r.simulation_result
            print(f"    COUNTERFACTUAL  safety {'PASS' if sr.safety_passed else 'FAIL'}")
            if sr.candidate_recall is not None:
                print(f"      recall     {sr.current_recall:.2f} -> {sr.candidate_recall:.2f}")
                print(f"      precision  {sr.current_precision:.2f} -> {sr.candidate_precision:.2f}")
                print(f"      FAST       {sr.current_fast_rate:.0%} -> {sr.candidate_fast_rate:.0%}")
                print(f"      DEEP       {sr.current_deep_rate:.0%} -> {sr.candidate_deep_rate:.0%}")
            if not sr.safety_passed:
                print(f"      -> {sr.selection_reason.split('.')[0]}.")

    # Q — no-safe-candidate: strict safety constraints
    from adaptive.counterfactual import run_threshold_counterfactual
    strict = run_threshold_counterfactual(svc._config, minimum_recall=0.999, minimum_precision=0.99)
    print(f"\nQ · STRICT SAFETY CONSTRAINTS (recall>=0.999, precision>=0.99)")
    print(f"    calibration.select status: {strict.status}  "
          f"(eligible {strict.eligible_candidate_count}/{strict.total_candidate_count})")
    print("    -> no configuration change recommended; safety constraints NOT relaxed.")

    # R — human approval gate
    approvable = next(
        (r for r in report.recommendations if r.type.value != "NO_ACTION"), None
    )
    if approvable is not None:
        updated = svc.adaptive_approve(
            approvable.recommendation_id, actor="demo-reviewer",
            comment="Approved for evaluation only.",
        )
        print("\nR · HUMAN APPROVAL GATE")
        print(f"    {updated.recommendation_id} -> {updated.status.value}")
        print(f"    {updated.approval.disclaimer}")

    from settings import load_settings
    print(f"\n{_SUB}")
    print(f"  PRODUCTION CONFIGURATION: {report.production_configuration_status}  "
          f"(deep_verification_risk_threshold still "
          f"{load_settings()['verification']['deep_verification_risk_threshold']})")
    print("  Recommendations are RECOMMENDATION ONLY — approval means "
          "APPROVED_FOR_EVALUATION, never applied to production.")
    print(f"{_SUB}")


def run_governance() -> None:
    """
    Closed-loop governance intelligence: traffic -> decisions -> incidents ->
    human review -> governance signals -> analysis -> RECOMMENDATION (never
    applied). Every number comes from real pipeline traces + governance actions.
    """
    from api.service import ControlPlaneService
    from governance.report import build_governance_report

    print(f"\n\n{_RULE}\nCONTROLPLANE GOVERNANCE INTELLIGENCE\n{_RULE}")

    svc = ControlPlaneService(fit_cost_baseline=True)
    svc.populate_operational_demo(250)
    traces = svc.all_traces()

    # a handful of human governance actions on real incidents
    blocks = [t for t in traces if t.final_decision.decision.value == "BLOCK"][:3]
    hrs = [t for t in traces if t.final_decision.decision.value == "HUMAN_REVIEW"][:4]
    for t in blocks:
        svc.record_governance_action(t.interaction_id, action="ACKNOWLEDGE")
        svc.record_governance_action(
            t.interaction_id, action="MODIFY_DECISION",
            comment="Route to human review rather than a hard block.",
            reviewer_decision="HUMAN_REVIEW",
        )
    for t in hrs[:2]:
        svc.record_governance_action(t.interaction_id, action="APPROVE_DECISION", comment="agree")
    for t in hrs[2:]:
        svc.record_governance_action(t.interaction_id, action="ESCALATE", comment="senior review")

    rep = build_governance_report(
        traces, svc.governance.get_all_actions(), svc.feedback.all()
    )
    o = rep.overview
    d, r, v = o.decisions, o.risk, o.verification
    ac = rep.application_comparison

    print(f"\nTRAFFIC\n    {o.traffic.total_interactions} interactions across "
          f"{len(o.traffic.by_application)} applications")
    print("\nDECISIONS")
    for label, rate in (
        ("ALLOW", d.allow_rate), ("ANNOTATE", d.annotate_rate), ("VERIFY", d.verify_rate),
        ("HUMAN_REVIEW", d.human_review_rate), ("BLOCK", d.block_rate),
    ):
        print(f"    {label:14s} {rate * 100:5.1f}%")
    print("\nRISK")
    print(f"    Average        {r.average_risk:.3f}")
    print(f"    P95            {r.p95_risk:.3f}")
    print(f"    High-risk      {r.high_risk_rate * 100:.1f}%")
    print("\nVERIFICATION")
    print(f"    FAST           {v.fast_rate * 100:.1f}%")
    print(f"    DEEP           {v.deep_rate * 100:.1f}%")

    print("\nAPPLICATION HOTSPOTS")
    print(f"    highest risk     {ac.highest_risk}")
    print(f"    highest volume   {ac.highest_volume}")
    print(f"    lowest oversight {ac.lowest_intervention}")

    rd = o.reviewer_disagreement
    print("\nREVIEWER SIGNALS  (governance signal — NOT ground truth)")
    print(f"    reviewed          {rd.reviewed_count}")
    print(f"    overrides         {rd.override_count}")
    if rd.override_rate is not None:
        print(f"    override rate     {rd.override_rate * 100:.1f}%")
    if rd.automated_to_reviewer_transitions:
        top = next(iter(rd.automated_to_reviewer_transitions))
        print(f"    common transition {top}")

    print("\nTOP GOVERNANCE INSIGHT")
    if rep.insights:
        top = rep.insights[0]
        print(f"    [{top.severity.value}] {top.code}")
        print(f"    {top.explanation.split('.')[0]}.")
        print(f"    -> {top.recommended_action.value}")
    else:
        print("    (no governance threshold crossed)")

    print("\nRECOMMENDATION")
    top_rec = rep.recommendations[0]
    print(f"    {top_rec.recommendation_type.value}"
          + (f"   application: {top_rec.application}" if top_rec.application else ""))
    print(f"    {top_rec.rationale.split('.')[0]}.")
    print(f"    disposition: {top_rec.disposition.value}")
    print(f"\n{_SUB}")
    print("    RECOMMENDATION ONLY — NOT APPLIED. Production configuration is unchanged.")
    print(f"{_SUB}")


def run_evaluation() -> None:
    from evaluation.evaluation import evaluate

    print(f"\n\n{_RULE}\nEVALUATION REPORT (150 synthetic evaluation cases)\n{_RULE}")
    report = evaluate().model_dump()
    print("  detector                     P     R     F1    FPR")
    for key in (
        "performance_hallucination",
        "pii",
        "toxicity",
        "bias",
        "cost_anomaly",
        "responsibility_overall",
        "any_risk_catch",
    ):
        m = report[key]
        print(
            f"  {key:26s} {m['precision']:.2f}  {m['recall']:.2f}  "
            f"{m['f1']:.2f}  {m['false_positive_rate']:.2f}"
        )
    print(f"\n  abstention (UNVERIFIED) rate : {report['abstention_rate']:.2f}")
    print(f"  coverage rate                : {report['coverage_rate']:.2f}")
    print(f"  intervention distribution    : {report['intervention_distribution']}")
    print(f"  human-review rate            : {report['human_review_rate']:.2f}")
    print(f"  mean latency (ms)            : {report['mean_latency_ms']:.2f}")
    print("\n  notes:")
    for note in report["notes"]:
        print(f"    - {note}")


def run_ablation() -> None:
    from evaluation.ablation import run_ablation as _run

    print(f"\n\n{_RULE}\nABLATION — WHY THE ARCHITECTURE MATTERS\n{_RULE}")
    report = _run()
    print("  mode                     catch-F1   human-escalation   clean-escalation")
    for mode in report.modes:
        print(
            f"  {mode.mode:24s} {mode.caught_risky['f1']:.2f}       "
            f"{mode.human_escalation_rate:>5.0%}              "
            f"{mode.clean_human_escalation_rate:>5.0%}"
        )
    print(f"\n  {report.conclusion}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ControlPlane.ai end-to-end demo")
    parser.add_argument("--evaluation", action="store_true", help="add the evaluation report")
    parser.add_argument("--ablation", action="store_true", help="add the ablation experiment")
    parser.add_argument(
        "--investigation", action="store_true", help="add the incident-investigation example"
    )
    parser.add_argument(
        "--governance", action="store_true", help="add the governance-intelligence summary"
    )
    parser.add_argument(
        "--adaptive", action="store_true",
        help="add the closed-loop adaptive governance report (K-R; runs the calibration bridge)",
    )
    parser.add_argument(
        "--enterprise", action="store_true",
        help="judge-ready enterprise demonstration transcript (runs the counterfactual)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="scenarios + investigation + governance + adaptive + enterprise + evaluation + ablation",
    )
    parser.add_argument("--save", metavar="PATH", help="also write the full output to a file")
    args = parser.parse_args()

    buffer = io.StringIO()
    sink = buffer if args.save else sys.stdout

    with contextlib.redirect_stdout(sink):
        run_scenarios()
        if args.investigation or args.all:
            run_investigation()
        if args.governance or args.all:
            run_governance()
        if args.adaptive or args.all:
            run_adaptive()
        if args.enterprise:
            run_enterprise(with_counterfactual=True)
        elif args.all:
            run_enterprise(with_counterfactual=False)
        if args.evaluation or args.all:
            run_evaluation()
        if args.ablation or args.all:
            run_ablation()
        print()

    if args.save:
        text = buffer.getvalue()
        sys.stdout.write(text)
        with open(args.save, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"[demo output written to {args.save}]")


if __name__ == "__main__":
    main()
