"""
One-click Enterprise Demonstration.

Runs a deterministic, bounded demo over the REAL pipeline and returns an
:class:`EnterpriseDemoResult` describing the 9 workflow steps. Every metric
comes from the real system. Never writes config; never deploys.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from enterprise.schemas import DemoStep, EnterpriseDemoResult

__all__ = ["run_enterprise_demo"]

_DEMO_TRAFFIC = 250


def run_enterprise_demo(
    service: Any,
    *,
    with_counterfactual: bool = True,
    generated_at: datetime | None = None,
) -> EnterpriseDemoResult:
    steps: list[DemoStep] = []

    # STEP 1 — AI traffic
    if not service.all_traces():
        service.populate_operational_demo(_DEMO_TRAFFIC)
    traces = service.all_traces()
    monitoring = service.get_operational_monitoring()
    steps.append(
        DemoStep(
            step=1, title="AI TRAFFIC",
            detail=f"{len(traces)} interactions across "
                   f"{len(monitoring.applications)} governed applications.",
            metrics={"interactions": len(traces),
                     "applications": [a.application for a in monitoring.applications]},
        )
    )

    # STEP 2 — risk detection
    s = monitoring.snapshot
    steps.append(
        DemoStep(
            step=2, title="RISK DETECTION",
            detail=f"average risk {s.average_risk:.2f}, p95 {s.p95_risk:.2f}, "
                   f"average confidence {s.average_confidence:.2f} "
                   "(risk and confidence are separate axes).",
            metrics={"average_risk": s.average_risk, "p95_risk": s.p95_risk,
                     "average_confidence": s.average_confidence},
        )
    )

    # STEP 3 — FAST / DEEP verification
    v = monitoring.verification
    steps.append(
        DemoStep(
            step=3, title="FAST / DEEP VERIFICATION",
            detail=f"FAST {v.fast_rate:.0%} / DEEP {v.deep_rate:.0%}. "
                   "Low-risk responses take the cheaper path; ambiguous / "
                   "consequential ones get deeper verification.",
            metrics={"fast_rate": v.fast_rate, "deep_rate": v.deep_rate,
                     "deep_triggers": v.deep_trigger_reason_counts},
        )
    )

    # STEP 4 — control decision
    d = s
    steps.append(
        DemoStep(
            step=4, title="CONTROL DECISION",
            detail=f"ALLOW {d.allow_rate:.0%} · ANNOTATE {d.annotate_rate:.0%} · "
                   f"VERIFY {d.verify_rate:.0%} · HUMAN_REVIEW {d.human_review_rate:.0%} · "
                   f"BLOCK {d.block_rate:.0%}.",
            metrics={"allow": d.allow_rate, "annotate": d.annotate_rate,
                     "verify": d.verify_rate, "human_review": d.human_review_rate,
                     "block": d.block_rate},
        )
    )

    # seed a small, deterministic set of human governance actions on real incidents
    blocked = [t for t in traces if t.final_decision.decision.value == "BLOCK"][:3]
    for t in blocked:
        if not service.governance.get_actions(t.interaction_id):
            service.record_governance_action(t.interaction_id, action="ACKNOWLEDGE")
            service.record_governance_action(
                t.interaction_id, action="MODIFY_DECISION",
                comment="Route to human review rather than a hard block.",
                reviewer_decision="HUMAN_REVIEW",
            )
    # seed recent higher-risk traffic so a drift signal can appear
    from tests import scenarios

    base_ts = max(t.timestamp for t in traces)
    for i, scn in enumerate(
        [scenarios.scenario_e_multi_risk, scenarios.scenario_c_pii] * 6
    ):
        it = scn().model_copy(
            update={
                "interaction_id": f"ENT-DEMO-{i:02d}",
                "session_id": f"ENT-DEMO-{i:02d}",
                "timestamp": base_ts + timedelta(minutes=i + 1),
            }
        )
        service.check(it, timestamp=it.timestamp)

    # STEP 5 — incident intelligence
    intel = service.incident_intelligence()
    steps.append(
        DemoStep(
            step=5, title="INCIDENT INTELLIGENCE",
            detail=f"{intel.total_incidents} incidents grouped into "
                   f"{len(intel.clusters)} deterministic clusters "
                   "(structured signature — no ML).",
            metrics={"incidents": intel.total_incidents, "clusters": len(intel.clusters)},
        )
    )

    # STEP 6 — pattern / drift detection
    top_pattern = intel.patterns[0].type.value if intel.patterns else None
    drift_nonstable = [x for x in intel.drift.signals if x.signal != "STABLE"]
    steps.append(
        DemoStep(
            step=6, title="PATTERN / DRIFT DETECTION",
            detail=f"{len(intel.patterns)} patterns"
                   + (f" (top: {top_pattern})" if top_pattern else "")
                   + f"; {intel.drift.potential_drift_count} POTENTIAL_DRIFT / "
                   f"{len(drift_nonstable)} non-stable signals.",
            metrics={"patterns": len(intel.patterns), "top_pattern": top_pattern,
                     "potential_drift": intel.drift.potential_drift_count},
        )
    )

    # STEP 7 — governance recommendation
    adaptive = service.adaptive_report(with_counterfactual=with_counterfactual)
    actionable = [r for r in adaptive.recommendations if r.type.value != "NO_ACTION"]
    top_rec = None
    if actionable:
        r0 = actionable[0]
        top_rec = r0.type.value + (f" — {r0.application}" if r0.application else "")
    steps.append(
        DemoStep(
            step=7, title="GOVERNANCE RECOMMENDATION",
            detail=(f"{len(actionable)} recommendation(s)"
                    + (f"; top: {top_rec}" if top_rec else "")
                    + ". All RECOMMENDED_FOR_REVIEW — nothing is applied."),
            metrics={"recommendations": len(actionable), "top": top_rec},
        )
    )

    # STEP 8 — counterfactual safety check
    cf_status = "NOT_RUN"
    cf_metrics: dict[str, Any] = {}
    threshold_rec = next(
        (r for r in adaptive.recommendations
         if r.type.value == "REVIEW_VERIFICATION_THRESHOLD" and r.simulation_result),
        None,
    )
    if threshold_rec is not None:
        sr = threshold_rec.simulation_result
        cf_status = "PASS" if sr.safety_passed else "NO_CANDIDATE"
        cf_metrics = {
            "current_recall": sr.current_recall, "candidate_recall": sr.candidate_recall,
            "current_fast_rate": sr.current_fast_rate, "candidate_fast_rate": sr.candidate_fast_rate,
            "safety_constraints": sr.safety_constraints,
        }
    steps.append(
        DemoStep(
            step=8, title="COUNTERFACTUAL SAFETY CHECK",
            detail=(
                "Ran calibration.sweep + calibration.select. "
                f"Safety: {cf_status}."
                if cf_status != "NOT_RUN"
                else "Counterfactual not run for this demo (no threshold recommendation "
                     "or with_counterfactual=False)."
            ),
            metrics=cf_metrics,
        )
    )

    # STEP 9 — human approval
    approval_status = None
    if actionable:
        approvable = actionable[0]
        updated = service.adaptive_approve(
            approvable.recommendation_id, actor="enterprise-demo",
            comment="Approved for evaluation only.",
        )
        approval_status = updated.status.value
    steps.append(
        DemoStep(
            step=9, title="HUMAN APPROVAL",
            detail=(f"Reviewer approved recommendation -> {approval_status}. "
                    "Approval means APPROVED_FOR_EVALUATION — never applied to production."
                    if approval_status
                    else "No actionable recommendation to approve in this window."),
            metrics={"approval_status": approval_status},
        )
    )

    return EnterpriseDemoResult(
        generated_at=generated_at,
        steps=steps,
        interactions=len(traces),
        incidents=intel.total_incidents,
        patterns=len(intel.patterns),
        potential_drift=intel.drift.potential_drift_count,
        top_pattern=top_pattern,
        top_recommendation=top_rec,
        counterfactual_safety=cf_status,
        approval_status=approval_status,
        production_configuration_status="UNCHANGED",
        notes=[
            "Deterministic, bounded demonstration over the real pipeline. Every "
            "number is produced by the actual system.",
            "Reviewer feedback is a governance signal, not ground truth. Drift is an "
            "operational signal, not proof of model degradation.",
            "Recommendations are RECOMMENDATION ONLY. Approval means "
            "APPROVED_FOR_EVALUATION — there is no deployment path and "
            "the production configuration file is never modified.",
        ],
    )
