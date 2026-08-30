"""
Governance audit timeline — built from stored objects.

DECISION -> INCIDENT -> REVIEWER_FEEDBACK -> PATTERN -> DRIFT ->
RECOMMENDATION -> COUNTERFACTUAL -> APPROVAL

Where a chronological timestamp is unavailable the event is shown in
causal workflow order only (never fabricated).
"""

from __future__ import annotations

from typing import Any

from decision.schemas import DecisionTrace
from enterprise.schemas import GovernanceTimeline, TimelineEvent

__all__ = ["build_governance_timeline"]

_NO_TS = "Chronological timestamp unavailable; displayed in causal workflow order."


def build_governance_timeline(
    traces: list[DecisionTrace],
    governance_actions: list[Any],
    feedback_records: list[Any],
    incident_intel: Any,
    adaptive_report: Any,
    *,
    focus_interaction_id: str | None = None,
) -> GovernanceTimeline:
    events: list[TimelineEvent] = []
    order = 0

    def _add(event_type, timestamp, entity, description):
        nonlocal order
        order += 1
        events.append(
            TimelineEvent(
                order=order,
                event_type=event_type,
                timestamp=timestamp,
                entity=entity,
                description=description,
                timestamp_note="" if timestamp is not None else _NO_TS,
            )
        )

    # 1 — a representative DECISION (the focus incident, or the highest-risk trace)
    focus = None
    if focus_interaction_id:
        focus = next((t for t in traces if t.interaction_id == focus_interaction_id), None)
    if focus is None and traces:
        focus = max(traces, key=lambda t: t.final_decision.overall_risk)
    if focus is not None:
        fd = focus.final_decision
        _add(
            "DECISION", focus.timestamp, focus.interaction_id,
            f"{focus.application}: ControlPlane decided {fd.decision.value} "
            f"(risk {fd.overall_risk:.2f}, {(focus.verification_path or 'DEEP').upper()} verification).",
        )
        # 2 — INCIDENT (if it is one)
        inc = next(
            (i for i in incident_intel.incidents if i.interaction_id == focus.interaction_id),
            None,
        )
        if inc is not None:
            _add(
                "INCIDENT", inc.timestamp, inc.incident_id,
                f"Recorded as a {inc.incident_severity} incident "
                f"(triggers: {', '.join(inc.incident_triggers)}).",
            )

    # 3 — REVIEWER_FEEDBACK
    for act in sorted(governance_actions, key=lambda a: a.action_id):
        if act.action.value in ("MODIFY_DECISION", "REJECT_DECISION"):
            rv = act.reviewer_decision or "REJECTED"
            _add(
                "REVIEWER_FEEDBACK", act.timestamp, act.interaction_id,
                f"Reviewer recorded {act.action.value}: {act.original_decision} -> {rv} "
                "(governance signal, not proof the automated decision was wrong).",
            )
            break
    for rec in sorted(feedback_records, key=lambda r: r.feedback_id):
        outcome = rec.outcome.value if hasattr(rec.outcome, "value") else str(rec.outcome)
        if outcome in ("modified", "rejected"):
            _add(
                "REVIEWER_FEEDBACK", getattr(rec, "timestamp", None), rec.interaction_id,
                f"Reviewer feedback: {outcome} (governance signal).",
            )
            break

    # 4 — PATTERN
    if incident_intel.patterns:
        p = incident_intel.patterns[0]
        _add(
            "PATTERN", incident_intel.generated_at, p.pattern_id,
            f"{p.type.value} detected across {', '.join(p.applications) or 'multiple apps'} "
            f"({p.incident_count} incidents, detection confidence {p.detection_confidence}).",
        )
    if incident_intel.reviewer_override_patterns:
        op = incident_intel.reviewer_override_patterns[0]
        _add(
            "PATTERN", None, op.pattern_id,
            f"Reviewer-override pattern {op.transition} in {op.application} "
            f"({op.count} occurrences).",
        )

    # 5 — DRIFT
    drift_nonstable = [s for s in incident_intel.drift.signals if s.signal != "STABLE"]
    if drift_nonstable:
        s = drift_nonstable[0]
        desc = (
            f"{s.scope} {s.metric}: {s.baseline:.3f} -> {s.recent:.3f} ({s.signal}). "
            "Operational drift signal, not proof of model degradation."
            if s.baseline is not None and s.recent is not None
            else f"{s.scope} {s.metric}: {s.signal}."
        )
        _add("DRIFT", None, s.scope, desc)

    # 6 — RECOMMENDATION
    actionable = [r for r in adaptive_report.recommendations if r.type.value != "NO_ACTION"]
    if actionable:
        r0 = actionable[0]
        _add(
            "RECOMMENDATION", adaptive_report.generated_at, r0.recommendation_id,
            f"{r0.type.value}" + (f" for {r0.application}" if r0.application else "")
            + f" ({r0.status.value}).",
        )
        # 7 — COUNTERFACTUAL
        if r0.simulation_result is not None:
            sr = r0.simulation_result
            _add(
                "COUNTERFACTUAL", None, r0.recommendation_id,
                f"Simulated (calibration.sweep + calibration.select). Safety: "
                f"{'PASS' if sr.safety_passed else 'no safe candidate'}.",
            )
        # 8 — APPROVAL
        if r0.approval is not None:
            _add(
                "APPROVAL", None, r0.recommendation_id,
                f"{r0.approval.decision} by {r0.approval.actor}. "
                "Production configuration remains UNCHANGED.",
            )

    return GovernanceTimeline(events=events)
