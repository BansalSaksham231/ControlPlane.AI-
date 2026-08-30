"""Assemble the :class:`AdaptiveGovernanceReport`."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from adaptive.recommendations import build_adaptive_recommendations
from adaptive.schemas import (
    AdaptiveConfig,
    AdaptiveGovernanceReport,
    RecommendationStatus,
)
from incident.schemas import IncidentIntelligenceReport

__all__ = ["build_adaptive_governance_report"]


def _observation(intelligence: IncidentIntelligenceReport) -> str:
    drift = [
        s
        for s in intelligence.drift.signals
        if s.signal == "POTENTIAL_DRIFT"
    ] or [s for s in intelligence.drift.signals if s.signal == "TREND"]
    if drift:
        s = drift[0]
        return (
            f"{s.scope} {s.metric} moved {s.baseline:.3f} -> {s.recent:.3f} "
            f"({s.signal})"
            if s.baseline is not None and s.recent is not None
            else f"{s.scope} {s.metric}: {s.signal}"
        )
    if intelligence.patterns:
        p = intelligence.patterns[0]
        apps = ", ".join(p.applications) or "multiple applications"
        return f"{apps}: {p.type.value} ({p.incident_count} incidents)"
    return "No recurring pattern or drift signal in the current window."


def build_adaptive_governance_report(
    intelligence: IncidentIntelligenceReport,
    *,
    config: AdaptiveConfig | None = None,
    calibration_selection: Any | None = None,
    approval_store: Any | None = None,
    generated_at: datetime | None = None,
) -> AdaptiveGovernanceReport:
    config = config or AdaptiveConfig()

    recommendations = build_adaptive_recommendations(
        intelligence,
        config=config,
        calibration_selection=calibration_selection,
        approval_store=approval_store,
    )

    non_stable = [s for s in intelligence.drift.signals if s.signal != "STABLE"]

    return AdaptiveGovernanceReport(
        generated_at=generated_at,
        config=config,
        observation=_observation(intelligence),
        pattern_count=len(intelligence.patterns),
        drift_signal_count=len(non_stable),
        potential_drift_count=intelligence.drift.potential_drift_count,
        reviewer_override_count=sum(
            op.count for op in intelligence.reviewer_override_patterns
        ),
        top_patterns=[
            {
                "pattern_id": p.pattern_id,
                "type": p.type.value,
                "severity": p.severity.value,
                "applications": p.applications,
                "incident_count": p.incident_count,
                "detection_confidence": p.detection_confidence,
                "affected_policy_rule": p.affected_policy_rule,
                "affected_dimension": p.affected_dimension,
                "recommended_next_step": p.recommended_next_step,
            }
            for p in intelligence.patterns[:8]
        ],
        drift_signals=[
            {
                "metric": s.metric,
                "scope": s.scope,
                "baseline": s.baseline,
                "recent": s.recent,
                "delta": s.delta,
                "direction": s.direction.value,
                "signal": s.signal,
                "sample_sufficient": s.sample_sufficient,
                "explanation": s.explanation,
            }
            for s in non_stable
        ],
        reviewer_override_patterns=[
            {
                "transition": op.transition,
                "application": op.application,
                "count": op.count,
                "override_rate": op.override_rate,
                "affected_reason_codes": op.affected_reason_codes,
                "affected_policy_rules": op.affected_policy_rules,
                "note": op.note,
            }
            for op in intelligence.reviewer_override_patterns
        ],
        recommendations=recommendations,
        approved_for_evaluation_count=sum(
            1 for r in recommendations if r.status is RecommendationStatus.APPROVED
        ),
        rejected_count=sum(
            1 for r in recommendations if r.status is RecommendationStatus.REJECTED
        ),
        production_configuration_status="UNCHANGED",
        notes=[
            "Closed-loop adaptive governance: detect -> record -> observe patterns -> "
            "detect drift -> attribute -> propose -> simulate -> check safety -> HUMAN "
            "APPROVAL -> APPROVED_FOR_EVALUATION. Production config is never touched.",
            "Recommendations are RECOMMENDATION ONLY. Approval means "
            "APPROVED_FOR_EVALUATION — there is no auto-deployment path and "
            "config/settings.yaml is never written.",
            "Reviewer disagreement / feedback are governance signals, not ground truth. "
            "Drift signals are operational, not proof of model degradation.",
        ],
    )
