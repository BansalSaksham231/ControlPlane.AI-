"""
Incident classification — pure, config-driven, read-only.

``classify_incident(trace, config)`` decides whether one stored
:class:`decision.schemas.DecisionTrace` is an operational incident and, if
so, returns a PII-safe :class:`monitoring.schemas.IncidentSummary` (only
structured trace fields — no claim text, no response text, no
``matched_text``).

It reads fields already on the trace. It NEVER re-runs a detector, the
decision engine, fusion, policy or verification, and never reads ground
truth. Both :class:`monitoring.engine.OperationalMonitor` and the
investigation layer use these functions, so incident logic lives in
exactly one place.

Incident definition (any one triggers)
--------------------------------------
    decision == BLOCK
    OR decision == HUMAN_REVIEW
    OR fusion.multi_risk
    OR (HIGH_CONSEQUENCE reason code AND overall_risk >= elevated_risk_threshold)

Incident severity (transparent — never changes the decision)
-----------------------------------------------------------
    CRITICAL  decision == BLOCK, or overall_risk >= critical_risk_threshold,
              or CRITICAL_PII in the reason codes
    HIGH      decision == HUMAN_REVIEW, or
              (consequence_score >= high_consequence_threshold and
               overall_risk >= elevated_risk_threshold), or
              (multi_risk and overall_risk >= elevated_risk_threshold)
    MEDIUM    an incident matching none of the above
"""

from __future__ import annotations

from decision.schemas import DecisionTrace
from monitoring.schemas import IncidentSeverity, IncidentSummary, MonitoringConfig

__all__ = [
    "INCIDENT_DEFINITION",
    "SEVERITY_RANK",
    "incident_triggers",
    "incident_severity",
    "classify_incident",
]

INCIDENT_DEFINITION = (
    "decision == BLOCK  OR  decision == HUMAN_REVIEW  OR  fusion.multi_risk  "
    "OR  (HIGH_CONSEQUENCE reason code AND overall_risk >= elevated_risk_threshold)"
)

SEVERITY_RANK: dict[IncidentSeverity, int] = {
    IncidentSeverity.CRITICAL: 0,
    IncidentSeverity.HIGH: 1,
    IncidentSeverity.MEDIUM: 2,
}


def _path(trace: DecisionTrace) -> str:
    return (trace.verification_path or "DEEP").upper()


def incident_triggers(trace: DecisionTrace, config: MonitoringConfig) -> list[str]:
    """The incident-rule clauses that fired for this trace (order-stable)."""
    fd = trace.final_decision
    decision = fd.decision.value
    risk = fd.overall_risk
    codes = list(fd.reason_codes)
    multi = bool(getattr(trace.fusion, "multi_risk", False))

    triggers: list[str] = []
    if decision == "BLOCK":
        triggers.append("BLOCK")
    if decision == "HUMAN_REVIEW":
        triggers.append("HUMAN_REVIEW")
    if multi:
        triggers.append("MULTI_RISK")
    if "HIGH_CONSEQUENCE" in codes and risk >= config.elevated_risk_threshold:
        triggers.append("HIGH_CONSEQUENCE_WITH_ELEVATED_RISK")
    return triggers


def incident_severity(
    config: MonitoringConfig,
    *,
    decision: str,
    risk: float,
    reason_codes: list[str],
    multi_risk: bool,
    consequence_score: float,
) -> tuple[IncidentSeverity, str]:
    if decision == "BLOCK":
        return IncidentSeverity.CRITICAL, "decision is BLOCK"
    if risk >= config.critical_risk_threshold:
        return (
            IncidentSeverity.CRITICAL,
            f"overall_risk {risk:.2f} >= critical_risk_threshold "
            f"{config.critical_risk_threshold:.2f}",
        )
    if "CRITICAL_PII" in reason_codes:
        return IncidentSeverity.CRITICAL, "CRITICAL_PII reason code present"
    if decision == "HUMAN_REVIEW":
        return IncidentSeverity.HIGH, "decision routed to HUMAN_REVIEW"
    if (
        consequence_score >= config.high_consequence_threshold
        and risk >= config.elevated_risk_threshold
    ):
        return (
            IncidentSeverity.HIGH,
            f"consequence {consequence_score:.2f} >= "
            f"{config.high_consequence_threshold:.2f} and risk {risk:.2f} >= "
            f"{config.elevated_risk_threshold:.2f}",
        )
    if multi_risk and risk >= config.elevated_risk_threshold:
        return (
            IncidentSeverity.HIGH,
            f"multi_risk with risk {risk:.2f} >= "
            f"{config.elevated_risk_threshold:.2f}",
        )
    return (
        IncidentSeverity.MEDIUM,
        "incident triggered but no CRITICAL/HIGH condition met",
    )


def classify_incident(
    trace: DecisionTrace, config: MonitoringConfig | None = None
) -> IncidentSummary | None:
    """Return an :class:`IncidentSummary` if ``trace`` is an incident, else ``None``."""
    config = config or MonitoringConfig()
    triggers = incident_triggers(trace, config)
    if not triggers:
        return None

    fd = trace.final_decision
    decision = fd.decision.value
    risk = fd.overall_risk
    codes = list(fd.reason_codes)
    multi = bool(getattr(trace.fusion, "multi_risk", False))
    consequence = trace.consequence.consequence_score

    severity, rationale = incident_severity(
        config,
        decision=decision,
        risk=risk,
        reason_codes=codes,
        multi_risk=multi,
        consequence_score=consequence,
    )
    return IncidentSummary(
        interaction_id=trace.interaction_id,
        application=trace.application,
        timestamp=trace.timestamp,
        action_type=trace.action_type,
        decision=decision,
        overall_risk=risk,
        confidence=fd.decision_confidence,
        dominant_dimension=getattr(trace.fusion, "dominant_dimension", None),
        reason_codes=codes,
        verification_path=_path(trace),
        consequence_score=consequence,
        criticality=trace.criticality.action_criticality,
        requires_human_review=decision in ("HUMAN_REVIEW", "BLOCK"),
        severity=severity,
        triggers=triggers,
        severity_rationale=rationale,
    )
