"""
Governance analytics — pure, deterministic aggregation over existing data.

``build_overview(traces, governance_actions, config)`` and
``compare_applications(...)`` read only:

    DecisionTrace           (decision, risk, confidence, verification, policy trace)
    monitoring.incidents.classify_incident(trace)   (incident attribution)
    investigation GovernanceAction                  (human review outcomes)

They never re-run a detector, the decision engine, fusion, policy or
verification, and never touch ground truth.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

from data.schemas import InterventionTier
from decision.schemas import DecisionTrace
from monitoring.incidents import classify_incident
from monitoring.metrics import mean_or_none, percentile_or_none, rate_or_none
from monitoring.schemas import MonitoringConfig
from governance.schemas import (
    ApplicationComparison,
    ApplicationGovernanceMetrics,
    ConfidenceDistribution,
    DecisionDistribution,
    DetectorContributionMetrics,
    GovernanceConfig,
    GovernanceOverview,
    HumanGovernanceMetrics,
    PolicyBehaviourMetrics,
    PolicyRuleMetric,
    ReviewerDisagreementMetrics,
    RiskDistribution,
    TrafficMetrics,
    VerificationMetrics,
)

__all__ = ["ts_key", "build_overview", "compare_applications", "incidents_for"]

_TIER_VALUES = ("ALLOW", "ANNOTATE", "VERIFY", "HUMAN_REVIEW", "BLOCK")
_HUMAN_TIERS = ("HUMAN_REVIEW", "BLOCK")
_DISAGREEMENT_ACTIONS = ("MODIFY_DECISION", "REJECT_DECISION")


def ts_key(ts: datetime) -> datetime:
    """Naive-UTC sort key, tz-safe (mixed aware/naive trace timestamps)."""
    if ts.tzinfo is not None:
        return ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _path(trace: DecisionTrace) -> str:
    return (trace.verification_path or "DEEP").upper()


def _mon_config(config: GovernanceConfig) -> MonitoringConfig:
    """A MonitoringConfig aligned with the governance thresholds (for incidents)."""
    return MonitoringConfig(
        low_confidence_threshold=config.low_confidence_threshold,
        elevated_risk_threshold=config.high_risk_threshold,
    )


def incidents_for(
    traces: Iterable[DecisionTrace], config: GovernanceConfig | None = None
) -> dict[str, Any]:
    """interaction_id -> IncidentSummary for every trace that is an incident."""
    mon = _mon_config(config or GovernanceConfig())
    out: dict[str, Any] = {}
    for trace in traces:
        inc = classify_incident(trace, mon)
        if inc is not None:
            out[trace.interaction_id] = inc
    return out


# ----------------------------------------------------------------------
# overview
# ----------------------------------------------------------------------


def build_overview(
    traces: list[DecisionTrace],
    governance_actions: list[Any],
    *,
    config: GovernanceConfig | None = None,
    generated_at: datetime | None = None,
) -> GovernanceOverview:
    config = config or GovernanceConfig()
    ordered = sorted(traces, key=lambda t: (ts_key(t.timestamp), t.interaction_id))
    n = len(ordered)
    incidents = incidents_for(ordered, config)

    return GovernanceOverview(
        generated_at=generated_at,
        basis="timestamped" if _has_distinct_timestamps(ordered) else "sequence-based",
        config=config,
        traffic=_traffic(ordered),
        decisions=_decisions(ordered, n),
        risk=_risk(ordered, config),
        confidence=_confidence(ordered, config),
        verification=_verification(ordered, n),
        detector_contribution=_detector_contribution(incidents),
        policy=_policy(ordered, n),
        human_governance=_human_governance(governance_actions),
        reviewer_disagreement=_reviewer_disagreement(governance_actions),
        notes=[
            "Aggregated from stored DecisionTrace records + governance actions. "
            "No detector, decision engine, fusion, policy or verification pass was re-run.",
            "No ground truth is read. Reviewer decisions are governance signals, "
            "not correctness labels.",
        ],
    )


def _has_distinct_timestamps(traces: list[DecisionTrace]) -> bool:
    seen = {ts_key(t.timestamp) for t in traces}
    return len(seen) > 1


def _traffic(traces: list[DecisionTrace]) -> TrafficMetrics:
    return TrafficMetrics(
        total_interactions=len(traces),
        by_application=dict(Counter(t.application for t in traces)),
        by_action_type=dict(Counter(t.action_type for t in traces)),
    )


def _decisions(traces: list[DecisionTrace], n: int) -> DecisionDistribution:
    c = Counter(t.final_decision.decision.value for t in traces)
    hr, bl = c.get("HUMAN_REVIEW", 0), c.get("BLOCK", 0)
    return DecisionDistribution(
        allow=c.get("ALLOW", 0),
        annotate=c.get("ANNOTATE", 0),
        verify=c.get("VERIFY", 0),
        human_review=hr,
        block=bl,
        allow_rate=rate_or_none(c.get("ALLOW", 0), n),
        annotate_rate=rate_or_none(c.get("ANNOTATE", 0), n),
        verify_rate=rate_or_none(c.get("VERIFY", 0), n),
        human_review_rate=rate_or_none(hr, n),
        block_rate=rate_or_none(bl, n),
        human_oversight_rate=rate_or_none(hr + bl, n),
    )


def _risk(traces: list[DecisionTrace], config: GovernanceConfig) -> RiskDistribution:
    risks = [t.final_decision.overall_risk for t in traces]
    high = sum(1 for r in risks if r >= config.high_risk_threshold)
    return RiskDistribution(
        average_risk=mean_or_none(risks),
        p50_risk=percentile_or_none(risks, 50),
        p95_risk=percentile_or_none(risks, 95),
        max_risk=max(risks) if risks else None,
        high_risk_rate=rate_or_none(high, len(risks)),
        high_risk_threshold=config.high_risk_threshold,
    )


def _confidence(traces: list[DecisionTrace], config: GovernanceConfig) -> ConfidenceDistribution:
    confs = [t.final_decision.decision_confidence for t in traces]
    low = sum(1 for c in confs if c < config.low_confidence_threshold)
    return ConfidenceDistribution(
        average_confidence=mean_or_none(confs),
        low_confidence_rate=rate_or_none(low, len(confs)),
        low_confidence_threshold=config.low_confidence_threshold,
    )


def _verification(traces: list[DecisionTrace], n: int) -> VerificationMetrics:
    fast = [t for t in traces if _path(t) == "FAST"]
    deep = [t for t in traces if _path(t) == "DEEP"]
    trig: Counter[str] = Counter()
    vlat: list[float] = []
    for t in traces:
        if t.verification is not None:
            trig.update(t.verification.deep_trigger_reasons)
            vlat.append(t.verification.total_verification_latency_ms)
    return VerificationMetrics(
        fast_count=len(fast),
        deep_count=len(deep),
        fast_rate=rate_or_none(len(fast), n),
        deep_rate=rate_or_none(len(deep), n),
        average_verification_latency_ms=mean_or_none(vlat),
        average_total_latency_ms=mean_or_none([t.latency_ms for t in traces]),
        deep_trigger_reason_counts={
            k: trig[k] for k in sorted(trig, key=lambda x: (-trig[x], x))
        },
    )


def _detector_contribution(incidents: dict[str, Any]) -> DetectorContributionMetrics:
    perf = resp = cost = multi = 0
    for inc in incidents.values():
        dim = inc.dominant_dimension
        if dim == "performance":
            perf += 1
        elif dim == "responsibility":
            resp += 1
        elif dim == "cost":
            cost += 1
        if "MULTI_RISK" in inc.triggers:
            multi += 1
    return DetectorContributionMetrics(
        total_incidents=len(incidents),
        performance_driven_incidents=perf,
        responsibility_driven_incidents=resp,
        cost_driven_incidents=cost,
        multi_risk_incidents=multi,
    )


def _policy(traces: list[DecisionTrace], n: int) -> PolicyBehaviourMetrics:
    fire: Counter[str] = Counter()
    tier_change: Counter[str] = Counter()
    to_hr: Counter[str] = Counter()
    to_block: Counter[str] = Counter()

    for trace in traces:
        for entry in trace.policy.rule_trace:
            if entry.fired:
                fire[entry.rule] += 1
        for step in trace.decision_path:
            if step.from_tier != step.to_tier:
                tier_change[step.rule] += 1
                if step.to_tier is InterventionTier.HUMAN_REVIEW:
                    to_hr[step.rule] += 1
                elif step.to_tier is InterventionTier.BLOCK:
                    to_block[step.rule] += 1

    rules = sorted(
        set(fire) | set(tier_change),
        key=lambda r: (-fire.get(r, 0), -tier_change.get(r, 0), r),
    )
    return PolicyBehaviourMetrics(
        total_interactions=n,
        rules=[
            PolicyRuleMetric(
                rule=r,
                fire_count=fire.get(r, 0),
                fire_rate=rate_or_none(fire.get(r, 0), n),
                tier_changing_count=tier_change.get(r, 0),
                human_review_count=to_hr.get(r, 0),
                block_count=to_block.get(r, 0),
            )
            for r in rules
        ],
    )


def _human_governance(governance_actions: list[Any]) -> HumanGovernanceMetrics:
    by_interaction: dict[str, list[Any]] = {}
    for act in governance_actions:
        by_interaction.setdefault(act.interaction_id, []).append(act)

    status_counter: Counter[str] = Counter()
    for actions in by_interaction.values():
        latest = actions[-1]
        status_counter[latest.new_status.value] += 1

    return HumanGovernanceMetrics(
        incidents_investigated=len(by_interaction),
        open=status_counter.get("OPEN", 0),
        acknowledged=status_counter.get("ACKNOWLEDGED", 0),
        reviewed=status_counter.get("REVIEWED", 0),
        escalated=status_counter.get("ESCALATED", 0),
        closed=status_counter.get("CLOSED", 0),
        action_counts=dict(Counter(a.action.value for a in governance_actions)),
    )


def _reviewer_disagreement(governance_actions: list[Any]) -> ReviewerDisagreementMetrics:
    reviewed = [
        a for a in governance_actions if a.action.value in _DISAGREEMENT_ACTIONS
    ]
    overrides = 0
    reviewer_dist: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    for act in reviewed:
        if act.action.value == "REJECT_DECISION":
            overrides += 1
        elif act.reviewer_decision is not None:
            reviewer_dist[act.reviewer_decision] += 1
            if act.reviewer_decision != act.original_decision:
                overrides += 1
                transitions[f"{act.original_decision} -> {act.reviewer_decision}"] += 1
    return ReviewerDisagreementMetrics(
        reviewed_count=len(reviewed),
        override_count=overrides,
        override_rate=rate_or_none(overrides, len(reviewed)),
        reviewer_decision_distribution=dict(reviewer_dist),
        automated_to_reviewer_transitions=dict(
            sorted(transitions.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
    )


# ----------------------------------------------------------------------
# application comparison  (Step 2)
# ----------------------------------------------------------------------


def compare_applications(
    traces: list[DecisionTrace],
    governance_actions: list[Any],
    *,
    config: GovernanceConfig | None = None,
) -> ApplicationComparison:
    config = config or GovernanceConfig()
    incidents = incidents_for(traces, config)

    # reviewer overrides per application (via the trace's application)
    trace_app = {t.interaction_id: t.application for t in traces}
    override_by_app: Counter[str] = Counter()
    reviewed_by_app: Counter[str] = Counter()
    for act in governance_actions:
        if act.action.value not in _DISAGREEMENT_ACTIONS:
            continue
        app = trace_app.get(act.interaction_id)
        if app is None:
            continue
        reviewed_by_app[app] += 1
        disagreed = act.action.value == "REJECT_DECISION" or (
            act.reviewer_decision is not None
            and act.reviewer_decision != act.original_decision
        )
        if disagreed:
            override_by_app[app] += 1

    by_app: dict[str, list[DecisionTrace]] = {}
    for t in traces:
        by_app.setdefault(t.application, []).append(t)

    rows: list[ApplicationGovernanceMetrics] = []
    for app in sorted(by_app):
        group = by_app[app]
        k = len(group)
        decisions = Counter(t.final_decision.decision.value for t in group)
        hr, bl = decisions.get("HUMAN_REVIEW", 0), decisions.get("BLOCK", 0)
        risks = [t.final_decision.overall_risk for t in group]
        confs = [t.final_decision.decision_confidence for t in group]
        n_incidents = sum(1 for t in group if t.interaction_id in incidents)
        rows.append(
            ApplicationGovernanceMetrics(
                application=app,
                volume=k,
                allow_rate=rate_or_none(decisions.get("ALLOW", 0), k),
                verify_rate=rate_or_none(decisions.get("VERIFY", 0), k),
                human_review_rate=rate_or_none(hr, k),
                block_rate=rate_or_none(bl, k),
                human_oversight_rate=rate_or_none(hr + bl, k),
                average_risk=mean_or_none(risks),
                p95_risk=percentile_or_none(risks, 95),
                average_confidence=mean_or_none(confs),
                low_confidence_rate=rate_or_none(
                    sum(1 for c in confs if c < config.low_confidence_threshold), k
                ),
                fast_rate=rate_or_none(
                    sum(1 for t in group if _path(t) == "FAST"), k
                ),
                deep_rate=rate_or_none(
                    sum(1 for t in group if _path(t) == "DEEP"), k
                ),
                incident_rate=rate_or_none(n_incidents, k),
                reviewer_override_rate=rate_or_none(
                    override_by_app.get(app, 0), reviewed_by_app.get(app, 0)
                ),
                average_latency_ms=mean_or_none([t.latency_ms for t in group]),
            )
        )

    highest_risk = max(rows, key=lambda r: r.average_risk or 0.0).application if rows else None
    highest_volume = max(rows, key=lambda r: r.volume).application if rows else None
    lowest_interv = (
        min(rows, key=lambda r: r.human_oversight_rate or 0.0).application if rows else None
    )
    return ApplicationComparison(
        applications=rows,
        highest_risk=highest_risk,
        highest_volume=highest_volume,
        lowest_intervention=lowest_interv,
        notes=[
            "The application name is a dimension of analysis only — there is no "
            "application-specific risk logic here.",
            "A lower risk / intervention rate is an observed profile, not a claim "
            "that an application is 'safer'.",
        ],
    )
