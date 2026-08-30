"""
Deterministic, transparent governance insights.

Each insight is a threshold crossing over already-computed governance
metrics — a *governance insight*, never a truth claim. Every insight
carries its supporting metrics and a recommended human next action.
"""

from __future__ import annotations

from typing import Any

from decision.schemas import DecisionTrace
from governance.analytics import incidents_for
from governance.schemas import (
    ApplicationComparison,
    GovernanceConfig,
    GovernanceInsight,
    GovernanceOverview,
    InsightSeverity,
    RecommendedAction,
)

__all__ = ["generate_insights"]


def _severity(observed: float, threshold: float) -> InsightSeverity:
    if threshold <= 0:
        return InsightSeverity.MEDIUM
    ratio = observed / threshold
    if ratio >= 1.6:
        return InsightSeverity.HIGH
    if ratio >= 1.25:
        return InsightSeverity.MEDIUM
    return InsightSeverity.LOW


def _example_incident_ids(
    traces: list[DecisionTrace], application: str, config: GovernanceConfig, limit: int = 3
) -> list[str]:
    incidents = incidents_for(
        [t for t in traces if t.application == application], config
    )
    return sorted(incidents)[:limit]


def generate_insights(
    overview: GovernanceOverview,
    comparison: ApplicationComparison,
    traces: list[DecisionTrace],
    *,
    config: GovernanceConfig | None = None,
) -> list[GovernanceInsight]:
    config = config or GovernanceConfig()
    insights: list[GovernanceInsight] = []
    apps = [a for a in comparison.applications if a.volume >= config.min_application_volume]

    # ---- HIGH_OVERRIDE_RATE (per application) ----
    for a in apps:
        rate = a.reviewer_override_rate
        if rate is not None and rate > config.high_override_rate:
            insights.append(
                GovernanceInsight(
                    code="HIGH_OVERRIDE_RATE",
                    severity=_severity(rate, config.high_override_rate),
                    title=f"Reviewer overrides are frequent for {a.application}",
                    explanation=(
                        f"Reviewers disagreed with the automated decision in "
                        f"{rate:.0%} of reviewed incidents for '{a.application}' "
                        f"(threshold {config.high_override_rate:.0%}). This is a "
                        "governance signal that the policy for this application "
                        "may warrant human review — it is NOT evidence that the "
                        "automated decisions were wrong."
                    ),
                    supporting_metrics={
                        "reviewer_override_rate": round(rate, 4),
                        "threshold": config.high_override_rate,
                        "block_rate": a.block_rate,
                        "human_oversight_rate": a.human_oversight_rate,
                    },
                    affected_applications=[a.application],
                    recommended_action=RecommendedAction.REVIEW_POLICY,
                    example_interaction_ids=_example_incident_ids(traces, a.application, config),
                )
            )

    # ---- HIGH_HUMAN_REVIEW_RATE (per application) ----
    for a in apps:
        rate = a.human_oversight_rate
        if rate is not None and rate > config.high_human_review_rate:
            insights.append(
                GovernanceInsight(
                    code="HIGH_HUMAN_REVIEW_RATE",
                    severity=_severity(rate, config.high_human_review_rate),
                    title=f"{a.application} sends a high share of traffic to human oversight",
                    explanation=(
                        f"{rate:.0%} of '{a.application}' interactions are routed to "
                        f"HUMAN_REVIEW or BLOCK (threshold "
                        f"{config.high_human_review_rate:.0%}). If recall is stable and "
                        "reviewer disagreement is low, the verification / risk-band "
                        "thresholds for this profile may be candidates for calibration."
                    ),
                    supporting_metrics={
                        "human_oversight_rate": round(rate, 4),
                        "threshold": config.high_human_review_rate,
                        "reviewer_override_rate": a.reviewer_override_rate,
                        "average_risk": a.average_risk,
                    },
                    affected_applications=[a.application],
                    recommended_action=RecommendedAction.REVIEW_THRESHOLD,
                    example_interaction_ids=_example_incident_ids(traces, a.application, config),
                )
            )

    # ---- LOW_CONFIDENCE_PATTERN (per application) ----
    for a in apps:
        rate = a.low_confidence_rate
        if rate is not None and rate > config.low_confidence_rate:
            insights.append(
                GovernanceInsight(
                    code="LOW_CONFIDENCE_PATTERN",
                    severity=_severity(rate, config.low_confidence_rate),
                    title=f"{a.application} has many low-confidence decisions",
                    explanation=(
                        f"{rate:.0%} of '{a.application}' decisions were made below the "
                        f"confidence threshold {overview.confidence.low_confidence_threshold:.2f} "
                        f"(governance threshold {config.low_confidence_rate:.0%}). This often "
                        "points to missing / weak evidence rather than an incorrect decision — "
                        "a detector-coverage question."
                    ),
                    supporting_metrics={
                        "low_confidence_rate": round(rate, 4),
                        "threshold": config.low_confidence_rate,
                        "deep_rate": a.deep_rate,
                    },
                    affected_applications=[a.application],
                    recommended_action=RecommendedAction.REVIEW_DETECTOR,
                    example_interaction_ids=_example_incident_ids(traces, a.application, config),
                )
            )

    # ---- DEEP_ROUTING_CONCENTRATION (per application) ----
    overall_deep = overview.verification.deep_rate or 0.0
    for a in apps:
        rate = a.deep_rate
        if (
            rate is not None
            and rate > config.deep_routing_rate
            and rate > overall_deep + 0.10
        ):
            insights.append(
                GovernanceInsight(
                    code="DEEP_ROUTING_CONCENTRATION",
                    severity=_severity(rate, config.deep_routing_rate),
                    title=f"{a.application} is disproportionately routed to DEEP verification",
                    explanation=(
                        f"{rate:.0%} of '{a.application}' interactions use DEEP verification "
                        f"(overall {overall_deep:.0%}). Progressive verification is saving "
                        "little compute for this profile; the FAST/DEEP thresholds are "
                        "calibration candidates."
                    ),
                    supporting_metrics={
                        "application_deep_rate": round(rate, 4),
                        "overall_deep_rate": round(overall_deep, 4),
                        "threshold": config.deep_routing_rate,
                    },
                    affected_applications=[a.application],
                    recommended_action=RecommendedAction.REVIEW_THRESHOLD,
                )
            )

    # ---- RULE_DOMINANCE (which rule drives most BLOCKs) ----
    block_total = sum(r.block_count for r in overview.policy.rules)
    for r in overview.policy.rules:
        if block_total >= 5 and r.block_count / block_total > config.rule_dominance_share:
            share = r.block_count / block_total
            insights.append(
                GovernanceInsight(
                    code="RULE_DOMINANCE",
                    severity=_severity(share, config.rule_dominance_share),
                    title=f"Policy rule {r.rule} drives most BLOCK decisions",
                    explanation=(
                        f"{r.rule} accounts for {share:.0%} of tier moves to BLOCK "
                        f"({r.block_count} of {block_total}). Concentrated intervention is "
                        "worth a policy review to confirm the rule is calibrated for the "
                        "current traffic mix."
                    ),
                    supporting_metrics={
                        "rule_block_count": r.block_count,
                        "total_block_moves": block_total,
                        "share": round(share, 4),
                        "fire_count": r.fire_count,
                    },
                    affected_rules=[r.rule],
                    recommended_action=RecommendedAction.REVIEW_POLICY,
                )
            )

    # ---- RISK_CONCENTRATION (few apps -> most high-risk traffic) ----
    high_by_app: dict[str, int] = {}
    thr = config.high_risk_threshold
    for t in traces:
        if t.final_decision.overall_risk >= thr:
            high_by_app[t.application] = high_by_app.get(t.application, 0) + 1
    total_high = sum(high_by_app.values())
    if total_high >= 5 and len(high_by_app) >= 2:
        top_app, top_n = max(high_by_app.items(), key=lambda kv: (kv[1], kv[0]))
        share = top_n / total_high
        if share > config.risk_concentration_share:
            insights.append(
                GovernanceInsight(
                    code="RISK_CONCENTRATION",
                    severity=_severity(share, config.risk_concentration_share),
                    title=f"High-risk traffic is concentrated in {top_app}",
                    explanation=(
                        f"'{top_app}' accounts for {share:.0%} of all high-risk interactions "
                        f"(risk >= {thr:.2f}). Governance attention (and any calibration) is "
                        "best focused there first."
                    ),
                    supporting_metrics={
                        "application": top_app,
                        "application_high_risk_count": top_n,
                        "total_high_risk_count": total_high,
                        "share": round(share, 4),
                    },
                    affected_applications=[top_app],
                    recommended_action=RecommendedAction.REVIEW_APPLICATION,
                    example_interaction_ids=_example_incident_ids(traces, top_app, config),
                )
            )

    # deterministic order
    _sev_rank = {
        InsightSeverity.HIGH: 0,
        InsightSeverity.MEDIUM: 1,
        InsightSeverity.LOW: 2,
        InsightSeverity.INFO: 3,
    }
    insights.sort(key=lambda i: (_sev_rank[i.severity], i.code, "".join(i.affected_applications)))
    return insights
