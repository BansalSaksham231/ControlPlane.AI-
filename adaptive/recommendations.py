"""
Adaptive recommendation engine.

Consumes incident patterns + drift + reviewer-override patterns and
produces deterministic :class:`AdaptiveRecommendation` objects. When a
counterfactual ``ConfigurationSelection`` is supplied, threshold
recommendations are enriched with the candidate config + simulation +
safety verdict + expected trade-off.

Nothing here mutates configuration. Default status is
``RECOMMENDED_FOR_REVIEW``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from adaptive.counterfactual import expected_tradeoff_text, selection_to_evaluation
from adaptive.schemas import (
    AdaptiveConfig,
    AdaptiveRecommendation,
    RecommendationSeverity,
    RecommendationStatus,
    RecommendationType,
)
from incident.schemas import IncidentIntelligenceReport

__all__ = ["build_adaptive_recommendations"]

_PATTERN_TO_TYPE = {
    "HIGH_OVERRIDE_PATTERN": RecommendationType.REVIEW_POLICY,
    "REPEATED_BLOCK_PATTERN": RecommendationType.REVIEW_POLICY,
    "POLICY_RULE_DOMINANCE": RecommendationType.REVIEW_POLICY,
    "REPEATED_VERIFY_PATTERN": RecommendationType.REVIEW_VERIFICATION_THRESHOLD,
    "DEEP_ROUTING_CONCENTRATION": RecommendationType.REVIEW_VERIFICATION_THRESHOLD,
    "RISK_CONCENTRATION": RecommendationType.REVIEW_APPLICATION,
    "DETECTOR_DOMINANCE": RecommendationType.REVIEW_DETECTOR,
    "LOW_CONFIDENCE_PATTERN": RecommendationType.REVIEW_DETECTOR,
    "CROSS_APPLICATION_PATTERN": RecommendationType.REVIEW_DETECTOR,
    "INCIDENT_SURGE": RecommendationType.INVESTIGATE_DRIFT,
}

_SEV_MAP = {
    "HIGH": RecommendationSeverity.HIGH,
    "MEDIUM": RecommendationSeverity.MEDIUM,
    "LOW": RecommendationSeverity.LOW,
    "INFO": RecommendationSeverity.INFO,
}
_SEV_RANK = {
    RecommendationSeverity.HIGH: 0,
    RecommendationSeverity.MEDIUM: 1,
    RecommendationSeverity.LOW: 2,
    RecommendationSeverity.INFO: 3,
}
_TYPE_RANK = {
    RecommendationType.REVIEW_POLICY: 0,
    RecommendationType.REVIEW_VERIFICATION_THRESHOLD: 1,
    RecommendationType.INVESTIGATE_DRIFT: 2,
    RecommendationType.REVIEW_APPLICATION: 3,
    RecommendationType.REVIEW_DETECTOR: 4,
    RecommendationType.NO_ACTION: 5,
}


def _rid(rtype: RecommendationType, application: str | None) -> str:
    key = f"{rtype.value}|{application or 'GLOBAL'}"
    return "REC-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10].upper()


def _proposed_change(rtype: RecommendationType, application: str | None) -> str:
    app = f" for '{application}'" if application else ""
    return {
        RecommendationType.REVIEW_POLICY: (
            f"Human review of the policy profile{app}. No automated threshold "
            "change — reviewer disagreement / repeated intervention is a "
            "governance signal, not a correctness finding."
        ),
        RecommendationType.REVIEW_VERIFICATION_THRESHOLD: (
            "Evaluate an alternative FAST/DEEP verification-threshold "
            "configuration via calibration.sweep + calibration.select "
            "(recommendation only — see the counterfactual)."
        ),
        RecommendationType.REVIEW_APPLICATION: (
            f"Focus governance attention{app}: it holds a disproportionate share "
            "of high-risk interactions."
        ),
        RecommendationType.REVIEW_DETECTOR: (
            "Review detector coverage / calibration for the dominant risk "
            "dimension driving these incidents."
        ),
        RecommendationType.INVESTIGATE_DRIFT: (
            "Investigate the recent operational change before considering any "
            "configuration adjustment."
        ),
        RecommendationType.NO_ACTION: "No adaptive action indicated.",
    }[rtype]


def build_adaptive_recommendations(
    intelligence: IncidentIntelligenceReport,
    *,
    config: AdaptiveConfig | None = None,
    calibration_selection: Any | None = None,
    approval_store: Any | None = None,
) -> list[AdaptiveRecommendation]:
    config = config or AdaptiveConfig()

    # group patterns by (recommendation type, application)
    buckets: dict[tuple[RecommendationType, str | None], dict[str, Any]] = {}
    for pat in intelligence.patterns:
        rtype = _PATTERN_TO_TYPE.get(pat.type.value)
        if rtype is None:
            continue
        apps = pat.applications or [None]
        for app in apps:
            key = (rtype, app)
            b = buckets.setdefault(
                key, {"patterns": [], "severity": RecommendationSeverity.INFO, "incidents": 0}
            )
            b["patterns"].append(pat)
            b["incidents"] += pat.incident_count
            sev = _SEV_MAP[pat.severity.value]
            if _SEV_RANK[sev] < _SEV_RANK[b["severity"]]:
                b["severity"] = sev

    # reviewer-override patterns reinforce REVIEW_POLICY
    for op in intelligence.reviewer_override_patterns:
        key = (RecommendationType.REVIEW_POLICY, op.application)
        b = buckets.setdefault(
            key, {"patterns": [], "severity": RecommendationSeverity.MEDIUM, "incidents": 0}
        )
        b.setdefault("override_transitions", []).append(op.transition)
        b["incidents"] += op.count

    recs: list[AdaptiveRecommendation] = []
    threshold_eval = None
    if calibration_selection is not None:
        threshold_eval = selection_to_evaluation(calibration_selection)

    for (rtype, app), b in buckets.items():
        pattern_ids = [p.pattern_id for p in b["patterns"]]
        pattern_types = sorted({p.type.value for p in b["patterns"]})
        evidence: dict[str, Any] = {
            "pattern_types": pattern_types,
            "incident_count": b["incidents"],
        }
        if "override_transitions" in b:
            evidence["reviewer_override_transitions"] = sorted(set(b["override_transitions"]))

        rec = AdaptiveRecommendation(
            recommendation_id=_rid(rtype, app),
            type=rtype,
            application=app,
            severity=b["severity"],
            trigger_patterns=sorted(set(pattern_ids)),
            evidence=evidence,
            rationale=_rationale(rtype, app, b, pattern_types),
            proposed_change=_proposed_change(rtype, app),
            status=RecommendationStatus.RECOMMENDED_FOR_REVIEW,
            points_to=_points_to(rtype, app),
        )

        if rtype is RecommendationType.REVIEW_VERIFICATION_THRESHOLD and threshold_eval is not None:
            rec.simulation_result = threshold_eval
            rec.current_configuration = threshold_eval.current_configuration
            rec.safety_constraints = threshold_eval.safety_constraints
            rec.expected_tradeoff = expected_tradeoff_text(threshold_eval)
            if threshold_eval.safety_passed and threshold_eval.candidate_configuration:
                rec.candidate_configuration = threshold_eval.candidate_configuration
                rec.status = RecommendationStatus.SIMULATED
            else:
                rec.status = RecommendationStatus.RECOMMENDED_FOR_REVIEW
                rec.rationale += (
                    "  Counterfactual: no safe candidate configuration was found "
                    "under the configured safety constraints — no threshold change "
                    "is recommended."
                )

        # overlay any human approval decision
        if approval_store is not None:
            approval = approval_store.get(rec.recommendation_id)
            if approval is not None:
                rec.approval = approval
                rec.status = (
                    RecommendationStatus.APPROVED
                    if approval.decision == "APPROVED_FOR_EVALUATION"
                    else RecommendationStatus.REJECTED
                )
        recs.append(rec)

    if not recs:
        recs.append(
            AdaptiveRecommendation(
                recommendation_id=_rid(RecommendationType.NO_ACTION, None),
                type=RecommendationType.NO_ACTION,
                severity=RecommendationSeverity.INFO,
                rationale=(
                    "No recurring incident pattern, drift signal or reviewer-override "
                    "pattern crossed a configured threshold."
                ),
                proposed_change=_proposed_change(RecommendationType.NO_ACTION, None),
                status=RecommendationStatus.RECOMMENDED_FOR_REVIEW,
            )
        )

    recs.sort(
        key=lambda r: (
            _TYPE_RANK[r.type],
            _SEV_RANK[r.severity],
            r.application or "",
            r.recommendation_id,
        )
    )
    return recs


def _rationale(rtype, app, bucket, pattern_types) -> str:
    app_s = f"'{app}' " if app else ""
    n = bucket["incidents"]
    types = ", ".join(pattern_types)
    if rtype is RecommendationType.REVIEW_POLICY:
        extra = ""
        if "override_transitions" in bucket:
            extra = (
                f" Reviewers repeatedly changed the tier ("
                + ", ".join(sorted(set(bucket["override_transitions"])))
                + ") — a governance signal, not evidence the automated decision was wrong."
            )
        return (
            f"{app_s}shows recurring intervention patterns ({types}) across {n} incidents.{extra} "
            "A human policy review is recommended."
        )
    if rtype is RecommendationType.REVIEW_VERIFICATION_THRESHOLD:
        return (
            f"{app_s}is heavily / repeatedly routed to DEEP verification ({types}, {n} incidents). "
            "The FAST/DEEP thresholds are calibration candidates."
        )
    if rtype is RecommendationType.REVIEW_APPLICATION:
        return f"{app_s}holds a disproportionate share of high-risk incidents ({types})."
    if rtype is RecommendationType.REVIEW_DETECTOR:
        return (
            f"One risk dimension dominates {n} incidents{(' for ' + app) if app else ''} "
            f"({types}); detector coverage / calibration for that dimension is worth a review."
        )
    if rtype is RecommendationType.INVESTIGATE_DRIFT:
        return (
            f"An operational surge / drift signal was detected ({types}). Investigate the "
            "recent change before considering any configuration adjustment."
        )
    return "No adaptive action indicated."


def _points_to(rtype, app) -> list[str]:
    if rtype is RecommendationType.REVIEW_VERIFICATION_THRESHOLD:
        return ["calibration.sweep", "calibration.select", "simulation.engine"]
    if rtype is RecommendationType.REVIEW_POLICY:
        return [f"policy profile: {app}" if app else "policy engine", "investigation"]
    if rtype is RecommendationType.INVESTIGATE_DRIFT:
        return ["incident.drift", "monitoring"]
    return ["investigation"]
