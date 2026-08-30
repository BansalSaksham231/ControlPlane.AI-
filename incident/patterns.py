"""
Recurring failure-pattern detection.

Connects incident clusters + records + drift + governance signals into
:class:`IncidentPattern` objects. Deterministic thresholds only.

``detection_confidence`` = confidence that the *operational pattern* is
real given the sample — NOT a claim about whether any AI response or
automated decision was correct.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from incident.clustering import readable_signature
from incident.schemas import (
    DriftReport,
    IncidentCluster,
    IncidentPattern,
    IncidentRecord,
    PatternSeverity,
    PatternType,
    Phase10IncidentConfig,
)

__all__ = ["detect_patterns"]

_SEV_RANK = {
    PatternSeverity.HIGH: 0,
    PatternSeverity.MEDIUM: 1,
    PatternSeverity.LOW: 2,
    PatternSeverity.INFO: 3,
}


def _pid(kind: str, key: str) -> str:
    return f"PAT-{kind}-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:8].upper()


def _detection_confidence(count: int, effect: float) -> float:
    """More incidents + larger effect => higher confidence the pattern is real."""
    size = min(1.0, count / 12.0)
    return round(min(1.0, 0.35 + 0.4 * size + 0.35 * min(1.0, effect)), 3)


def _severity(observed: float, threshold: float) -> PatternSeverity:
    if threshold <= 0:
        return PatternSeverity.MEDIUM
    ratio = observed / threshold
    if ratio >= 1.6:
        return PatternSeverity.HIGH
    if ratio >= 1.25:
        return PatternSeverity.MEDIUM
    return PatternSeverity.LOW


def detect_patterns(
    records: list[IncidentRecord],
    clusters: list[IncidentCluster],
    drift: DriftReport,
    *,
    governance_actions: list[Any] | None = None,
    config: Phase10IncidentConfig | None = None,
    deep_rate_by_app: dict[str, float] | None = None,
) -> list[IncidentPattern]:
    config = config or Phase10IncidentConfig()
    governance_actions = governance_actions or []
    deep_rate_by_app = deep_rate_by_app or {}
    n_total = len(records)
    patterns: list[IncidentPattern] = []

    by_app: dict[str, list[IncidentRecord]] = {}
    for r in records:
        by_app.setdefault(r.application, []).append(r)

    # ---- REPEATED_BLOCK / REPEATED_VERIFY (per recurring cluster) ----
    for cl in clusters:
        if cl.incident_count < config.pattern_min_incidents:
            continue
        block = cl.decisions.get("BLOCK", 0)
        verify = cl.decisions.get("VERIFY", 0)
        if block >= config.pattern_min_incidents and block == cl.incident_count:
            patterns.append(
                IncidentPattern(
                    pattern_id=_pid("BLK", cl.cluster_id),
                    type=PatternType.REPEATED_BLOCK_PATTERN,
                    severity=_severity(block, config.pattern_min_incidents),
                    applications=cl.affected_applications,
                    incident_count=block,
                    detection_confidence=_detection_confidence(block, 1.0),
                    evidence={
                        "signature": cl.pattern_signature,
                        "dominant_reason_codes": cl.dominant_reason_codes,
                        "dominant_policy_rules": cl.dominant_policy_rules,
                        "average_risk": cl.average_risk,
                    },
                    explanation=(
                        f"{block} interactions matching '{cl.pattern_signature}' were "
                        f"repeatedly BLOCKed (average risk {cl.average_risk:.2f})."
                    ),
                    affected_dimension=cl.dominant_dimension,
                    affected_policy_rule=(cl.dominant_policy_rules[0] if cl.dominant_policy_rules else None),
                    representative_incidents=cl.representative_incidents,
                    cluster_ids=[cl.cluster_id],
                    recommended_next_step="REVIEW_POLICY",
                )
            )
        elif verify >= config.pattern_min_incidents and verify == cl.incident_count:
            patterns.append(
                IncidentPattern(
                    pattern_id=_pid("VER", cl.cluster_id),
                    type=PatternType.REPEATED_VERIFY_PATTERN,
                    severity=_severity(verify, config.pattern_min_incidents * 2),
                    applications=cl.affected_applications,
                    incident_count=verify,
                    detection_confidence=_detection_confidence(verify, 0.6),
                    evidence={
                        "signature": cl.pattern_signature,
                        "dominant_reason_codes": cl.dominant_reason_codes,
                        "average_risk": cl.average_risk,
                    },
                    explanation=(
                        f"{verify} interactions matching '{cl.pattern_signature}' repeatedly "
                        f"required VERIFY."
                    ),
                    affected_dimension=cl.dominant_dimension,
                    affected_policy_rule=(cl.dominant_policy_rules[0] if cl.dominant_policy_rules else None),
                    representative_incidents=cl.representative_incidents,
                    cluster_ids=[cl.cluster_id],
                    recommended_next_step="REVIEW_VERIFICATION_THRESHOLD",
                )
            )

    # ---- LOW_CONFIDENCE_PATTERN (per recurring cluster) ----
    for cl in clusters:
        if (
            cl.incident_count >= config.pattern_min_incidents
            and cl.average_confidence is not None
            and cl.average_confidence < config.low_confidence_threshold
        ):
            patterns.append(
                IncidentPattern(
                    pattern_id=_pid("LOWC", cl.cluster_id),
                    type=PatternType.LOW_CONFIDENCE_PATTERN,
                    severity=_severity(
                        config.low_confidence_threshold - cl.average_confidence, 0.1
                    ),
                    applications=cl.affected_applications,
                    incident_count=cl.incident_count,
                    detection_confidence=_detection_confidence(cl.incident_count, 0.5),
                    evidence={
                        "average_confidence": cl.average_confidence,
                        "threshold": config.low_confidence_threshold,
                        "signature": cl.pattern_signature,
                    },
                    explanation=(
                        f"{cl.incident_count} incidents matching '{cl.pattern_signature}' were "
                        f"decided at low confidence (mean {cl.average_confidence:.2f}); usually "
                        "a missing-evidence / detector-coverage signal, not a wrong decision."
                    ),
                    affected_dimension=cl.dominant_dimension,
                    representative_incidents=cl.representative_incidents,
                    cluster_ids=[cl.cluster_id],
                    recommended_next_step="REVIEW_DETECTOR",
                )
            )

    # ---- HIGH_OVERRIDE_PATTERN (per application) ----
    for app, recs in sorted(by_app.items()):
        overridden = [r for r in recs if r.reviewer_signal]
        reviewed = overridden  # a reviewer_signal is only set on MODIFY/REJECT
        if len(reviewed) >= config.pattern_min_incidents:
            rate = 1.0  # every reviewer_signal here is a disagreement
            transitions = Counter(r.reviewer_signal for r in overridden)
            top_t = transitions.most_common(1)[0][0]
            patterns.append(
                IncidentPattern(
                    pattern_id=_pid("OVR", app + top_t),
                    type=PatternType.HIGH_OVERRIDE_PATTERN,
                    severity=_severity(len(reviewed), config.pattern_min_incidents),
                    applications=[app],
                    incident_count=len(reviewed),
                    detection_confidence=_detection_confidence(len(reviewed), rate),
                    evidence={
                        "transitions": dict(transitions),
                        "reviewed_incidents": len(reviewed),
                    },
                    explanation=(
                        f"{len(reviewed)} '{app}' incidents received a reviewer governance "
                        f"signal; the most common is {top_t}. Reviewer disagreement is an "
                        "operational signal, not evidence the automated decision was wrong."
                    ),
                    affected_dimension=Counter(
                        r.dominant_dimension for r in overridden if r.dominant_dimension
                    ).most_common(1)[0][0]
                    if any(r.dominant_dimension for r in overridden)
                    else None,
                    affected_policy_rule=Counter(
                        rule for r in overridden for rule in r.tier_changing_rules
                    ).most_common(1)[0][0]
                    if any(r.tier_changing_rules for r in overridden)
                    else None,
                    representative_incidents=[r.interaction_id for r in overridden[:3]],
                    recommended_next_step="REVIEW_POLICY",
                )
            )

    # ---- DEEP_ROUTING_CONCENTRATION (per application — over ALL traffic) ----
    for app, share in sorted(deep_rate_by_app.items()):
        recs = by_app.get(app, [])
        if share >= config.deep_routing_rate and len(recs) >= config.pattern_min_incidents:
            patterns.append(
                IncidentPattern(
                    pattern_id=_pid("DEEP", app),
                    type=PatternType.DEEP_ROUTING_CONCENTRATION,
                    severity=_severity(share, config.deep_routing_rate),
                    applications=[app],
                    incident_count=len(recs),
                    detection_confidence=_detection_confidence(len(recs), share),
                    evidence={
                        "application_deep_rate": round(share, 4),
                        "threshold": config.deep_routing_rate,
                        "incidents_in_application": len(recs),
                    },
                    explanation=(
                        f"{share:.0%} of ALL '{app}' interactions used DEEP verification — "
                        "progressive verification is saving little compute for this profile."
                    ),
                    affected_dimension=None,
                    representative_incidents=[r.interaction_id for r in recs[:3]],
                    recommended_next_step="REVIEW_VERIFICATION_THRESHOLD",
                )
            )

    # ---- POLICY_RULE_DOMINANCE ----
    rule_moves: Counter[str] = Counter()
    for r in records:
        rule_moves.update(r.tier_changing_rules)
    total_moves = sum(rule_moves.values())
    for rule, cnt in rule_moves.items():
        if total_moves >= 5 and cnt / total_moves >= config.rule_dominance_share:
            share = cnt / total_moves
            affected = sorted({r.application for r in records if rule in r.tier_changing_rules})
            patterns.append(
                IncidentPattern(
                    pattern_id=_pid("RULE", rule),
                    type=PatternType.POLICY_RULE_DOMINANCE,
                    severity=_severity(share, config.rule_dominance_share),
                    applications=affected,
                    incident_count=cnt,
                    detection_confidence=_detection_confidence(cnt, share),
                    evidence={"rule": rule, "share_of_tier_moves": round(share, 4)},
                    explanation=(
                        f"Policy rule {rule} accounts for {share:.0%} of all incident tier "
                        f"moves ({cnt} of {total_moves})."
                    ),
                    affected_policy_rule=rule,
                    recommended_next_step="REVIEW_POLICY",
                )
            )

    # ---- DETECTOR_DOMINANCE ----
    dims: Counter[str] = Counter(r.dominant_dimension for r in records if r.dominant_dimension)
    dim_total = sum(dims.values())
    for dim, cnt in dims.items():
        if dim_total >= 5 and cnt / dim_total >= config.detector_dominance_share:
            share = cnt / dim_total
            patterns.append(
                IncidentPattern(
                    pattern_id=_pid("DET", dim),
                    type=PatternType.DETECTOR_DOMINANCE,
                    severity=_severity(share, config.detector_dominance_share),
                    applications=sorted(
                        {r.application for r in records if r.dominant_dimension == dim}
                    ),
                    incident_count=cnt,
                    detection_confidence=_detection_confidence(cnt, share),
                    evidence={"dimension": dim, "share_of_incidents": round(share, 4)},
                    explanation=(
                        f"The {dim} risk dimension is the dominant driver in {share:.0%} of "
                        f"incidents ({cnt} of {dim_total})."
                    ),
                    affected_dimension=dim,
                    recommended_next_step="REVIEW_DETECTOR",
                )
            )

    # ---- RISK_CONCENTRATION ----
    high_by_app: Counter[str] = Counter(
        r.application for r in records if r.overall_risk >= config.high_risk_threshold
    )
    total_high = sum(high_by_app.values())
    if total_high >= 5 and len(high_by_app) >= 2:
        top_app, top_n = high_by_app.most_common(1)[0]
        share = top_n / total_high
        if share >= config.risk_concentration_share:
            patterns.append(
                IncidentPattern(
                    pattern_id=_pid("RISKC", top_app),
                    type=PatternType.RISK_CONCENTRATION,
                    severity=_severity(share, config.risk_concentration_share),
                    applications=[top_app],
                    incident_count=top_n,
                    detection_confidence=_detection_confidence(top_n, share),
                    evidence={
                        "application": top_app,
                        "high_risk_share": round(share, 4),
                        "total_high_risk_incidents": total_high,
                    },
                    explanation=(
                        f"'{top_app}' holds {share:.0%} of all high-risk incidents "
                        f"(risk >= {config.high_risk_threshold:.2f})."
                    ),
                    representative_incidents=[
                        r.interaction_id
                        for r in sorted(
                            (r for r in records if r.application == top_app),
                            key=lambda x: (-x.overall_risk, x.interaction_id),
                        )[:3]
                    ],
                    recommended_next_step="REVIEW_APPLICATION",
                )
            )

    # ---- CROSS_APPLICATION_PATTERN ----
    def _app_free(sig: str) -> str:
        return "|".join(p for p in sig.split("|") if not p.startswith("app:"))

    xapp: dict[str, set[str]] = {}
    xapp_incidents: dict[str, list[IncidentRecord]] = {}
    for r in records:
        key = _app_free(r.signature)
        xapp.setdefault(key, set()).add(r.application)
        xapp_incidents.setdefault(key, []).append(r)
    for key, apps in xapp.items():
        recs = xapp_incidents[key]
        if len(apps) >= 2 and len(recs) >= config.pattern_min_incidents:
            patterns.append(
                IncidentPattern(
                    pattern_id=_pid("XAPP", key),
                    type=PatternType.CROSS_APPLICATION_PATTERN,
                    severity=PatternSeverity.MEDIUM,
                    applications=sorted(apps),
                    incident_count=len(recs),
                    detection_confidence=_detection_confidence(len(recs), 0.7),
                    evidence={
                        "shared_signature": readable_signature(recs[0]),
                        "applications": sorted(apps),
                    },
                    explanation=(
                        f"The same incident signature ('{readable_signature(recs[0])}') recurs "
                        f"across {len(apps)} applications ({len(recs)} incidents)."
                    ),
                    affected_dimension=recs[0].dominant_dimension,
                    representative_incidents=[r.interaction_id for r in recs[:3]],
                    recommended_next_step="REVIEW_DETECTOR",
                )
            )

    # ---- INCIDENT_SURGE (from drift on incident_rate) ----
    for s in drift.signals:
        if (
            s.scope == "global"
            and s.metric == "incident_rate"
            and s.direction == s.direction.INCREASING
            and s.baseline is not None
            and s.recent is not None
            and s.baseline > 0
            and s.recent >= s.baseline * config.surge_factor
        ):
            patterns.append(
                IncidentPattern(
                    pattern_id=_pid("SURGE", "global"),
                    type=PatternType.INCIDENT_SURGE,
                    severity=PatternSeverity.HIGH if s.signal == "POTENTIAL_DRIFT" else PatternSeverity.MEDIUM,
                    applications=sorted({r.application for r in records}),
                    incident_count=n_total,
                    detection_confidence=_detection_confidence(
                        drift.recent_window_n, min(1.0, s.recent / max(s.baseline, 1e-6) - 1)
                    ),
                    evidence={
                        "historical_incident_rate": s.baseline,
                        "recent_incident_rate": s.recent,
                        "surge_factor": config.surge_factor,
                        "drift_signal": s.signal,
                    },
                    explanation=(
                        f"Incident rate rose from {s.baseline:.1%} to {s.recent:.1%} between the "
                        "historical and recent windows — an operational surge, not confirmed drift."
                    ),
                    recommended_next_step="INVESTIGATE_DRIFT",
                )
            )

    patterns.sort(key=lambda p: (_SEV_RANK[p.severity], p.type.value, "".join(p.applications), p.pattern_id))
    return patterns
