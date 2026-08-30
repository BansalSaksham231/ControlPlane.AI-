"""Assemble the :class:`IncidentIntelligenceReport`."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime
from typing import Any

from decision.schemas import DecisionTrace
from incident.attribution import attribute_pattern
from incident.clustering import cluster_incidents
from incident.drift import build_drift_report
from incident.patterns import detect_patterns
from incident.schemas import (
    IncidentIntelligenceReport,
    Phase10IncidentConfig,
    ReviewerOverridePattern,
)
from incident.store import IncidentStore

__all__ = ["build_incident_intelligence", "build_reviewer_override_patterns"]


def build_reviewer_override_patterns(
    records: list[Any],
    governance_actions: list[Any],
    *,
    config: Phase10IncidentConfig | None = None,
) -> list[ReviewerOverridePattern]:
    config = config or Phase10IncidentConfig()
    rec_by_id = {r.interaction_id: r for r in records}
    app_by_id = {r.interaction_id: r.application for r in records}

    # group MODIFY_DECISION disagreements by (app, original -> reviewer)
    groups: dict[tuple[str, str, str], list[Any]] = {}
    reviewed_by_app_orig: Counter[tuple[str, str]] = Counter()
    for act in governance_actions:
        if act.action.value not in ("MODIFY_DECISION", "REJECT_DECISION"):
            continue
        app = app_by_id.get(act.interaction_id, "unknown")
        original = act.original_decision
        reviewed_by_app_orig[(app, original)] += 1
        if act.action.value == "REJECT_DECISION":
            reviewer = "REJECTED"
        elif act.reviewer_decision and act.reviewer_decision != original:
            reviewer = act.reviewer_decision
        else:
            continue
        groups.setdefault((app, original, reviewer), []).append(act)

    out: list[ReviewerOverridePattern] = []
    for (app, original, reviewer), acts in groups.items():
        if len(acts) < config.pattern_min_incidents:
            continue
        reason_codes: Counter[str] = Counter()
        rules: Counter[str] = Counter()
        for a in acts:
            rec = rec_by_id.get(a.interaction_id)
            if rec is not None:
                reason_codes.update(rec.reason_codes)
                rules.update(rec.tier_changing_rules)
        rate = len(acts) / max(1, reviewed_by_app_orig[(app, original)])
        out.append(
            ReviewerOverridePattern(
                pattern_id="RVP-"
                + hashlib.sha1(f"{app}|{original}|{reviewer}".encode()).hexdigest()[:8].upper(),
                application=app,
                original_decision=original,
                reviewer_decision=reviewer,
                transition=f"{original} -> {reviewer}",
                count=len(acts),
                override_rate=round(min(1.0, rate), 4),
                affected_reason_codes=[c for c, _ in reason_codes.most_common(4)],
                affected_policy_rules=[r for r, _ in rules.most_common(4)],
                representative_incidents=[a.interaction_id for a in acts[:3]],
            )
        )
    out.sort(key=lambda p: (-p.count, p.application, p.transition))
    return out


def build_incident_intelligence(
    traces: list[DecisionTrace],
    governance_actions: list[Any] | None = None,
    feedback_records: list[Any] | None = None,
    *,
    config: Phase10IncidentConfig | None = None,
    generated_at: datetime | None = None,
) -> IncidentIntelligenceReport:
    config = config or Phase10IncidentConfig()
    governance_actions = list(governance_actions or [])
    feedback_records = list(feedback_records or [])

    store = IncidentStore(config)
    records = store.ingest(traces, governance_actions, feedback_records)
    clusters = cluster_incidents(records, config)
    drift = build_drift_report(traces, governance_actions, config=config)

    # deep-routing is an ALL-traffic property, not an incidents-only one
    deep_by_app: dict[str, list[int]] = {}
    for t in traces:
        is_deep = 1 if (t.verification_path or "DEEP").upper() == "DEEP" else 0
        deep_by_app.setdefault(t.application, []).append(is_deep)
    deep_rate_by_app = {
        app: round(sum(v) / len(v), 4) for app, v in deep_by_app.items() if v
    }

    patterns = detect_patterns(
        records,
        clusters,
        drift,
        governance_actions=governance_actions,
        config=config,
        deep_rate_by_app=deep_rate_by_app,
    )

    rec_by_id = {r.interaction_id: r for r in records}
    attributions = []
    for pat in patterns:
        ids = set(pat.representative_incidents)
        for cid in pat.cluster_ids:
            cl = next((c for c in clusters if c.cluster_id == cid), None)
            if cl:
                ids.update(cl.representative_incidents)
        # for app-scoped patterns, attribute over that application's incidents
        scoped = [
            r
            for r in records
            if (not pat.applications or r.application in pat.applications)
        ]
        subject = scoped if scoped else [rec_by_id[i] for i in ids if i in rec_by_id]
        attributions.append(attribute_pattern(pat, subject))

    override_patterns = build_reviewer_override_patterns(
        records, governance_actions, config=config
    )

    return IncidentIntelligenceReport(
        generated_at=generated_at,
        config=config,
        total_incidents=len(records),
        incidents=records,
        clusters=clusters,
        patterns=patterns,
        attributions=attributions,
        reviewer_override_patterns=override_patterns,
        drift=drift,
        notes=[
            "Read-only incident intelligence. No detector / decision engine / fusion / "
            "policy / verification pass was re-run; no ground truth was read.",
            "Clusters use a deterministic structured signature (application + dimension + "
            "tier + verification path + reason codes + tier-changing rules + consequence / "
            "criticality band) — no LLM, no embeddings, no randomness.",
            "Reviewer feedback / overrides are governance signals, not ground truth, and "
            "not evidence that an automated decision was incorrect.",
            "'detection_confidence' is confidence in the pattern, not correctness of any "
            "AI response. Drift signals are operational, not proof of model degradation.",
        ],
    )
