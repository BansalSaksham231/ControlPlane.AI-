"""
Deterministic incident clustering.

Incidents are grouped by a transparent structured **signature**
(application + dominant dimension + decision tier + verification path +
reason codes + tier-changing policy rules + consequence / criticality
band). No LLM, no embeddings, no randomness.
"""

from __future__ import annotations

import hashlib
from collections import Counter

from incident.schemas import IncidentCluster, IncidentRecord, Phase10IncidentConfig
from incident.store import ts_key

__all__ = ["cluster_incidents", "readable_signature"]


def _short_id(key: str) -> str:
    return "CLU-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10].upper()


def readable_signature(record: IncidentRecord) -> str:
    parts = [record.application, record.decision]
    if record.dominant_dimension:
        parts.append(record.dominant_dimension)
    codes = record.reason_codes[:2]
    parts.extend(codes)
    if record.verification_path == "DEEP":
        parts.append("DEEP")
    return " / ".join(parts)


def _top(counter: Counter[str], n: int = 3) -> list[str]:
    return [k for k, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def cluster_incidents(
    records: list[IncidentRecord], config: Phase10IncidentConfig | None = None
) -> list[IncidentCluster]:
    config = config or Phase10IncidentConfig()

    groups: dict[str, list[IncidentRecord]] = {}
    for rec in records:
        groups.setdefault(rec.signature, []).append(rec)

    clusters: list[IncidentCluster] = []
    for signature, members in groups.items():
        members = sorted(members, key=lambda r: (ts_key(r.timestamp), r.interaction_id))
        risks = [m.overall_risk for m in members]
        confs = [m.decision_confidence for m in members]
        reason_counter: Counter[str] = Counter()
        rule_counter: Counter[str] = Counter()
        for m in members:
            reason_counter.update(m.reason_codes)
            rule_counter.update(m.tier_changing_rules)
        dims = Counter(m.dominant_dimension for m in members if m.dominant_dimension)

        clusters.append(
            IncidentCluster(
                cluster_id=_short_id(signature),
                pattern_signature=readable_signature(members[0]),
                incident_count=len(members),
                is_recurring=len(members) >= config.recurring_min_incidents,
                affected_applications=sorted({m.application for m in members}),
                representative_incidents=[
                    m.interaction_id
                    for m in sorted(members, key=lambda r: (-r.overall_risk, r.interaction_id))[:3]
                ],
                average_risk=round(sum(risks) / len(risks), 4),
                max_risk=round(max(risks), 4),
                average_confidence=round(sum(confs) / len(confs), 4),
                decisions=dict(Counter(m.decision for m in members)),
                verification_paths=dict(Counter(m.verification_path for m in members)),
                dominant_dimension=(_top(dims, 1)[0] if dims else None),
                dominant_reason_codes=_top(reason_counter),
                dominant_policy_rules=_top(rule_counter),
                reviewer_signal_count=sum(1 for m in members if m.reviewer_signal),
                first_seen=members[0].timestamp,
                last_seen=members[-1].timestamp,
            )
        )

    # deterministic: most incidents first, then by cluster_id
    clusters.sort(key=lambda c: (-c.incident_count, c.cluster_id))
    return clusters
