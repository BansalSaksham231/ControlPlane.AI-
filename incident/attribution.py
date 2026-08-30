"""
Deterministic attribution: "what appears to be driving this pattern?"

Produces observed-association percentages only. The narrative uses
"associated with" / "dominant contributor" / "observed alongside" — never
"caused by".
"""

from __future__ import annotations

import hashlib
from collections import Counter

from incident.schemas import AttributionResult, IncidentPattern, IncidentRecord

__all__ = ["attribute_pattern"]


def _shares(counter: Counter[str], total: int) -> dict[str, float]:
    if total == 0:
        return {}
    return {
        k: round(v / total, 4)
        for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    }


def attribute_pattern(
    pattern: IncidentPattern, incidents: list[IncidentRecord]
) -> AttributionResult:
    n = len(incidents)
    dim_counter: Counter[str] = Counter(
        i.dominant_dimension for i in incidents if i.dominant_dimension
    )
    reason_counter: Counter[str] = Counter()
    rule_counter: Counter[str] = Counter()
    decision_counter: Counter[str] = Counter(i.decision for i in incidents)
    overrides = 0
    for i in incidents:
        reason_counter.update(i.reason_codes)
        rule_counter.update(i.tier_changing_rules)
        if i.reviewer_signal:
            overrides += 1

    dim_shares = _shares(dim_counter, n)
    reason_shares = _shares(reason_counter, n)
    rule_shares = _shares(rule_counter, n)
    dominant_dim = next(iter(dim_shares), None)
    dominant_rule = next(iter(rule_shares), None)
    override_share = round(overrides / n, 4) if n else 0.0

    bits: list[str] = []
    if dominant_dim:
        bits.append(
            f"the {dominant_dim} detector is the dominant contributor "
            f"({dim_shares[dominant_dim]:.0%} of incidents)"
        )
    top_reason = next(iter(reason_shares), None)
    if top_reason:
        bits.append(
            f"{top_reason} is observed alongside {reason_shares[top_reason]:.0%} of incidents"
        )
    if dominant_rule:
        bits.append(
            f"policy rule {dominant_rule} is associated with "
            f"{rule_shares[dominant_rule]:.0%} of the tier moves"
        )
    if override_share > 0:
        bits.append(
            f"reviewers left a governance signal on {override_share:.0%} of incidents"
        )
    narrative = (
        f"Across {n} incidents in this pattern, " + "; ".join(bits) + "."
        if bits
        else f"Across {n} incidents, no single factor dominates."
    )

    return AttributionResult(
        pattern_id=pattern.pattern_id,
        incident_count=n,
        dominant_dimension=dominant_dim,
        dimension_shares=dim_shares,
        reason_code_shares=reason_shares,
        decision_shares=_shares(decision_counter, n),
        dominant_policy_rule=dominant_rule,
        policy_rule_shares=rule_shares,
        reviewer_override_share=override_share,
        narrative=narrative,
    )
