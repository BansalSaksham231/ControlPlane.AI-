"""
Explainability View Builder  (Phase 6 — Explainability, Step 2).

    DecisionTrace  ──►  ExplainabilitySummary

This module is a **presentation adapter** and nothing more. It:

* does NOT run a detector, the decision engine, risk fusion, the
  consequence / criticality engine, the verification router, the session
  manager, the policy engine or the policy-simulation engine;
* does NOT compute a new risk score, a new confidence score, a new
  consequence / criticality score or a new intervention tier;
* does NOT read ground truth / evaluation labels — they are not present
  on a ``DecisionTrace``;
* does NOT re-implement PII redaction. Every free-text, claim and
  evidence field is taken from :func:`decision.replay.build_replay`,
  which already applied the responsibility detector's own redaction.

Every meaningful value is copied verbatim from the stored trace (or from
the already-redacted incident replay). The only transformations are
object construction, direct field mapping and boolean flags that restate
a value already on the trace (e.g. ``used_deep = verification_path == "DEEP"``).
"""

from __future__ import annotations

from decision.replay import build_replay
from decision.schemas import DecisionTrace
from explainability.schemas import (
    ConfidenceSummary,
    ConsequenceExplanation,
    ConsequenceFactorRow,
    CounterfactualExplanation,
    CriticalEventRow,
    CriticalityExplanation,
    DecisionPathExplanation,
    EvidenceExplanation,
    ExplainabilitySummary,
    HumanReviewExplanation,
    PolicyRuleExplanation,
    RiskDimensionExplanation,
    RiskSummary,
    SessionMemoryExplanation,
    VerificationExplanation,
)

__all__ = ["build_explanation"]


def _session_memory(trace: DecisionTrace) -> SessionMemoryExplanation | None:
    """
    Map the session's contextual snapshot (carried on ``trace.session``) to the
    presentation contract. Returns ``None`` for a single-turn / stateless trace,
    a legacy ``session`` block with no ``snapshot`` key, or an empty snapshot —
    so nothing downstream has to special-case those.
    """
    session = getattr(trace, "session", None)
    if not isinstance(session, dict):
        return None
    snap = session.get("snapshot")
    if not isinstance(snap, dict):
        return None

    events = [e for e in (snap.get("critical_events") or []) if isinstance(e, dict)]
    has_critical = bool(snap.get("has_critical_history") or events)
    turns = int(snap.get("turns_recorded", 0) or 0)
    # Only populate when the multi-turn history is worth showing: >1 prior turn,
    # or any critical history. Single-turn traces carry ``session_memory=None``.
    if turns <= 1 and not has_critical:
        return None

    def _f(mapping: dict, key: str) -> float:
        try:
            return max(0.0, min(1.0, float(mapping.get(key, 0.0) or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    return SessionMemoryExplanation(
        turns_recorded=turns,
        has_critical_history=has_critical,
        critical_floor=_f(snap, "critical_floor"),
        critical_floor_applied=bool(session.get("critical_floor_applied", False)),
        peak_performance_risk=_f(snap, "peak_performance_risk"),
        peak_responsibility_risk=_f(snap, "peak_responsibility_risk"),
        peak_cost_risk=_f(snap, "peak_cost_risk"),
        pii_entity_keys=[str(k) for k in (snap.get("pii_entity_keys") or [])],
        reason_code_counts={
            str(k): int(v) for k, v in (snap.get("reason_code_counts") or {}).items()
        },
        critical_events=[
            CriticalEventRow(
                turn_index=int(e["turn_index"]),
                trigger=str(e.get("trigger", "")),
                decision=str(e.get("decision", "")),
                risk_at_event=_f(e, "risk_at_event"),
            )
            for e in events
            if "turn_index" in e
        ],
        explanation=str(session.get("explanation", "")),
    )

_HUMAN_TIERS = ("HUMAN_REVIEW", "BLOCK")


def build_explanation(
    trace: DecisionTrace,
    *,
    counterfactuals: list[CounterfactualExplanation] | None = None,
) -> ExplainabilitySummary:
    """
    Build the UI-ready :class:`ExplainabilitySummary` for one stored
    decision trace.

    ``counterfactuals`` (optional) are *already-produced*
    :class:`CounterfactualExplanation` objects (e.g. from the existing
    policy-simulation engine, formatted by a caller). They are passed
    through unchanged — this builder never runs a simulation.

    Deterministic: for a given ``trace`` the serialized output is stable.
    """
    replay = build_replay(trace)

    fd = trace.final_decision
    fusion = trace.fusion
    perf = trace.performance
    crit = trace.criticality
    cons = trace.consequence
    rep_ver = replay.verification
    rep_risk = replay.risk_signals

    # ---- 3. risk dimensions (copied from trace.fusion.risk_breakdown) ----
    dim_text = {
        "performance": rep_risk.performance_explanation,
        "responsibility": rep_risk.responsibility_explanation,
        "cost": rep_risk.cost_explanation,
    }
    risk_dimensions = [
        RiskDimensionExplanation(
            dimension=row.dimension,
            risk=row.risk,
            confidence=row.confidence,
            weight=row.weight,
            weighted_contribution=row.weighted_contribution,
            status=(perf.status.value if row.dimension == "performance" else None),
            is_dominant=(row.dimension == fusion.dominant_dimension),
            explanation=dim_text.get(row.dimension, ""),
        )
        for row in fusion.risk_breakdown
    ]

    # ---- 5. evidence (redacted claim-level view from the incident replay) ----
    evidence = [
        EvidenceExplanation(
            claim=claim.claim,                       # already redacted by build_replay
            status=claim.status,
            retrieval_similarity=claim.retrieval_similarity,
            nli_label=claim.nli_label,
            nli_confidence=claim.nli_confidence,
            evidence_strength=claim.evidence_strength,
            claim_risk=claim.claim_risk,
            supporting_evidence=claim.top_evidence,  # already redacted, rank-1 only
        )
        for claim in replay.claims
    ]

    # ---- 7. policy rules + tier path (from the replay decision path) ----
    policy_rules = [
        PolicyRuleExplanation(
            rule=step.rule,
            fired=step.fired,
            tier_before=step.tier_before,
            tier_after=step.tier_after,
            changed_tier=bool(step.fired and step.tier_before != step.tier_after),
            effect=step.effect,
            detail=step.detail,                      # already redacted
        )
        for step in replay.decision_path
    ]
    decision_path = [
        DecisionPathExplanation(
            rule=step.rule,
            from_tier=step.tier_before,
            to_tier=step.tier_after,
            reason=step.detail,                      # already redacted
        )
        for step in replay.decision_path
        if step.rule == "RISK_BAND"
        or (step.fired and step.tier_before != step.tier_after)
    ]

    # ---- 8. human review (rules that landed on HUMAN_REVIEW / BLOCK) ----
    human_conditions = [
        step.rule
        for step in replay.decision_path
        if step.fired and step.tier_after in _HUMAN_TIERS
    ]

    return ExplainabilitySummary(
        # 1. what decision
        decision=fd.decision,
        overall_risk=fd.overall_risk,
        decision_confidence=fd.decision_confidence,
        verification_path=trace.verification_path,
        human_review_required=replay.final_decision.requires_human_review,
        # 2. why
        primary_reasons=list(fd.reason_codes),
        decision_drivers=[d.rule for d in trace.decision_drivers],
        # 3. risk
        risk_summary=RiskSummary(
            overall_risk=fd.overall_risk,
            dominant_dimension=fusion.dominant_dimension,
            dominant_risk=fusion.dominant_risk,
            multi_risk=bool(fusion.multi_risk),
            weighted_only_risk=fusion.weighted_only_risk,
            criticality_weighted_performance_risk=(
                trace.criticality_weighted_performance_risk
            ),
            severity_rule_applied=bool(fusion.severity_rule_applied),
            severity_floor_applied=bool(fusion.severity_floor_applied),
        ),
        confidence_summary=ConfidenceSummary(
            decision_confidence=fd.decision_confidence,
            fused_confidence=fusion.confidence,
            fused_uncertainty=fusion.uncertainty,
            performance_confidence=perf.confidence,
            verification_confidence=perf.verification_confidence,
        ),
        risk_dimensions=risk_dimensions,
        # 4. verification
        verification_summary=VerificationExplanation(
            verification_path=trace.verification_path,
            used_deep=trace.verification_path == "DEEP",
            deep_was_forced=bool(rep_ver.deep_was_forced),
            deep_trigger_reasons=list(rep_ver.deep_trigger_reasons),
            reason_for_deep_verification=rep_ver.reason_for_deep_verification,
            preliminary_risk=rep_ver.preliminary_risk,
            preliminary_confidence=rep_ver.preliminary_confidence,
            final_risk=rep_ver.final_risk,
            final_confidence=rep_ver.final_confidence,
            disagreement_score=rep_ver.disagreement_score,
            evidence_available=rep_ver.evidence_available,
            explanation=(
                trace.verification.explanation if trace.verification else ""
            ),
        ),
        # 5. evidence
        evidence=evidence,
        # 6. why it matters
        consequence_summary=ConsequenceExplanation(
            consequence_score=cons.consequence_score,
            severity_band=cons.severity_band,
            dominant_factors=list(cons.dominant_factors),
            factors=[
                ConsequenceFactorRow(
                    factor=c.factor,
                    value=c.value,
                    weight=c.weight,
                    weighted_contribution=c.weighted_contribution,
                )
                for c in cons.contributions
            ],
            explanation=replay.consequence.explanation,
        ),
        criticality_summary=CriticalityExplanation(
            action_criticality=crit.action_criticality,
            band=crit.band,
            dominant_factors=list(crit.dominant_factors),
            max_claim_criticality=crit.max_claim_criticality,
            explanation=replay.criticality.explanation,
        ),
        # 7. policy
        policy_rules=policy_rules,
        decision_path=decision_path,
        # 8. human review
        human_review=HumanReviewExplanation(
            required=replay.final_decision.requires_human_review,
            decision=fd.decision,
            triggering_conditions=human_conditions,
            explanation=replay.explanation,
        ),
        # 9. counterfactuals — passed through, never computed here
        counterfactuals=list(counterfactuals or []),
        # 12. multi-turn session memory (None for single-turn / stateless traces)
        session_memory=_session_memory(trace),
        # top-level redacted "why"
        explanation=replay.explanation,
    )
