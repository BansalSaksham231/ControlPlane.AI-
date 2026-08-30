"""
Incident Replay — reconstruct what happened during a ControlPlane decision.

``build_replay(trace)`` transforms an **already-created** ``DecisionTrace``
into a clean, redacted, judge-friendly ``IncidentReplay``.

It is a pure reconstruction layer:

    trace  ->  replay          (this module)

    NOT

    interaction -> decision -> replay

It never re-runs a detector, the fusion engine, the policy engine or the
decision engine. Every number is copied verbatim from the stored trace.

Safety:

* All free text (claim text, retrieved evidence, explanations, rule
  details) is passed through the same PII redaction the responsibility
  detector produced — raw ``matched_text`` is never exposed.
* Ground-truth / evaluation-only fields (``ground_truth_*``,
  ``expected_decision``, ``final_outcome``) are never read — they are not
  even present on a ``DecisionTrace``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from decision.schemas import DecisionTrace
from verification.schemas import DisagreementBreakdown

__all__ = ["IncidentReplay", "IncidentReplayNotFound", "build_replay"]


# ------------------------------------------------------------------ redaction


class _Redactor:
    """Ordered substring redaction driven by the trace's own PII findings."""

    def __init__(self, trace: DecisionTrace) -> None:
        pairs: list[tuple[str, str]] = []
        for finding in trace.responsibility.pii.findings:
            raw = (finding.matched_text or "").strip()
            if raw:
                pairs.append((raw, finding.redacted_text or "[REDACTED]"))
        # Longest raw match first so overlapping identifiers redact cleanly.
        self._pairs = sorted(set(pairs), key=lambda p: len(p[0]), reverse=True)

    def text(self, value: str | None) -> str:
        if not value:
            return value or ""
        out = value
        for raw, redacted in self._pairs:
            if raw in out:
                out = out.replace(raw, redacted)
        return out


# ------------------------------------------------------------------ schema


class ReplayInteraction(BaseModel):
    interaction_id: str
    timestamp: datetime
    application: str
    action_type: str
    model: str | None = None            # not stored on the trace
    response: str                       # redacted


class ReplayRiskSignals(BaseModel):
    performance_risk: float
    responsibility_risk: float
    cost_risk: float
    overall_risk: float
    weighted_only_risk: float | None = None
    criticality_weighted_performance_risk: float | None = None
    dominant_dimension: str | None = None
    multi_risk: bool = False

    performance_status: str
    responsibility_reason_codes: list[str] = Field(default_factory=list)
    cost_anomaly_types: list[str] = Field(default_factory=list)
    cost_triggered_dimensions: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    performance_explanation: str
    responsibility_explanation: str
    cost_explanation: str
    fusion_explanation: str


class ReplayClaim(BaseModel):
    claim: str                          # redacted
    status: str
    claim_risk: float
    evidence_strength: float
    retrieval_similarity: float
    nli_label: str | None = None
    nli_confidence: float | None = None
    top_evidence: str | None = None     # redacted, rank-1 only


class ReplayConfidence(BaseModel):
    performance_confidence: float
    performance_uncertainty: float
    performance_evidence_quality: float
    verification_confidence: float
    fused_confidence: float
    fused_uncertainty: float
    decision_confidence: float
    note: str = (
        "RISK is how dangerous the interaction looks; CONFIDENCE is how sure "
        "ControlPlane is about that assessment. They are independent."
    )


class ReplayCriticalityFactorRow(BaseModel):
    factor: str
    value: float
    weight: float
    weighted_contribution: float
    band: str


class ReplayClaimCriticality(BaseModel):
    claim: str                          # redacted
    criticality: float
    signals: list[str] = Field(default_factory=list)


class ReplayCriticality(BaseModel):
    action_criticality: float
    band: str
    dominant_factors: list[str] = Field(default_factory=list)
    max_claim_criticality: float
    factors: list[ReplayCriticalityFactorRow] = Field(default_factory=list)
    claim_criticalities: list[ReplayClaimCriticality] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    explanation: str


class ReplayConsequence(BaseModel):
    financial_impact: float
    reversibility: float
    sensitivity: float
    blast_radius: float
    action_automation: float
    consequence_score: float
    severity_band: str
    dominant_factors: list[str] = Field(default_factory=list)
    explanation: str


class ReplayDecisionStep(BaseModel):
    rule: str
    fired: bool
    tier_before: str | None = None
    tier_after: str | None = None
    effect: str
    detail: str                         # redacted


class ReplayFinalDecision(BaseModel):
    decision: str
    overall_risk: float
    decision_confidence: float
    verification_path: str
    requires_human_review: bool
    triggered_rules: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    explanation: str                    # redacted


class ReplayVerification(BaseModel):
    verification_path: str
    deep_trigger_reasons: list[str] = Field(default_factory=list)
    reason_for_deep_verification: str = ""
    deep_was_forced: bool = False
    preliminary_risk: float | None = None
    preliminary_confidence: float | None = None
    final_risk: float | None = None
    final_confidence: float | None = None
    disagreement_score: float | None = None
    disagreement_breakdown: DisagreementBreakdown | None = None
    evidence_available: bool | None = None
    fast_path_latency_ms: float | None = None
    deep_path_latency_ms: float | None = None
    total_verification_latency_ms: float | None = None


class ReplayLatency(BaseModel):
    total_pipeline_latency_ms: float | None = None
    verification_latency_ms: float | None = None
    fast_path_latency_ms: float | None = None
    deep_path_latency_ms: float | None = None
    performance_ms: float | None = None
    responsibility_ms: float | None = None
    cost_ms: float | None = None
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)


class ReplayBaselineOutcome(BaseModel):
    simulated: bool = True
    label: str = "SIMULATED BASELINE"
    narrative: str
    controlplane_intervention: str


class IncidentReplay(BaseModel):
    found: bool = True
    replay_note: str = (
        "Reconstructed from the stored decision trace. Detectors, fusion, "
        "consequence, criticality and policy were NOT re-run; every value is "
        "copied from what actually happened."
    )

    interaction: ReplayInteraction
    verification: ReplayVerification
    risk_signals: ReplayRiskSignals
    claims: list[ReplayClaim] = Field(default_factory=list)
    confidence: ReplayConfidence
    criticality: ReplayCriticality
    consequence: ReplayConsequence
    decision_path: list[ReplayDecisionStep] = Field(default_factory=list)
    tier_transitions: list[str] = Field(default_factory=list)
    final_decision: ReplayFinalDecision
    explanation: str                    # redacted top-level "why"
    latency: ReplayLatency
    baseline_outcome: ReplayBaselineOutcome


class IncidentReplayNotFound(BaseModel):
    found: bool = False
    interaction_id: str
    message: str


# ------------------------------------------------------------------ builder


def _stage(trace: DecisionTrace, key: str) -> float | None:
    value = trace.stage_latency_ms.get(key)
    return round(value, 3) if isinstance(value, (int, float)) else None


def _baseline(trace: DecisionTrace) -> ReplayBaselineOutcome:
    fd = trace.final_decision
    action = trace.action_type
    automation = trace.consequence.factors.action_automation
    decision = fd.decision.value

    if decision == "ALLOW":
        narrative = (
            f"Without an independent oversight layer, this '{action}' interaction "
            "would have proceeded on the upstream application's normal execution "
            "path — the same outcome ControlPlane reached here."
        )
        intervention = "None — ControlPlane also allowed the response."
    else:
        proceeded = (
            "would have executed automatically"
            if automation >= 0.8
            else "would have proceeded to execution"
        )
        narrative = (
            f"Without an independent oversight layer, this '{action}' interaction "
            f"{proceeded} on the upstream application's normal execution path, "
            f"including any downstream effects, despite the risk signals below."
        )
        if decision in ("HUMAN_REVIEW", "BLOCK"):
            intervention = (
                f"ControlPlane intervened: {decision} "
                "(the response/action was withheld pending human review)."
            )
        else:
            intervention = (
                f"ControlPlane intervened: {decision} "
                "(the response was released with an added verification/annotation step)."
            )
    return ReplayBaselineOutcome(narrative=narrative, controlplane_intervention=intervention)


def build_replay(trace: DecisionTrace) -> IncidentReplay:
    """Reconstruct an :class:`IncidentReplay` from a stored ``DecisionTrace``."""
    red = _Redactor(trace)
    fd = trace.final_decision
    perf = trace.performance
    resp = trace.responsibility
    cost = trace.cost
    fusion = trace.fusion
    crit = trace.criticality
    cons = trace.consequence
    policy = trace.policy
    ver = trace.verification

    interaction = ReplayInteraction(
        interaction_id=trace.interaction_id,
        timestamp=trace.timestamp,
        application=trace.application,
        action_type=trace.action_type,
        model=None,
        response=red.text(resp.redacted_response),
    )

    risk_signals = ReplayRiskSignals(
        performance_risk=fd.performance_risk,
        responsibility_risk=fd.responsibility_risk,
        cost_risk=fd.cost_risk,
        overall_risk=fd.overall_risk,
        weighted_only_risk=getattr(fusion, "weighted_only_risk", None),
        criticality_weighted_performance_risk=trace.criticality_weighted_performance_risk,
        dominant_dimension=getattr(fusion, "dominant_dimension", None),
        multi_risk=bool(getattr(fusion, "multi_risk", False)),
        performance_status=perf.status.value,
        responsibility_reason_codes=list(resp.reason_codes),
        cost_anomaly_types=list(cost.anomaly_types),
        cost_triggered_dimensions=list(cost.triggered_dimensions),
        reason_codes=list(fd.reason_codes),
        performance_explanation=red.text(perf.explanation),
        responsibility_explanation=red.text(resp.explanation),
        cost_explanation=red.text(cost.explanation),
        fusion_explanation=red.text(fusion.explanation),
    )

    claims = [
        ReplayClaim(
            claim=red.text(c.claim),
            status=c.status.value,
            claim_risk=c.claim_risk,
            evidence_strength=c.evidence_strength,
            retrieval_similarity=c.retrieval_similarity,
            nli_label=c.nli_label.value if c.nli_label else None,
            nli_confidence=c.nli_confidence,
            top_evidence=red.text(c.top_evidence[0].text) if c.top_evidence else None,
        )
        for c in perf.claim_results
    ]

    confidence = ReplayConfidence(
        performance_confidence=perf.confidence,
        performance_uncertainty=getattr(perf, "uncertainty", round(1 - perf.confidence, 4)),
        performance_evidence_quality=getattr(perf, "evidence_quality", 0.0),
        verification_confidence=perf.verification_confidence,
        fused_confidence=fusion.confidence,
        fused_uncertainty=getattr(fusion, "uncertainty", round(1 - fusion.confidence, 4)),
        decision_confidence=fd.decision_confidence,
    )

    criticality = ReplayCriticality(
        action_criticality=crit.action_criticality,
        band=crit.band,
        dominant_factors=list(crit.dominant_factors),
        max_claim_criticality=crit.max_claim_criticality,
        factors=[
            ReplayCriticalityFactorRow(
                factor=f.factor,
                value=f.value,
                weight=f.weight,
                weighted_contribution=f.weighted_contribution,
                band=f.band,
            )
            for f in crit.factors
        ],
        claim_criticalities=[
            ReplayClaimCriticality(
                claim=red.text(cc.claim),
                criticality=cc.criticality,
                signals=list(cc.signals),
            )
            for cc in crit.claim_criticalities
        ],
        reason_codes=list(crit.reason_codes),
        explanation=red.text(crit.explanation),
    )

    consequence = ReplayConsequence(
        financial_impact=cons.factors.financial_impact,
        reversibility=cons.factors.reversibility,
        sensitivity=cons.factors.sensitivity,
        blast_radius=cons.factors.blast_radius,
        action_automation=cons.factors.action_automation,
        consequence_score=cons.consequence_score,
        severity_band=cons.severity_band,
        dominant_factors=list(cons.dominant_factors),
        explanation=red.text(cons.explanation),
    )

    # decision path — from the policy rule trace (source of truth), enriched
    # with the running tier before/after each rule.
    steps: list[ReplayDecisionStep] = []
    transitions: list[str] = []
    prev_tier = "ALLOW"
    for entry in policy.rule_trace:
        after = entry.tier_after.value if entry.tier_after is not None else prev_tier
        steps.append(
            ReplayDecisionStep(
                rule=entry.rule,
                fired=entry.fired,
                tier_before=prev_tier,
                tier_after=after,
                effect=entry.effect,
                detail=red.text(entry.detail),
            )
        )
        if entry.rule == "RISK_BAND" or (entry.fired and after != prev_tier):
            transitions.append(f"{prev_tier} -> {after} ({entry.rule})")
        prev_tier = after

    final_decision = ReplayFinalDecision(
        decision=fd.decision.value,
        overall_risk=fd.overall_risk,
        decision_confidence=fd.decision_confidence,
        verification_path=fd.verification_path,
        requires_human_review=fd.decision.value in ("HUMAN_REVIEW", "BLOCK"),
        triggered_rules=list(fd.triggered_rules),
        reason_codes=list(fd.reason_codes),
        explanation=red.text(fd.explanation),
    )

    verification = ReplayVerification(
        verification_path=trace.verification_path,
        deep_trigger_reasons=list(ver.deep_trigger_reasons) if ver else [],
        reason_for_deep_verification=ver.reason_for_deep_verification if ver else "",
        deep_was_forced=bool(ver.deep_was_forced) if ver else False,
        preliminary_risk=ver.preliminary_risk if ver else None,
        preliminary_confidence=ver.preliminary_confidence if ver else None,
        final_risk=ver.final_risk if ver else None,
        final_confidence=ver.final_confidence if ver else None,
        disagreement_score=ver.disagreement_score if ver else None,
        disagreement_breakdown=ver.disagreement_breakdown if ver else None,
        evidence_available=ver.evidence_available if ver else perf.evidence_available,
        fast_path_latency_ms=ver.fast_path_latency_ms if ver else None,
        deep_path_latency_ms=ver.deep_path_latency_ms if ver else None,
        total_verification_latency_ms=(
            ver.total_verification_latency_ms if ver else None
        ),
    )

    latency = ReplayLatency(
        total_pipeline_latency_ms=round(trace.latency_ms, 3),
        verification_latency_ms=(
            ver.total_verification_latency_ms if ver else None
        ),
        fast_path_latency_ms=_stage(trace, "fast_path_ms"),
        deep_path_latency_ms=_stage(trace, "deep_path_ms"),
        performance_ms=_stage(trace, "performance_ms"),
        responsibility_ms=_stage(trace, "responsibility_ms"),
        cost_ms=_stage(trace, "cost_ms"),
        stage_latency_ms={
            k: round(v, 3) for k, v in trace.stage_latency_ms.items()
        },
    )

    return IncidentReplay(
        interaction=interaction,
        verification=verification,
        risk_signals=risk_signals,
        claims=claims,
        confidence=confidence,
        criticality=criticality,
        consequence=consequence,
        decision_path=steps,
        tier_transitions=transitions,
        final_decision=final_decision,
        explanation=red.text(fd.explanation),
        latency=latency,
        baseline_outcome=_baseline(trace),
    )
