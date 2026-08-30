"""Data contracts for the Policy Engine."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from data.schemas import InterventionTier

TIER_ORDER: list[InterventionTier] = [
    InterventionTier.ALLOW,
    InterventionTier.ANNOTATE,
    InterventionTier.VERIFY,
    InterventionTier.HUMAN_REVIEW,
    InterventionTier.BLOCK,
]

TIER_RANK: dict[InterventionTier, int] = {tier: i for i, tier in enumerate(TIER_ORDER)}


def tier_max(*tiers: InterventionTier) -> InterventionTier:
    """Return the most severe tier among the arguments."""
    return max(tiers, key=lambda t: TIER_RANK[t])


def tier_escalate(tier: InterventionTier, steps: int) -> InterventionTier:
    """Move ``tier`` up ``steps`` positions, clamped to BLOCK."""
    return TIER_ORDER[min(TIER_RANK[tier] + max(0, steps), len(TIER_ORDER) - 1)]


class PolicyInput(BaseModel):
    """Everything the policy engine needs — all production-derived, no ground truth."""

    application: str
    action_type: str

    overall_risk: float = Field(ge=0, le=1)
    performance_risk: float = Field(ge=0, le=1)
    responsibility_risk: float = Field(ge=0, le=1)
    cost_risk: float = Field(ge=0, le=1)
    consequence_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)

    performance_status: str = "UNVERIFIED"
    dominant_dimension: str = "performance"
    multi_risk: bool = False

    financial_impact: float = Field(default=0.0, ge=0, le=1)
    irreversibility: float = Field(default=0.0, ge=0, le=1)
    blast_radius: float = Field(default=0.0, ge=0, le=1)
    action_automation: float = Field(default=0.0, ge=0, le=1)
    action_criticality: float = Field(default=0.0, ge=0, le=1)
    is_external_action: bool = False

    contains_critical_pii: bool = False
    critical_pii_types: list[str] = Field(default_factory=list)
    toxicity_risk: float = Field(default=0.0, ge=0, le=1)
    responsibility_reason_codes: list[str] = Field(default_factory=list)

    # When False, the confidence-aware policy rules are skipped (used by
    # the ablation study to isolate the value of risk+confidence).
    confidence_aware: bool = True

    @classmethod
    def from_results(
        cls,
        *,
        application: str,
        action_type: str,
        fusion: Any,
        consequence: Any,
        performance: Any = None,
        responsibility: Any = None,
        criticality: Any = None,
        confidence_aware: bool = True,
    ) -> "PolicyInput":
        factors = getattr(consequence, "factors", None)
        return cls(
            application=application,
            action_type=action_type,
            overall_risk=getattr(fusion, "overall_risk", 0.0),
            performance_risk=getattr(fusion, "performance_risk", 0.0),
            responsibility_risk=getattr(fusion, "responsibility_risk", 0.0),
            cost_risk=getattr(fusion, "cost_risk", 0.0),
            consequence_score=getattr(consequence, "consequence_score", 0.0),
            financial_impact=getattr(factors, "financial_impact", 0.0),
            irreversibility=getattr(factors, "reversibility", 0.0),
            blast_radius=getattr(factors, "blast_radius", 0.0),
            action_automation=getattr(factors, "action_automation", 0.0),
            action_criticality=getattr(criticality, "action_criticality", 0.0),
            is_external_action=action_type in ("external_communication", "account_cancellation"),
            confidence=getattr(fusion, "confidence", 0.5),
            dominant_dimension=getattr(fusion, "dominant_dimension", "performance"),
            multi_risk=bool(getattr(fusion, "multi_risk", False)),
            performance_status=(
                getattr(getattr(performance, "status", None), "value", None)
                or "UNVERIFIED"
            ),
            contains_critical_pii=bool(
                getattr(responsibility, "contains_critical_pii", False)
            ),
            critical_pii_types=list(
                getattr(responsibility, "critical_pii_types", []) or []
            ),
            toxicity_risk=float(getattr(responsibility, "toxicity_risk", 0.0) or 0.0),
            responsibility_reason_codes=list(
                getattr(responsibility, "reason_codes", []) or []
            ),
            confidence_aware=confidence_aware,
        )


class RuleTraceEntry(BaseModel):
    rule: str
    fired: bool
    effect: str
    detail: str
    # Intervention tier after this rule was applied (Phase 2: source for
    # DecisionTrace.decision_path). Optional / defaulted for compatibility.
    tier_after: InterventionTier | None = None


class PolicyDecision(BaseModel):
    """Policy engine output — a *proposed* tier plus the full rule trace."""

    application: str
    proposed_tier: InterventionTier
    base_tier: InterventionTier
    requires_human_review: bool

    triggered_rules: list[str]
    reason_codes: list[str] = Field(default_factory=list)
    rule_trace: list[RuleTraceEntry]

    explanation: str

    def drivers(self) -> list[RuleTraceEntry]:
        """The rule-trace entries that actually moved the tier."""
        return [e for e in self.rule_trace if e.fired]
