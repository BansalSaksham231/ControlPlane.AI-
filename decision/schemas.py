"""
Data contracts for the Decision Engine.

``FinalDecision`` (from ``data.schemas``) is the compact, API-facing
result. ``DecisionTrace`` is the full auditable bundle: every detector
output, the fusion breakdown, the consequence assessment and the policy
rule trace, plus latency. Nothing here reads ground truth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from consequence.schemas import ConsequenceAssessment
from criticality.schemas import CriticalityAssessment
from data.schemas import FinalDecision, InterventionTier
from detectors.cost.schemas import CostResult
from detectors.performance.schemas import PerformanceResult
from detectors.responsibility.schemas import ResponsibilityResult
from fusion.schemas import FusionResult
from policy.schemas import PolicyDecision, RuleTraceEntry
from verification.schemas import VerificationReport

__all__ = ["FinalDecision", "DecisionTrace", "DecisionDriver", "DecisionPathStep"]


class DecisionDriver(BaseModel):
    """One policy rule that actually moved the intervention tier."""

    rule: str
    effect: str
    detail: str


class DecisionPathStep(BaseModel):
    """One transition in how the intervention tier was reached."""

    rule: str
    from_tier: InterventionTier
    to_tier: InterventionTier
    reason: str


def _mask_matched_text(node: Any) -> None:
    """Recursively replace every finding's ``matched_text`` with its redaction."""
    if isinstance(node, dict):
        if "matched_text" in node and "redacted_text" in node:
            node["matched_text"] = node["redacted_text"]
        for value in node.values():
            _mask_matched_text(value)
    elif isinstance(node, list):
        for item in node:
            _mask_matched_text(item)


class DecisionTrace(BaseModel):
    """Complete, replayable record of one ControlPlane evaluation."""

    interaction_id: str
    timestamp: datetime
    application: str
    action_type: str
    # Production-visible metadata: which model produced the response.
    # Optional / defaulted so traces built by older callers still validate.
    # Purely informational — no decision logic reads this.
    model: str | None = None

    final_decision: FinalDecision

    performance: PerformanceResult
    responsibility: ResponsibilityResult
    cost: CostResult
    criticality: CriticalityAssessment
    fusion: FusionResult
    consequence: ConsequenceAssessment
    policy: PolicyDecision

    session: dict[str, Any] | None = None

    # Round 2 upgrade: performance risk after criticality amplification —
    # this is what actually went into fusion.
    criticality_weighted_performance_risk: float = Field(default=0.0, ge=0, le=1)
    # The policy rules that moved the tier, in order.
    decision_drivers: list[DecisionDriver] = Field(default_factory=list)

    # --- Progressive Verification (Phase 2) ---
    verification_path: str = "DEEP"                     # FAST / DEEP
    verification: VerificationReport | None = None      # how it was verified
    decision_path: list[DecisionPathStep] = Field(default_factory=list)

    pre_session_overall_risk: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    detectors_parallel: bool = False

    def redacted_dump(self) -> dict[str, Any]:
        """
        JSON dump of the full trace with raw PII spans masked.

        The unredacted trace (with ``matched_text``) is kept server-side for
        authorised audit retrieval; anything leaving the service uses this.
        """
        data = self.model_dump(mode="json")
        for section in ("performance", "responsibility", "cost"):
            _mask_matched_text(data.get(section))
        # performance evidence text can also echo response content
        return data

    def audit_summary(self) -> dict[str, Any]:
        """A flat, PII-redacted summary suitable for an audit log line."""
        fd = self.final_decision
        return {
            "interaction_id": self.interaction_id,
            "timestamp": self.timestamp.isoformat(),
            "application": self.application,
            "action_type": self.action_type,
            "decision": fd.decision.value,
            "overall_risk": fd.overall_risk,
            "performance_risk": fd.performance_risk,
            "responsibility_risk": fd.responsibility_risk,
            "cost_risk": fd.cost_risk,
            "consequence_score": self.consequence.consequence_score,
            "decision_confidence": fd.decision_confidence,
            "triggered_rules": list(fd.triggered_rules),
            "reason_codes": list(fd.reason_codes),
            "decision_confidence_band": (
                "high" if fd.decision_confidence >= 0.65
                else "low" if fd.decision_confidence < 0.45
                else "medium"
            ),
            "action_criticality": self.criticality.action_criticality,
            "decision_drivers": [d.rule for d in self.decision_drivers],
            "decision_path": [
                f"{s.from_tier.value}->{s.to_tier.value} ({s.rule})"
                for s in self.decision_path
            ],
            "verification_path": self.verification_path,
            "verification_latency_ms": (
                self.verification.total_verification_latency_ms
                if self.verification is not None
                else None
            ),
            "performance_status": self.performance.status.value,
            "responsibility_findings": [
                {
                    "category": f.category.value,
                    "subtype": f.subtype,
                    "severity": f.severity.value,
                    "redacted_text": f.redacted_text,
                }
                for f in self.responsibility.findings
            ],
            "cost_anomalies": list(self.cost.triggered_dimensions),
            "explanation": fd.explanation,
        }
