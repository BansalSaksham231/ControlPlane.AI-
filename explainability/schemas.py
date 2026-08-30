"""
Presentation-oriented explanation contract  (Phase 6 — Explainability, Step 1).

This module defines a **view** over an already-computed
:class:`decision.schemas.DecisionTrace`. It exists so a future UI can
render the full ControlPlane reasoning chain to a human judge without
re-deriving anything.

Hard rules for this contract
----------------------------
* It NEVER runs a detector, the fusion engine, the policy engine, the
  consequence / criticality engine or the verification router.
* It NEVER computes a new risk or confidence score — every number is a
  verbatim copy of a value already on the trace / incident replay.
* It NEVER contains ground-truth or evaluation-only labels — those are not
  even present on a ``DecisionTrace``.
* It NEVER contains raw PII: there is no raw-span field, no raw finding, no
  unredacted response. Free-text fields are expected to be
  populated from the **already-redacted** text on the trace / the
  :class:`decision.replay.IncidentReplay` (which the responsibility
  detector produced), not from raw sources.

Every model sets ``extra="forbid"`` so a forbidden field cannot be
smuggled in by a future caller.

Architecture::

    Interaction -> ControlPlane pipeline -> DecisionTrace -> Explainability View -> UI
                                                             ^^^^^^^^^^^^^^^^^^^^
                                                             (this contract only)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from data.schemas import InterventionTier
from verification.schemas import VerificationPath

__all__ = [
    "RiskDimensionExplanation",
    "EvidenceExplanation",
    "ConsequenceFactorRow",
    "ConsequenceExplanation",
    "CriticalityExplanation",
    "PolicyRuleExplanation",
    "DecisionPathExplanation",
    "VerificationExplanation",
    "RiskSummary",
    "ConfidenceSummary",
    "HumanReviewExplanation",
    "CounterfactualExplanation",
    "CriticalEventRow",
    "SessionMemoryExplanation",
    "ExplainabilitySummary",
]

_RISK_CONFIDENCE_NOTE = (
    "RISK is how dangerous the interaction looks; CONFIDENCE is how sure "
    "ControlPlane is about that assessment. They are independent axes."
)


class _ExplainModel(BaseModel):
    """Base for every explainability contract — closed to unknown fields."""

    model_config = ConfigDict(extra="forbid")


# ======================================================================
# per-dimension / per-row views
# ======================================================================


class RiskDimensionExplanation(_ExplainModel):
    """One risk dimension's contribution to the fused overall risk."""

    dimension: str                                   # performance / responsibility / cost
    risk: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    weighted_contribution: float = Field(ge=0, le=1)
    status: str | None = None                         # e.g. performance status, else None
    is_dominant: bool = False
    explanation: str = ""                             # redacted detector explanation


class EvidenceExplanation(_ExplainModel):
    """What the evidence said about one extracted claim (no raw PII)."""

    claim: str                                       # redacted
    status: str                                      # SUPPORTED / CONTRADICTED / NEUTRAL / NO_EVIDENCE
    retrieval_similarity: float = Field(ge=0, le=1)
    nli_label: str | None = None
    nli_confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_strength: float = Field(ge=0, le=1)
    claim_risk: float = Field(ge=0, le=1)
    supporting_evidence: str | None = None            # redacted, rank-1 chunk only


class ConsequenceFactorRow(_ExplainModel):
    factor: str
    value: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    weighted_contribution: float = Field(ge=0, le=1)


class ConsequenceExplanation(_ExplainModel):
    """How serious the outcome would be *if the AI is wrong* (not a risk)."""

    consequence_score: float = Field(ge=0, le=1)
    severity_band: str
    dominant_factors: list[str] = Field(default_factory=list)
    factors: list[ConsequenceFactorRow] = Field(default_factory=list)
    explanation: str = ""                             # redacted


class CriticalityExplanation(_ExplainModel):
    """"How much does it matter if this response is wrong?" (action criticality)."""

    action_criticality: float = Field(ge=0, le=1)
    band: str
    dominant_factors: list[str] = Field(default_factory=list)
    max_claim_criticality: float = Field(default=0.0, ge=0, le=1)
    explanation: str = ""                             # redacted


class PolicyRuleExplanation(_ExplainModel):
    """One entry of the policy rule trace, with the tier before/after it."""

    rule: str
    fired: bool
    tier_before: InterventionTier | None = None
    tier_after: InterventionTier | None = None
    changed_tier: bool = False
    effect: str = ""
    detail: str = ""                                  # redacted


class DecisionPathExplanation(_ExplainModel):
    """One tier transition on the way to the final intervention tier."""

    rule: str
    from_tier: InterventionTier
    to_tier: InterventionTier
    reason: str = ""                                  # redacted


class VerificationExplanation(_ExplainModel):
    """Whether FAST or DEEP verification ran, and why."""

    verification_path: VerificationPath              # FAST / DEEP (validated)
    used_deep: bool
    deep_was_forced: bool = False
    deep_trigger_reasons: list[str] = Field(default_factory=list)
    reason_for_deep_verification: str = ""

    preliminary_risk: float | None = Field(default=None, ge=0, le=1)
    preliminary_confidence: float | None = Field(default=None, ge=0, le=1)
    final_risk: float | None = Field(default=None, ge=0, le=1)
    final_confidence: float | None = Field(default=None, ge=0, le=1)
    disagreement_score: float | None = Field(default=None, ge=0, le=1)
    evidence_available: bool | None = None

    explanation: str = ""


# ======================================================================
# summaries
# ======================================================================


class RiskSummary(_ExplainModel):
    """The fused-risk picture — copied verbatim from ``trace.fusion``."""

    overall_risk: float = Field(ge=0, le=1)
    dominant_dimension: str
    dominant_risk: float = Field(ge=0, le=1)
    multi_risk: bool = False
    weighted_only_risk: float | None = Field(default=None, ge=0, le=1)
    criticality_weighted_performance_risk: float | None = Field(default=None, ge=0, le=1)
    severity_rule_applied: bool = False
    severity_floor_applied: bool = False


class ConfidenceSummary(_ExplainModel):
    """How sure ControlPlane is — deliberately separate from risk."""

    decision_confidence: float = Field(ge=0, le=1)
    fused_confidence: float = Field(ge=0, le=1)
    fused_uncertainty: float = Field(ge=0, le=1)
    performance_confidence: float = Field(ge=0, le=1)
    verification_confidence: float = Field(ge=0, le=1)
    note: str = _RISK_CONFIDENCE_NOTE


class HumanReviewExplanation(_ExplainModel):
    """Did a human-review / block condition trigger, and what caused it?"""

    required: bool
    decision: InterventionTier
    triggering_conditions: list[str] = Field(default_factory=list)
    explanation: str = ""


class CounterfactualExplanation(_ExplainModel):
    """
    "What would have happened without a particular rule?"

    This contract only *presents* a counterfactual result. The result
    itself is produced by the existing policy-simulation engine — the
    explainability layer never recomputes it.
    """

    rule_removed: str
    original_decision: InterventionTier
    counterfactual_decision: InterventionTier
    decision_changed: bool
    rules_no_longer_firing: list[str] = Field(default_factory=list)
    reason_codes_removed: list[str] = Field(default_factory=list)
    summary: str = ""
    simulated: bool = True
    note: str = (
        "Produced by the policy-simulation engine; the explainability view "
        "only displays it and does not recompute any risk."
    )


# ======================================================================
# multi-turn session memory
# ======================================================================


class CriticalEventRow(_ExplainModel):
    """One earlier turn in this session that crossed a critical boundary."""

    turn_index: int = Field(ge=1)
    trigger: str                                     # BLOCK | CRITICAL_PII | SEVERE_TOXICITY
    decision: str
    risk_at_event: float = Field(ge=0, le=1)


class SessionMemoryExplanation(_ExplainModel):
    """
    The stateful multi-turn memory that fed this decision — the session's
    :class:`session.schemas.ContextualSnapshot` as it stood at the start of the
    turn. Every value is a verbatim copy from ``trace.session``; the
    explainability layer computes nothing here. Redacted PII keys only.
    """

    turns_recorded: int = 0
    has_critical_history: bool = False

    critical_floor: float = Field(default=0.0, ge=0, le=1)
    # Did the non-decaying floor actually raise THIS interaction's scrutiny?
    critical_floor_applied: bool = False

    peak_performance_risk: float = Field(default=0.0, ge=0, le=1)
    peak_responsibility_risk: float = Field(default=0.0, ge=0, le=1)
    peak_cost_risk: float = Field(default=0.0, ge=0, le=1)

    pii_entity_keys: list[str] = Field(default_factory=list)     # "<subtype>:<redacted>"
    reason_code_counts: dict[str, int] = Field(default_factory=dict)
    critical_events: list[CriticalEventRow] = Field(default_factory=list)

    explanation: str = ""


# ======================================================================
# top-level contract
# ======================================================================


class ExplainabilitySummary(_ExplainModel):
    """
    The single object a UI needs to explain one ControlPlane decision.

    It is a pure view of a ``DecisionTrace`` (plus, optionally, already-
    computed counterfactual results). Nothing here is recomputed.
    """

    # 1. WHAT decision was made?
    decision: InterventionTier
    # 2. HOW risky?   3. HOW confident?
    overall_risk: float = Field(ge=0, le=1)
    decision_confidence: float = Field(ge=0, le=1)
    # 6. FAST or DEEP?
    verification_path: VerificationPath
    # 10. human-review condition?
    human_review_required: bool

    # 4. WHY  (canonical reason codes + the rules that moved the tier)
    primary_reasons: list[str] = Field(default_factory=list)
    decision_drivers: list[str] = Field(default_factory=list)

    # 5. WHICH risk dimensions contributed?
    risk_summary: RiskSummary
    confidence_summary: ConfidenceSummary
    risk_dimensions: list[RiskDimensionExplanation] = Field(default_factory=list)

    # 6. verification detail
    verification_summary: VerificationExplanation

    # 7. WHAT evidence supported / contradicted the response?
    evidence: list[EvidenceExplanation] = Field(default_factory=list)

    # 8. WHAT consequence / criticality changed how serious the action is?
    consequence_summary: ConsequenceExplanation
    criticality_summary: CriticalityExplanation

    # 9. WHICH policy rules changed the tier?   + the tier path
    policy_rules: list[PolicyRuleExplanation] = Field(default_factory=list)
    decision_path: list[DecisionPathExplanation] = Field(default_factory=list)

    # 10. human review detail
    human_review: HumanReviewExplanation

    # 11. WHAT would have happened without a particular rule?
    counterfactuals: list[CounterfactualExplanation] = Field(default_factory=list)

    # 12. multi-turn session memory (None for a single-turn / stateless trace)
    session_memory: SessionMemoryExplanation | None = None

    # top-level redacted "why" sentence
    explanation: str

    source: str = "DecisionTrace"
    notes: list[str] = Field(
        default_factory=lambda: [
            "This is a presentation view of a stored decision trace. No "
            "detector, fusion, policy, consequence, criticality or "
            "verification computation was re-run.",
            "No ground-truth label is read; the view does not judge whether "
            "the decision was correct.",
            "All free text is the already-redacted text from the trace — "
            "raw PII spans are never included.",
            _RISK_CONFIDENCE_NOTE,
        ]
    )
