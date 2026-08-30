"""
Policy simulation & counterfactual analysis.

Both run the *real* pipeline (``DecisionEngine.evaluate``) — nothing here
fabricates an alternate decision.

* ``simulate_policies`` — the same interaction under different application
  policy profiles. Demonstrates why per-application governance exists.
* ``compare_decisions`` — the same interaction with a few production-
  visible fields changed ("what if the refund were ₹100?"). Shows which
  rules stopped / started firing and why the tier moved.

Ground-truth / evaluation fields can never be modified — the whitelist
below is the only surface a counterfactual may touch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from data.schemas import ActionType, Application, Interaction, ModelName, UserType
from decision.engine import DecisionEngine

# Production-visible fields a counterfactual is allowed to change.
_MUTABLE_FIELDS: dict[str, Any] = {
    "response": str,
    "context": str,
    "prompt": str,
    "tokens_in": int,
    "tokens_out": int,
    "latency_ms": float,
    "tool_calls": int,
    "retry_count": int,
    "action_type": ActionType,
    "action_amount_inr": float,
    "affected_entities": int,
    "application": Application,
    "user_type": UserType,
    "model": ModelName,
}

_FORBIDDEN_FIELDS = {
    "ground_truth_hallucination", "ground_truth_pii", "ground_truth_toxicity",
    "ground_truth_bias", "ground_truth_cost_anomaly", "ground_truth_performance_risk",
    "ground_truth_responsibility_risk", "ground_truth_cost_risk", "expected_decision",
    "human_review_expected", "final_outcome",
}

_SIM_TS = datetime(2026, 8, 21, 12, 0, 0)


class PolicyOutcome(BaseModel):
    profile: str
    decision: str
    overall_risk: float
    decision_confidence: float
    triggered_rules: list[str]
    reason_codes: list[str]
    requires_human_review: bool
    explanation: str


class PolicySimulation(BaseModel):
    interaction_id: str
    base_application: str
    outcomes: list[PolicyOutcome]
    differs: bool
    summary: str


class CounterfactualResult(BaseModel):
    interaction_id: str
    changed_fields: dict[str, Any]
    rejected_fields: list[str]

    original_decision: str
    counterfactual_decision: str
    tier_changed: bool

    original_overall_risk: float
    counterfactual_overall_risk: float

    rules_added: list[str]
    rules_removed: list[str]
    reason_codes_added: list[str]
    reason_codes_removed: list[str]

    original_explanation: str
    counterfactual_explanation: str
    summary: str


# ------------------------------------------------------------------ policy sim


def simulate_policies(
    engine: DecisionEngine,
    interaction: Interaction,
    profiles: list[str],
) -> PolicySimulation:
    outcomes: list[PolicyOutcome] = []
    for profile in profiles:
        trace = engine.evaluate(
            interaction, timestamp=_SIM_TS, record_session=False, policy_profile=profile
        )
        fd = trace.final_decision
        outcomes.append(
            PolicyOutcome(
                profile=profile,
                decision=fd.decision.value,
                overall_risk=fd.overall_risk,
                decision_confidence=fd.decision_confidence,
                triggered_rules=list(fd.triggered_rules),
                reason_codes=list(fd.reason_codes),
                requires_human_review=fd.decision.value in ("HUMAN_REVIEW", "BLOCK"),
                explanation=trace.policy.explanation,
            )
        )

    distinct = {o.decision for o in outcomes}
    differs = len(distinct) > 1
    if differs:
        pairs = ", ".join(f"{o.profile} -> {o.decision}" for o in outcomes)
        summary = (
            f"The same interaction is governed differently by application: {pairs}. "
            "Stricter profiles (e.g. decision_support) escalate the same risk further."
        )
    else:
        summary = (
            f"All simulated profiles reach {next(iter(distinct))} for this "
            "interaction — the signal is unambiguous enough that policy tolerance "
            "does not change the outcome."
        )
    return PolicySimulation(
        interaction_id=interaction.interaction_id,
        base_application=interaction.application.value,
        outcomes=outcomes,
        differs=differs,
        summary=summary,
    )


# ------------------------------------------------------------------ counterfactual


def _coerce(field: str, value: Any) -> Any:
    kind = _MUTABLE_FIELDS[field]
    if kind in (Application, UserType, ModelName, ActionType):
        return kind(value)
    return kind(value)


def build_counterfactual(
    interaction: Interaction, modified_fields: dict[str, Any]
) -> tuple[Interaction, dict[str, Any], list[str]]:
    """Return (new_interaction, applied_changes, rejected_field_names)."""
    applied: dict[str, Any] = {}
    rejected: list[str] = []
    for field, value in (modified_fields or {}).items():
        if field in _FORBIDDEN_FIELDS or field not in _MUTABLE_FIELDS:
            rejected.append(field)
            continue
        try:
            applied[field] = _coerce(field, value)
        except (ValueError, TypeError):
            rejected.append(field)
    updated = interaction.model_copy(
        update={
            **applied,
            "interaction_id": f"{interaction.interaction_id}-CF",
        }
    )
    return updated, applied, rejected


def compare_decisions(
    engine: DecisionEngine,
    interaction: Interaction,
    modified_fields: dict[str, Any],
) -> CounterfactualResult:
    cf_interaction, applied, rejected = build_counterfactual(interaction, modified_fields)

    original = engine.evaluate(interaction, timestamp=_SIM_TS, record_session=False)
    counterfactual = engine.evaluate(cf_interaction, timestamp=_SIM_TS, record_session=False)

    o_fd, c_fd = original.final_decision, counterfactual.final_decision
    o_rules, c_rules = set(o_fd.triggered_rules), set(c_fd.triggered_rules)
    o_codes, c_codes = set(o_fd.reason_codes), set(c_fd.reason_codes)

    tier_changed = o_fd.decision != c_fd.decision
    display = {
        k: (v.value if hasattr(v, "value") else v) for k, v in applied.items()
    }
    if tier_changed:
        direction = "reduced" if _rank(c_fd.decision) < _rank(o_fd.decision) else "raised"
        summary = (
            f"Changing {', '.join(display)} {direction} the intervention from "
            f"{o_fd.decision.value} to {c_fd.decision.value} "
            f"(risk {o_fd.overall_risk:.2f} -> {c_fd.overall_risk:.2f})."
        )
    else:
        summary = (
            f"Changing {', '.join(display) or 'nothing'} did not change the "
            f"intervention ({o_fd.decision.value}); "
            f"risk {o_fd.overall_risk:.2f} -> {c_fd.overall_risk:.2f}."
        )

    return CounterfactualResult(
        interaction_id=interaction.interaction_id,
        changed_fields=display,
        rejected_fields=rejected,
        original_decision=o_fd.decision.value,
        counterfactual_decision=c_fd.decision.value,
        tier_changed=tier_changed,
        original_overall_risk=o_fd.overall_risk,
        counterfactual_overall_risk=c_fd.overall_risk,
        rules_added=sorted(c_rules - o_rules),
        rules_removed=sorted(o_rules - c_rules),
        reason_codes_added=sorted(c_codes - o_codes),
        reason_codes_removed=sorted(o_codes - c_codes),
        original_explanation=o_fd.explanation,
        counterfactual_explanation=c_fd.explanation,
        summary=summary,
    )


def _rank(tier: Any) -> int:
    from policy.schemas import TIER_RANK

    return TIER_RANK[tier]
