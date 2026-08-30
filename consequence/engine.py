"""
Consequence Engine.

Derives the five-factor consequence model from an interaction's
structured action metadata:

    financial_impact   scaled from action_amount_inr
    reversibility      how hard the action is to undo (by action type)
    sensitivity        how sensitive the action/data is (by action type)
    blast_radius       scaled from affected_entities
    action_automation  how autonomously the action executes (by action type)

The weighted sum (weights from ``config/settings.yaml``) is the
``consequence_score`` in [0, 1]. This is severity if the AI is wrong —
NOT the probability that it is wrong. Ground truth is never read.

The per-action tables here are kept deliberately identical to
``data.generator.compute_consequence_factors`` so the engine and the
evaluation dataset agree; ``tests/test_consequence.py`` locks that in.
"""

from __future__ import annotations

from typing import Any

from common.timing import clamp01
from consequence.schemas import (
    ConsequenceAssessment,
    ConsequenceContribution,
    ConsequenceFactors,
)
from data.schemas import ActionType, Interaction
from settings import load_settings

ENGINE_NAME = "consequence"

_REVERSIBILITY_BY_ACTION = {
    "information": 0.05,
    "refund": 0.30,
    "account_update": 0.40,
    "account_cancellation": 0.85,
    "external_communication": 0.90,
    "recommendation": 0.50,
}

_SENSITIVITY_BY_ACTION = {
    "information": 0.20,
    "refund": 0.40,
    "account_update": 0.50,
    "account_cancellation": 0.70,
    "external_communication": 0.60,
    "recommendation": 0.50,
}

_AUTOMATION_BY_ACTION = {
    "information": 0.90,
    "refund": 0.60,
    "account_update": 0.60,
    "account_cancellation": 0.50,
    "external_communication": 0.80,
    "recommendation": 0.70,
}

_FINANCIAL_SCALE_INR = 500_000.0
_BLAST_SCALE_ENTITIES = 100.0

_FACTOR_LABELS = {
    "financial_impact": "financial impact",
    "reversibility": "irreversibility",
    "sensitivity": "sensitivity",
    "blast_radius": "blast radius",
    "action_automation": "automation",
}


class ConsequenceEngine:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        settings = config if config is not None else load_settings()
        self._weights: dict[str, float] = dict(settings["consequence_weights"])

    # ------------------------------------------------------------------

    def assess(self, interaction: Interaction) -> ConsequenceAssessment:
        action_value = (
            interaction.action_type.value
            if isinstance(interaction.action_type, ActionType)
            else str(interaction.action_type)
        )

        values = {
            "financial_impact": clamp01(interaction.action_amount_inr / _FINANCIAL_SCALE_INR),
            "reversibility": _REVERSIBILITY_BY_ACTION.get(action_value, 0.5),
            "sensitivity": _SENSITIVITY_BY_ACTION.get(action_value, 0.5),
            "blast_radius": clamp01(interaction.affected_entities / _BLAST_SCALE_ENTITIES),
            "action_automation": _AUTOMATION_BY_ACTION.get(action_value, 0.5),
        }

        contributions: list[ConsequenceContribution] = []
        score = 0.0
        for factor, value in values.items():
            weight = float(self._weights[factor])
            weighted = value * weight
            score += weighted
            contributions.append(
                ConsequenceContribution(
                    factor=factor,
                    value=round(value, 4),
                    weight=round(weight, 4),
                    weighted_contribution=round(weighted, 4),
                )
            )
        score = clamp01(score)

        factors = ConsequenceFactors(
            financial_impact=round(values["financial_impact"], 4),
            reversibility=round(values["reversibility"], 4),
            sensitivity=round(values["sensitivity"], 4),
            blast_radius=round(values["blast_radius"], 4),
            action_automation=round(values["action_automation"], 4),
            consequence_score=round(score, 4),
        )

        ranked = sorted(
            contributions, key=lambda c: c.weighted_contribution, reverse=True
        )
        dominant = [c.factor for c in ranked[:2] if c.weighted_contribution > 0]
        band = self._band(score)
        explanation = self._explain(action_value, band, score, ranked, interaction)

        return ConsequenceAssessment(
            factors=factors,
            consequence_score=round(score, 4),
            severity_band=band,
            contributions=contributions,
            dominant_factors=dominant,
            explanation=explanation,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _band(score: float) -> str:
        if score >= 0.66:
            return "high"
        if score >= 0.4:
            return "moderate"
        return "low"

    @staticmethod
    def _explain(
        action_value: str,
        band: str,
        score: float,
        ranked: list[ConsequenceContribution],
        interaction: Interaction,
    ) -> str:
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        pieces = [
            f"If this '{action_value}' action is wrong, the consequence severity is "
            f"{band} (score {score:.2f})."
        ]
        detail = (
            f"Largest driver: {_FACTOR_LABELS[top.factor]} "
            f"({top.value:.2f} x weight {top.weight:.2f})"
        )
        if second is not None and second.weighted_contribution > 0:
            detail += (
                f"; then {_FACTOR_LABELS[second.factor]} "
                f"({second.value:.2f} x weight {second.weight:.2f})"
            )
        pieces.append(detail + ".")
        if interaction.action_amount_inr > 0:
            pieces.append(f"Action amount: ₹{interaction.action_amount_inr:,.2f}.")
        if interaction.affected_entities > 1:
            pieces.append(f"Affects {interaction.affected_entities} entities.")
        return " ".join(pieces)


def assess_consequence(
    interaction: Interaction, config: dict[str, Any] | None = None
) -> ConsequenceAssessment:
    """Convenience one-shot wrapper."""
    return ConsequenceEngine(config=config).assess(interaction)
