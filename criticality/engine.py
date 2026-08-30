"""
Claim / Action Criticality engine.

Not all wrong answers matter equally. "The office is in Mumbai" and
"the customer is eligible for a ₹480,000 refund" may both be unsupported,
but the second is far more consequential to get wrong.

``action_criticality`` in [0,1] is a transparent weighted blend of five
normalized factors, all from production-visible fields:

    financial_impact   scaled from action_amount_inr
    irreversibility    from action_type (how hard to undo)
    sensitivity        from action_type
    blast_radius       scaled from affected_entities
    automation         from action_type (how autonomously it executes)

Per-claim criticality additionally scans the *claim text itself* for
money amounts, action verbs and named entities — again production-visible
(it is the model's own output), never ground truth.

This runs AFTER the detectors and BEFORE fusion: the decision engine uses
``action_criticality`` to amplify performance risk (a risky claim that
also matters a lot should weigh more), and the policy engine uses it for
reason codes such as HIGH_FINANCIAL_IMPACT / IRREVERSIBLE_ACTION.
"""

from __future__ import annotations

import re
from typing import Any

from common.reason_codes import ReasonCode, dedupe
from common.timing import clamp01
from consequence.engine import (
    _AUTOMATION_BY_ACTION,
    _REVERSIBILITY_BY_ACTION,
    _SENSITIVITY_BY_ACTION,
)
from data.schemas import ActionType, Interaction
from criticality.schemas import (
    ClaimCriticality,
    CriticalityAssessment,
    CriticalityFactor,
)
from settings import load_settings

ENGINE_NAME = "criticality"

_MONEY_RE = re.compile(r"(?:₹|\$|\brs\.?\b|\binr\b|\busd\b)\s?[\d,]{2,}", re.IGNORECASE)
_BIG_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{5,}\b")
_ENTITY_RE = re.compile(
    r"\b(account|customer|employee|email|phone|card|ssn|aadhaar|address|refund|payment)\b",
    re.IGNORECASE,
)

_FACTOR_LABELS = {
    "financial_impact": "financial impact",
    "irreversibility": "irreversibility",
    "sensitivity": "sensitivity",
    "blast_radius": "blast radius",
    "automation": "automation",
}


def _band(value: float, high: float, moderate: float) -> str:
    if value >= high:
        return "HIGH"
    if value >= moderate:
        return "MEDIUM"
    return "LOW"


class CriticalityEngine:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        settings = config if config is not None else load_settings()
        cfg = settings["criticality"]
        self._weights: dict[str, float] = dict(cfg["factor_weights"])
        self._financial_scale = float(cfg["financial_scale_inr"])
        self._blast_scale = float(cfg["blast_scale_entities"])
        self._band_high = float(cfg["band_high"])
        self._band_moderate = float(cfg["band_moderate"])
        self._fin_trigger = float(cfg["high_financial_impact_trigger"])
        self._irr_trigger = float(cfg["irreversible_action_trigger"])
        self._blast_trigger = float(cfg["high_blast_radius_trigger"])
        self._amplification = float(cfg["performance_amplification"])
        self._action_verbs = {v.lower() for v in cfg["action_verbs"]}

    # ------------------------------------------------------------------

    @property
    def performance_amplification(self) -> float:
        return self._amplification

    def amplify_performance_risk(self, performance_risk: float, criticality: float) -> float:
        """
        eff = risk + (1 - risk) * risk * (criticality - moderate) * amplification

        Amplification only engages once criticality is at least "moderate"
        — a benign informational response is not made riskier just because
        it happens to be wrong. A moderate risk over a critical action is
        pushed materially higher; an already-high risk is nudged toward 1.0.
        """
        risk = clamp01(performance_risk)
        crit = clamp01(criticality)
        if crit <= self._band_moderate:
            return risk
        headroom = (crit - self._band_moderate) / (1.0 - self._band_moderate)
        return clamp01(risk + (1.0 - risk) * risk * headroom * self._amplification)

    # ------------------------------------------------------------------

    def assess(
        self, interaction: Interaction, performance: Any | None = None
    ) -> CriticalityAssessment:
        action_value = (
            interaction.action_type.value
            if isinstance(interaction.action_type, ActionType)
            else str(interaction.action_type)
        )

        values = {
            "financial_impact": clamp01(
                interaction.action_amount_inr / self._financial_scale
            ),
            "irreversibility": _REVERSIBILITY_BY_ACTION.get(action_value, 0.5),
            "sensitivity": _SENSITIVITY_BY_ACTION.get(action_value, 0.5),
            "blast_radius": clamp01(
                interaction.affected_entities / self._blast_scale
            ),
            "automation": _AUTOMATION_BY_ACTION.get(action_value, 0.5),
        }

        factors: list[CriticalityFactor] = []
        score = 0.0
        for name, value in values.items():
            weight = float(self._weights[name])
            weighted = value * weight
            score += weighted
            factors.append(
                CriticalityFactor(
                    factor=name,
                    value=round(value, 4),
                    weight=round(weight, 4),
                    weighted_contribution=round(weighted, 4),
                    band=_band(value, 0.66, 0.4),
                )
            )
        action_criticality = clamp01(score)

        ranked = sorted(factors, key=lambda f: f.weighted_contribution, reverse=True)
        dominant = [f.factor for f in ranked[:2] if f.weighted_contribution > 0]

        claim_criticalities = self._claim_criticalities(performance, action_criticality)
        max_claim = max(
            (c.criticality for c in claim_criticalities), default=0.0
        )

        reason_codes = self._reason_codes(values, action_value)
        band = (
            "high"
            if action_criticality >= self._band_high
            else "moderate"
            if action_criticality >= self._band_moderate
            else "low"
        )
        explanation = self._explain(
            action_value, band, action_criticality, ranked, interaction
        )

        return CriticalityAssessment(
            action_criticality=round(action_criticality, 4),
            band=band,
            factors=factors,
            dominant_factors=dominant,
            claim_criticalities=claim_criticalities,
            max_claim_criticality=round(max_claim, 4),
            reason_codes=reason_codes,
            explanation=explanation,
        )

    # ------------------------------------------------------------------

    def text_criticality(self, text: str) -> tuple[float, list[str]]:
        """Heuristic importance of a single claim from its own words."""
        signals: list[str] = []
        score = 0.0
        lowered = (text or "").lower()

        if _MONEY_RE.search(text or "") or _BIG_NUMBER_RE.search(text or ""):
            score += 0.55
            signals.append("monetary_amount")
        verbs = [v for v in self._action_verbs if re.search(rf"\b{re.escape(v)}", lowered)]
        if verbs:
            score += 0.35
            signals.append(f"action_verb:{verbs[0]}")
        if _ENTITY_RE.search(text or ""):
            score += 0.20
            signals.append("named_entity")

        return clamp01(score), signals

    def _claim_criticalities(
        self, performance: Any | None, action_criticality: float
    ) -> list[ClaimCriticality]:
        claim_results = getattr(performance, "claim_results", None) or []
        out: list[ClaimCriticality] = []
        for claim_result in claim_results:
            text = getattr(claim_result, "claim", "")
            text_crit, signals = self.text_criticality(text)
            # A claim's criticality blends the action it sits within with
            # what the claim text itself asserts.
            combined = clamp01(0.6 * action_criticality + 0.4 * text_crit)
            if text_crit > 0:
                signals = signals
            else:
                signals = ["inherits_action_criticality"]
            out.append(
                ClaimCriticality(
                    claim=text,
                    criticality=round(combined, 4),
                    signals=signals,
                )
            )
        return out

    def _reason_codes(self, values: dict[str, float], action_value: str) -> list[str]:
        codes: list[ReasonCode] = []
        if values["financial_impact"] >= self._fin_trigger:
            codes.append(ReasonCode.HIGH_FINANCIAL_IMPACT)
        if values["irreversibility"] >= self._irr_trigger:
            codes.append(ReasonCode.IRREVERSIBLE_ACTION)
        if values["blast_radius"] >= self._blast_trigger:
            codes.append(ReasonCode.HIGH_BLAST_RADIUS)
        if values["automation"] >= 0.8 and action_value in (
            "external_communication",
            "recommendation",
            "refund",
            "account_update",
        ):
            codes.append(ReasonCode.AUTOMATED_EXTERNAL_ACTION)
        return dedupe(codes)

    @staticmethod
    def _explain(
        action_value: str,
        band: str,
        score: float,
        ranked: list[CriticalityFactor],
        interaction: Interaction,
    ) -> str:
        top = ranked[0]
        pieces = [
            f"Action criticality {score:.2f} ({band}) for a '{action_value}' action."
        ]
        detail = (
            f"Largest contributor: {_FACTOR_LABELS[top.factor]} "
            f"({top.value:.2f} x weight {top.weight:.2f}, {top.band})"
        )
        if len(ranked) > 1 and ranked[1].weighted_contribution > 0:
            detail += (
                f"; then {_FACTOR_LABELS[ranked[1].factor]} "
                f"({ranked[1].value:.2f}, {ranked[1].band})"
            )
        pieces.append(detail + ".")
        if interaction.action_amount_inr > 0:
            pieces.append(f"Action amount ₹{interaction.action_amount_inr:,.2f}.")
        if interaction.affected_entities > 1:
            pieces.append(f"Affects {interaction.affected_entities} entities.")
        return " ".join(pieces)


def assess_criticality(
    interaction: Interaction,
    performance: Any | None = None,
    config: dict[str, Any] | None = None,
) -> CriticalityAssessment:
    """Convenience one-shot wrapper."""
    return CriticalityEngine(config=config).assess(interaction, performance)
