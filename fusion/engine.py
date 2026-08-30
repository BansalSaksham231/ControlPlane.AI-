"""
Risk Fusion Engine.

Combines the three independent risk dimensions (performance,
responsibility, cost) into a single ``overall_risk`` using a documented,
configurable model — never a silent average.

Model
-----
    weighted   = Σ weight_i * risk_i
    max_dim    = max(risk_i)
    pull       = severity_pull       if max_dim >= severity_trigger
                 severity_pull_low   otherwise
    blended    = (1 - pull) * weighted + pull * max_dim
    if max_dim >= severity_floor_trigger:
        blended = max(blended, severity_floor_value)
    overall    = clamp(blended, 0, 1)

The ``pull`` term is the conservative safety rule: when any single
dimension is severe, the fused risk is pulled toward that dimension so a
real problem is not diluted by low scores elsewhere. All coefficients
live in ``config/settings.yaml`` under ``fusion``.
"""

from __future__ import annotations

from statistics import pstdev
from typing import Any

from common.timing import clamp01
from fusion.schemas import DimensionContribution, FusionResult
from settings import load_settings

ENGINE_NAME = "fusion"

_DIMENSION_LABELS = {
    "performance": "performance/grounding",
    "responsibility": "responsibility",
    "cost": "cost/operational",
}


class RiskFusionEngine:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        settings = config if config is not None else load_settings()
        fcfg = settings["fusion"]
        self._weights: dict[str, float] = dict(fcfg["dimension_weights"])
        self._severity_trigger = float(fcfg["severity_trigger"])
        self._severity_pull = float(fcfg["severity_pull"])
        self._severity_pull_low = float(fcfg["severity_pull_low"])
        self._floor_trigger = float(fcfg["severity_floor_trigger"])
        self._floor_value = float(fcfg["severity_floor_value"])

    # ------------------------------------------------------------------

    def fuse(
        self,
        performance: Any = None,
        responsibility: Any = None,
        cost: Any = None,
    ) -> FusionResult:
        perf_risk = _get(performance, "performance_risk")
        resp_risk = _get(responsibility, "overall_responsibility_risk")
        cost_risk = _get(cost, "cost_risk")

        perf_conf = _get(performance, "confidence", default=0.5)
        resp_conf = _get(responsibility, "confidence", default=0.5)
        cost_conf = _get(cost, "confidence", default=0.5)

        return self.fuse_scores(
            perf_risk,
            resp_risk,
            cost_risk,
            performance_confidence=perf_conf,
            responsibility_confidence=resp_conf,
            cost_confidence=cost_conf,
        )

    def fuse_scores(
        self,
        performance_risk: float,
        responsibility_risk: float,
        cost_risk: float,
        performance_confidence: float = 0.5,
        responsibility_confidence: float = 0.5,
        cost_confidence: float = 0.5,
    ) -> FusionResult:
        risks = {
            "performance": clamp01(performance_risk),
            "responsibility": clamp01(responsibility_risk),
            "cost": clamp01(cost_risk),
        }
        confidences = {
            "performance": clamp01(performance_confidence),
            "responsibility": clamp01(responsibility_confidence),
            "cost": clamp01(cost_confidence),
        }

        weighted = sum(self._weights[d] * risks[d] for d in risks)
        max_dim_name = max(risks, key=risks.get)
        max_dim = risks[max_dim_name]

        severity_rule = max_dim >= self._severity_trigger
        pull = self._severity_pull if severity_rule else self._severity_pull_low
        blended = (1 - pull) * weighted + pull * max_dim

        floor_applied = max_dim >= self._floor_trigger
        if floor_applied:
            blended = max(blended, self._floor_value)

        overall = clamp01(blended)

        breakdown = [
            DimensionContribution(
                dimension=d,
                risk=round(risks[d], 4),
                weight=round(self._weights[d], 4),
                weighted_contribution=round(self._weights[d] * risks[d], 4),
                confidence=round(confidences[d], 4),
            )
            for d in ("performance", "responsibility", "cost")
        ]

        confidence = self._fused_confidence(risks, confidences)
        # "Multi-risk": two or more dimensions elevated at once.
        elevated = sum(1 for r in risks.values() if r >= 0.45)
        multi_risk = elevated >= 2
        explanation = self._explain(
            risks, max_dim_name, max_dim, weighted, overall, severity_rule, floor_applied
        )
        if multi_risk:
            explanation += " Multiple risk dimensions are elevated simultaneously."

        return FusionResult(
            performance_risk=round(risks["performance"], 4),
            responsibility_risk=round(risks["responsibility"], 4),
            cost_risk=round(risks["cost"], 4),
            overall_risk=round(overall, 4),
            dominant_dimension=max_dim_name,
            dominant_risk=round(max_dim, 4),
            risk_breakdown=breakdown,
            confidence=round(confidence, 4),
            uncertainty=round(clamp01(1.0 - confidence), 4),
            multi_risk=multi_risk,
            weighted_only_risk=round(clamp01(weighted), 4),
            severity_rule_applied=severity_rule,
            severity_floor_applied=floor_applied,
            explanation=explanation,
        )

    # ------------------------------------------------------------------

    def _fused_confidence(
        self, risks: dict[str, float], confidences: dict[str, float]
    ) -> float:
        # Weight each dimension's confidence by its share of the total risk
        # mass (a dimension that is not contributing risk barely matters),
        # falling back to the configured weights when all risks are ~0.
        risk_mass = sum(risks.values())
        if risk_mass < 1e-6:
            weights = self._weights
        else:
            weights = {d: risks[d] / risk_mass for d in risks}
        weighted_conf = sum(weights[d] * confidences[d] for d in risks) / (
            sum(weights.values()) or 1.0
        )
        # Disagreement penalty: sharply divergent dimension risks reduce how
        # much we trust a single fused number.
        spread = pstdev(list(risks.values())) if len(risks) > 1 else 0.0
        return clamp01(weighted_conf - 0.25 * spread)

    @staticmethod
    def _explain(
        risks: dict[str, float],
        max_dim_name: str,
        max_dim: float,
        weighted: float,
        overall: float,
        severity_rule: bool,
        floor_applied: bool,
    ) -> str:
        label = _DIMENSION_LABELS[max_dim_name]
        parts = [
            f"Dimension risks — performance {risks['performance']:.2f}, "
            f"responsibility {risks['responsibility']:.2f}, cost {risks['cost']:.2f}.",
            f"Weighted blend alone would be {weighted:.2f}.",
        ]
        if severity_rule:
            parts.append(
                f"The {label} dimension is severe ({max_dim:.2f}), so the "
                f"conservative safety rule pulls the fused risk up toward it: "
                f"overall risk {overall:.2f}."
            )
        else:
            parts.append(
                f"No single dimension is severe; overall risk {overall:.2f} "
                f"(dominant dimension: {label})."
            )
        if floor_applied:
            parts.append(
                "A hard severity floor was applied because a dimension exceeded "
                "the critical threshold."
            )
        return " ".join(parts)


def _get(obj: Any, attr: str, default: float = 0.0) -> float:
    if obj is None:
        return default
    if isinstance(obj, (int, float)):
        return float(obj)
    return float(getattr(obj, attr, default))


def fuse_risk(
    performance: Any,
    responsibility: Any,
    cost: Any,
    config: dict[str, Any] | None = None,
) -> FusionResult:
    """Convenience one-shot wrapper."""
    return RiskFusionEngine(config=config).fuse(performance, responsibility, cost)
