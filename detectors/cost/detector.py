"""
Cost / Operational Risk Detector.

Estimates the cost of an interaction from a transparent additive model
(input tokens + output tokens + tool calls + retries, at configurable
prototype rates) and flags operational anomalies — unusual token usage,
latency, retries, tool-call counts or overall cost spikes — against a
baseline.

The rates are illustrative demo figures and do NOT represent any real
provider's billing.
"""

from __future__ import annotations

from typing import Any

from common.timing import Stopwatch, clamp01
from data.schemas import Interaction
from detectors.cost.baseline import CostBaseline
from detectors.cost.schemas import CostAnomalyIndicator, CostBreakdown, CostResult
from settings import load_settings

DETECTOR_NAME = "cost"

# (dimension, is_ratio_based) — count-style dimensions use an absolute cap.
_RATIO_DIMENSIONS = ("tokens_in", "tokens_out", "latency_ms", "estimated_cost_inr")


class CostDetector:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        baseline: CostBaseline | None = None,
    ) -> None:
        settings = config if config is not None else load_settings()
        self._cfg = settings["cost_detector"]
        self.baseline = baseline or CostBaseline.from_config(settings)

    # ------------------------------------------------------------------

    def estimate_cost(self, interaction: Interaction) -> CostBreakdown:
        model = interaction.model.value
        input_rate = self._rate("input_rate_per_1k_inr", model)
        output_rate = self._rate("output_rate_per_1k_inr", model)

        input_cost = interaction.tokens_in / 1000.0 * input_rate
        output_cost = interaction.tokens_out / 1000.0 * output_rate
        tool_cost = interaction.tool_calls * float(self._cfg["tool_call_cost_inr"])
        retry_cost = interaction.retry_count * float(self._cfg["retry_cost_inr"])
        total = input_cost + output_cost + tool_cost + retry_cost

        return CostBreakdown(
            input_cost_inr=round(input_cost, 6),
            output_cost_inr=round(output_cost, 6),
            tool_cost_inr=round(tool_cost, 6),
            retry_cost_inr=round(retry_cost, 6),
            total_cost_inr=round(total, 6),
        )

    def detect(self, interaction: Interaction) -> CostResult:
        watch = Stopwatch()
        breakdown = self.estimate_cost(interaction)
        app = interaction.application.value
        model = interaction.model.value

        multipliers = self._cfg["anomaly_multipliers"]
        indicators: list[CostAnomalyIndicator] = []

        observed_values = {
            "tokens_in": float(interaction.tokens_in),
            "tokens_out": float(interaction.tokens_out),
            "latency_ms": float(interaction.latency_ms),
            "estimated_cost_inr": float(breakdown.total_cost_inr),
        }
        for dimension in _RATIO_DIMENSIONS:
            baseline_value = self.baseline.get(app, model, dimension)
            if baseline_value <= 0:
                continue
            observed = observed_values[dimension]
            ratio = observed / baseline_value
            threshold = float(multipliers[dimension])
            indicators.append(
                CostAnomalyIndicator(
                    dimension=dimension,
                    observed=round(observed, 4),
                    baseline=round(baseline_value, 4),
                    ratio=round(ratio, 3),
                    threshold=threshold,
                    triggered=ratio >= threshold,
                    explanation=(
                        f"{dimension} = {observed:.2f} vs baseline {baseline_value:.2f} "
                        f"({ratio:.1f}x; flags at {threshold:.1f}x)."
                    ),
                )
            )

        # Count-style absolute caps.
        indicators.append(
            self._count_indicator(
                "tool_calls",
                float(interaction.tool_calls),
                float(self._cfg["max_normal_tool_calls"]),
            )
        )
        indicators.append(
            self._count_indicator(
                "retry_count",
                float(interaction.retry_count),
                float(self._cfg["max_normal_retries"]),
            )
        )

        triggered = [ind for ind in indicators if ind.triggered]
        cost_risk = self._risk(triggered)
        confidence = 0.82 if self.baseline.source == "empirical" else 0.7

        efficiency, cost_per_success, retry_inefficiency = self._efficiency(
            interaction, breakdown, app, model
        )
        anomaly_types = self._anomaly_types(interaction, triggered, breakdown)
        explanation = self._explain(
            breakdown, triggered, cost_risk, efficiency, anomaly_types
        )

        return CostResult(
            estimated_cost_inr=breakdown.total_cost_inr,
            cost_breakdown=breakdown,
            cost_risk=round(cost_risk, 4),
            confidence=confidence,
            anomaly_indicators=indicators,
            triggered_dimensions=[ind.dimension for ind in triggered],
            cost_efficiency_score=round(efficiency, 4),
            cost_per_success_inr=round(cost_per_success, 6),
            retry_inefficiency=round(retry_inefficiency, 4),
            anomaly_types=anomaly_types,
            baseline_source=self.baseline.source,
            explanation=explanation,
            latency_ms=watch.total_ms(),
        )

    # ------------------------------------------------------------------

    def _efficiency(
        self, interaction: Interaction, breakdown: CostBreakdown, app: str, model: str
    ) -> tuple[float, float, float]:
        """
        cost_efficiency_score in [0,1]: 1.0 == on par with the baseline.

        Penalised by wasted spend — retries (earlier attempts thrown away),
        tool-call overhead disproportionate to output, and output far
        larger than the baseline.
        """
        total = max(breakdown.total_cost_inr, 1e-9)
        # Cost to produce ONE successful response — retries are wasted work,
        # so a 3-retry interaction "cost" ~4x a clean one for one success.
        attempts = 1 + interaction.retry_count
        cost_per_success = total  # the retry overhead is already in `total`

        retry_inefficiency = clamp01(breakdown.retry_cost_inr / total)
        tool_overhead = clamp01(breakdown.tool_cost_inr / total)

        baseline_cost = self.baseline.get(app, model, "estimated_cost_inr")
        cost_ratio = total / baseline_cost if baseline_cost > 0 else 1.0
        # 1.0 at/under baseline, decaying as cost balloons past it.
        cost_component = clamp01(1.0 / max(cost_ratio, 1.0))

        efficiency = clamp01(
            0.55 * cost_component
            + 0.25 * (1.0 - retry_inefficiency)
            + 0.20 * (1.0 - min(1.0, tool_overhead * 1.5))
        )
        # An interaction that needed several attempts is inherently less
        # efficient regardless of absolute cost.
        if attempts > 1:
            efficiency = clamp01(efficiency * (1.0 / attempts) ** 0.4)
        return efficiency, cost_per_success, retry_inefficiency

    def _anomaly_types(
        self,
        interaction: Interaction,
        triggered: list[CostAnomalyIndicator],
        breakdown: CostBreakdown,
    ) -> list[str]:
        dims = {ind.dimension for ind in triggered}
        types: list[str] = []
        if dims & {"tokens_in", "tokens_out"}:
            types.append("TOKEN_SPIKE")
        if "retry_count" in dims:
            types.append("RETRY_SPIKE")
        if "tool_calls" in dims:
            # A tool loop = many tool calls without a proportionate amount
            # of useful output.
            if interaction.tool_calls >= 5 and interaction.tokens_out < 60 * interaction.tool_calls:
                types.append("TOOL_LOOP")
            else:
                types.append("TOOL_SPIKE")
        if "latency_ms" in dims:
            types.append("LATENCY_SPIKE")
        if "estimated_cost_inr" in dims:
            types.append("COST_PER_SUCCESS_SPIKE")
        return types

    # ------------------------------------------------------------------

    def _rate(self, table_key: str, model: str) -> float:
        table = self._cfg[table_key]
        return float(table.get(model, table["default"]))

    def _count_indicator(
        self, dimension: str, observed: float, cap: float
    ) -> CostAnomalyIndicator:
        ratio = observed / cap if cap > 0 else 0.0
        return CostAnomalyIndicator(
            dimension=dimension,
            observed=observed,
            baseline=cap,
            ratio=round(ratio, 3),
            threshold=1.0,
            triggered=observed > cap,
            explanation=(
                f"{dimension} = {int(observed)} vs normal maximum {int(cap)}."
            ),
        )

    def _risk(self, triggered: list[CostAnomalyIndicator]) -> float:
        if not triggered:
            return 0.0
        per_indicator = float(self._cfg["risk_per_indicator"])
        risk = 0.0
        for indicator in triggered:
            # A dimension that is far past its threshold contributes more.
            severity = min(2.0, indicator.ratio / max(indicator.threshold, 1e-6))
            risk += per_indicator * (0.6 + 0.4 * severity)
        return clamp01(risk)

    @staticmethod
    def _explain(
        breakdown: CostBreakdown,
        triggered: list[CostAnomalyIndicator],
        cost_risk: float,
        efficiency: float,
        anomaly_types: list[str],
    ) -> str:
        base = (
            f"Estimated interaction cost ₹{breakdown.total_cost_inr:.4f} "
            f"(input ₹{breakdown.input_cost_inr:.4f}, output "
            f"₹{breakdown.output_cost_inr:.4f}, tools "
            f"₹{breakdown.tool_cost_inr:.4f}, retries "
            f"₹{breakdown.retry_cost_inr:.4f}); efficiency {efficiency:.2f}."
        )
        if not triggered:
            return base + " Usage is within normal operating bounds; cost risk is low."
        dims = ", ".join(ind.dimension for ind in triggered)
        types = ", ".join(anomaly_types) or "operational anomaly"
        return (
            base
            + f" Anomaly type(s): {types} (dimensions: {dims}). Cost risk {cost_risk:.2f}."
        )


def detect_cost(
    interaction: Interaction, config: dict[str, Any] | None = None
) -> CostResult:
    """Convenience one-shot wrapper."""
    return CostDetector(config=config).detect(interaction)
