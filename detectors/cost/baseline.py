"""
Cost / usage baselines.

Two modes:

* ``from_config`` — static prototype baselines only (no data needed). This
  keeps the Cost Detector usable stand-alone.
* ``fit`` — derive per-(application, model) percentile baselines from a
  corpus of production Interactions. Anomaly detection then compares an
  interaction against ``max(static, empirical)`` so a quiet segment can
  never make the static floor *lower*.

Ground truth is never read here — only production-visible usage fields.
"""

from __future__ import annotations

from typing import Any, Iterable

from data.schemas import Interaction

_NUMERIC_DIMENSIONS = ("tokens_in", "tokens_out", "latency_ms", "tool_calls", "retry_count")


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


class CostBaseline:
    def __init__(
        self,
        static: dict[str, float],
        empirical: dict[tuple[str, str], dict[str, float]] | None = None,
        source: str = "static",
    ) -> None:
        self._static = dict(static)
        self._empirical = empirical or {}
        self.source = source

    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CostBaseline":
        static = dict(config["cost_detector"]["static_baselines"])
        return cls(static=static, empirical=None, source="static")

    @classmethod
    def fit(
        cls,
        interactions: Iterable[Interaction],
        config: dict[str, Any],
        estimate_cost=None,
        percentile: float | None = None,
    ) -> "CostBaseline":
        cdcfg = config["cost_detector"]
        fraction = percentile if percentile is not None else float(cdcfg["baseline_percentile"])
        static = dict(cdcfg["static_baselines"])

        buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
        for interaction in interactions:
            key = (interaction.application.value, interaction.model.value)
            slot = buckets.setdefault(key, {dim: [] for dim in _NUMERIC_DIMENSIONS})
            slot["tokens_in"].append(float(interaction.tokens_in))
            slot["tokens_out"].append(float(interaction.tokens_out))
            slot["latency_ms"].append(float(interaction.latency_ms))
            slot["tool_calls"].append(float(interaction.tool_calls))
            slot["retry_count"].append(float(interaction.retry_count))
            if estimate_cost is not None:
                slot.setdefault("estimated_cost_inr", []).append(
                    float(estimate_cost(interaction).total_cost_inr)
                )

        empirical: dict[tuple[str, str], dict[str, float]] = {}
        for key, dims in buckets.items():
            empirical[key] = {
                dim: _percentile(sorted(values), fraction)
                for dim, values in dims.items()
                if values
            }
        return cls(static=static, empirical=empirical, source="empirical")

    # ------------------------------------------------------------------

    def get(self, application: str, model: str, dimension: str) -> float:
        static_value = float(self._static.get(dimension, 0.0))
        empirical_value = float(
            self._empirical.get((application, model), {}).get(dimension, 0.0)
        )
        return max(static_value, empirical_value)
