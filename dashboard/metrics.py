"""
Monitoring / dashboard metrics.

Aggregates *actual* ``DecisionTrace`` outputs from the pipeline into a
single summary object. No fabricated numbers — every field is derived
from real traces passed in by the caller.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Iterable, Sequence

from pydantic import BaseModel

from decision.schemas import DecisionTrace
from data.schemas import InterventionTier

# Risk histograms use five fixed-width buckets over [0, 1]. The final upper
# bound is 1.0001 so a risk of exactly 1.0 lands in the last bucket.
_RISK_BUCKET_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
_RISK_BUCKET_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

# A "top risk category" is only counted when the signal is materially
# elevated, so a low-risk redacted identifier does not inflate the panel.
_MATERIAL_RISK = 0.5
_MATERIAL_UNVERIFIED_PERFORMANCE_RISK = 0.4
_CONTRADICTION_STATUSES = ("CONTRADICTED", "PARTIALLY_SUPPORTED")


def histogram(values: Sequence[float]) -> dict[str, int]:
    """Bucket ``values`` into the five fixed risk bands, returned label -> count."""
    counts = {label: 0 for label in _RISK_BUCKET_LABELS}
    for value in values:
        for bucket_index in range(len(_RISK_BUCKET_LABELS)):
            lower = _RISK_BUCKET_EDGES[bucket_index]
            upper = _RISK_BUCKET_EDGES[bucket_index + 1]
            if lower <= value < upper:
                counts[_RISK_BUCKET_LABELS[bucket_index]] += 1
                break
    return counts


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(
        len(sorted_values) - 1,
        int(round(fraction * (len(sorted_values) - 1))),
    )
    return sorted_values[index]


class DashboardMetrics(BaseModel):
    generated_at: datetime | None = None
    total_interactions: int

    decisions: dict[str, int]
    allow_rate: float
    annotate_rate: float
    verify_rate: float
    human_review_rate: float
    block_rate: float

    performance_risk_histogram: dict[str, int]
    responsibility_risk_histogram: dict[str, int]
    cost_risk_histogram: dict[str, int]
    overall_risk_histogram: dict[str, int]

    performance_status_distribution: dict[str, int]
    top_risk_categories: list[dict[str, Any]]
    triggered_rule_counts: dict[str, int]

    mean_latency_ms: float
    p95_latency_ms: float
    mean_detector_confidence: float

    total_estimated_cost_inr: float
    mean_estimated_cost_inr: float

    session_escalations: int
    by_application: dict[str, dict[str, int]]


def _accumulate_top_risk_categories(
    trace: DecisionTrace, category_counts: Counter[str]
) -> None:
    """Add this trace's materially-elevated risk categories to the running tally."""
    performance = trace.performance
    responsibility = trace.responsibility

    if performance.status.value in _CONTRADICTION_STATUSES:
        category_counts["performance:contradiction"] += 1
    elif (
        performance.status.value == "UNVERIFIED"
        and performance.performance_risk >= _MATERIAL_UNVERIFIED_PERFORMANCE_RISK
    ):
        category_counts["performance:unverified"] += 1

    if responsibility.pii_risk >= _MATERIAL_RISK:
        category_counts["responsibility:pii"] += 1
    if responsibility.toxicity_risk >= _MATERIAL_RISK:
        category_counts["responsibility:toxicity"] += 1
    if responsibility.bias_risk >= _MATERIAL_RISK:
        category_counts["responsibility:bias"] += 1
    if trace.cost.cost_risk >= _MATERIAL_RISK:
        category_counts["cost:anomaly"] += 1


def compute_dashboard_metrics(
    traces: Iterable[DecisionTrace],
    *,
    session_manager: Any | None = None,
    generated_at: datetime | None = None,
) -> DashboardMetrics:
    traces = list(traces)
    trace_count = len(traces)
    if trace_count == 0:
        raise ValueError("compute_dashboard_metrics requires at least one trace")

    decision_counts = Counter(t.final_decision.decision.value for t in traces)
    performance_status_counts = Counter(t.performance.status.value for t in traces)
    triggered_rule_counts: Counter[str] = Counter()
    risk_category_counts: Counter[str] = Counter()
    decisions_by_application: dict[str, Counter[str]] = {}

    latencies_ms: list[float] = []
    decision_confidences: list[float] = []
    estimated_costs_inr: list[float] = []

    for trace in traces:
        decision = trace.final_decision
        triggered_rule_counts.update(decision.triggered_rules)
        latencies_ms.append(trace.latency_ms)
        decision_confidences.append(decision.decision_confidence)
        estimated_costs_inr.append(trace.cost.estimated_cost_inr)

        app_counts = decisions_by_application.setdefault(trace.application, Counter())
        app_counts[decision.decision.value] += 1

        _accumulate_top_risk_categories(trace, risk_category_counts)

    latencies_ms_sorted = sorted(latencies_ms)

    def _tier_rate(tier: InterventionTier) -> float:
        return round(decision_counts.get(tier.value, 0) / trace_count, 4)

    session_escalations = 0
    if session_manager is not None:
        for session_id in session_manager.active_sessions():
            if session_manager.get_state(session_id).escalated:
                session_escalations += 1

    return DashboardMetrics(
        generated_at=generated_at,
        total_interactions=trace_count,
        decisions={
            tier.value: decision_counts.get(tier.value, 0) for tier in InterventionTier
        },
        allow_rate=_tier_rate(InterventionTier.ALLOW),
        annotate_rate=_tier_rate(InterventionTier.ANNOTATE),
        verify_rate=_tier_rate(InterventionTier.VERIFY),
        human_review_rate=_tier_rate(InterventionTier.HUMAN_REVIEW),
        block_rate=_tier_rate(InterventionTier.BLOCK),
        performance_risk_histogram=histogram(
            [t.performance.performance_risk for t in traces]
        ),
        responsibility_risk_histogram=histogram(
            [t.responsibility.overall_responsibility_risk for t in traces]
        ),
        cost_risk_histogram=histogram([t.cost.cost_risk for t in traces]),
        overall_risk_histogram=histogram(
            [t.final_decision.overall_risk for t in traces]
        ),
        performance_status_distribution=dict(performance_status_counts),
        top_risk_categories=[
            {"category": category, "count": count}
            for category, count in risk_category_counts.most_common(10)
        ],
        triggered_rule_counts=dict(triggered_rule_counts.most_common()),
        mean_latency_ms=round(sum(latencies_ms) / trace_count, 3),
        p95_latency_ms=round(_percentile(latencies_ms_sorted, 0.95), 3),
        mean_detector_confidence=round(
            sum(decision_confidences) / trace_count, 4
        ),
        total_estimated_cost_inr=round(sum(estimated_costs_inr), 4),
        mean_estimated_cost_inr=round(sum(estimated_costs_inr) / trace_count, 6),
        session_escalations=session_escalations,
        by_application={
            application: dict(counts)
            for application, counts in decisions_by_application.items()
        },
    )


def collect_traces(
    config: dict[str, Any] | None = None,
    *,
    limit: int | None = 150,
    source: str = "evaluation",
):
    """
    Run the pipeline over synthetic data and return (traces, session_manager).

    ``source="evaluation"`` uses the 150-case evaluation set; ``"traffic"``
    uses the head of the 6000-case production-traffic set.
    """
    from evaluation.evaluation import _EVAL_TS, build_engine, load_evaluation_dataset
    from session.manager import SessionManager

    session_manager = SessionManager(config)
    engine = build_engine(config, session_manager=session_manager)

    if source == "evaluation":
        interactions = [
            interaction for interaction, _ in load_evaluation_dataset(config)
        ]
    else:
        import random

        from data.generator import generate_interactions
        from settings import load_settings

        resolved_config = config or load_settings()
        interactions = generate_interactions(
            resolved_config, random.Random(resolved_config["seed"])
        )

    if limit is not None:
        interactions = interactions[:limit]

    traces = [engine.evaluate(i, timestamp=_EVAL_TS) for i in interactions]
    return traces, session_manager
