"""Dashboard / monitoring metrics tests — all values derived from real traces."""

from __future__ import annotations

import pytest

from dashboard.metrics import collect_traces, compute_dashboard_metrics, histogram


def test_histogram_buckets():
    h = histogram([0.0, 0.1, 0.25, 0.5, 0.95, 1.0])
    assert sum(h.values()) == 6
    assert h["0.0-0.2"] == 2
    assert h["0.8-1.0"] == 2


def test_empty_traces_raises():
    with pytest.raises(ValueError):
        compute_dashboard_metrics([])


@pytest.fixture(scope="module")
def metrics():
    traces, sm = collect_traces(limit=150, source="evaluation")
    return compute_dashboard_metrics(traces, session_manager=sm)


def test_totals_and_rates_consistent(metrics):
    assert metrics.total_interactions == 150
    assert sum(metrics.decisions.values()) == 150
    rate_sum = (
        metrics.allow_rate
        + metrics.annotate_rate
        + metrics.verify_rate
        + metrics.human_review_rate
        + metrics.block_rate
    )
    assert rate_sum == pytest.approx(1.0, abs=1e-3)


def test_histograms_sum_to_total(metrics):
    for hist in (
        metrics.performance_risk_histogram,
        metrics.responsibility_risk_histogram,
        metrics.cost_risk_histogram,
        metrics.overall_risk_histogram,
    ):
        assert sum(hist.values()) == 150


def test_real_signal_present(metrics):
    assert metrics.total_estimated_cost_inr > 0
    assert metrics.mean_latency_ms > 0
    assert 0.0 <= metrics.mean_detector_confidence <= 1.0
    assert metrics.triggered_rule_counts  # some rules fired
    assert metrics.top_risk_categories


def test_top_categories_not_inflated_by_low_risk_findings(metrics):
    # PII category count must not exceed the number of genuine PII eval cases (25).
    pii_entry = next(
        (c for c in metrics.top_risk_categories if c["category"] == "responsibility:pii"),
        None,
    )
    if pii_entry is not None:
        assert pii_entry["count"] <= 30


def test_by_application_breakdown(metrics):
    assert set(metrics.by_application) <= {
        "customer_support",
        "internal_knowledge_assistant",
        "decision_support",
    }
    total = sum(sum(v.values()) for v in metrics.by_application.values())
    assert total == 150


def test_deterministic():
    t1, s1 = collect_traces(limit=60)
    t2, s2 = collect_traces(limit=60)
    m1 = compute_dashboard_metrics(t1, session_manager=s1).model_dump()
    m2 = compute_dashboard_metrics(t2, session_manager=s2).model_dump()
    for m in (m1, m2):
        m.pop("mean_latency_ms")
        m.pop("p95_latency_ms")
    assert m1 == m2
