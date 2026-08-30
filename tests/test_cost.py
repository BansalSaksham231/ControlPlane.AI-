"""Cost / Operational Risk Detector tests."""

from __future__ import annotations

import random

import pytest

from data.generator import generate_interactions
from data.schemas import ActionType, Application, Interaction, ModelName, UserType
from detectors.cost.baseline import CostBaseline
from detectors.cost.detector import CostDetector, detect_cost
from detectors.cost.schemas import CostResult
from settings import load_settings


@pytest.fixture(scope="module")
def config():
    return load_settings()


@pytest.fixture(scope="module")
def detector(config):
    return CostDetector(config=config)


def _interaction(**overrides) -> Interaction:
    base = dict(
        interaction_id="INT-C-1",
        timestamp="2026-08-21T12:00:00",
        application=Application.CUSTOMER_SUPPORT,
        user_type=UserType.EXTERNAL_CUSTOMER,
        model=ModelName.GPT_4O_MINI,
        session_id="S1",
        prompt="p",
        context="c",
        response="r",
        tokens_in=60,
        tokens_out=90,
        latency_ms=400.0,
        tool_calls=0,
        retry_count=0,
        action_type=ActionType.INFORMATION,
        action_amount_inr=0.0,
        affected_entities=1,
    )
    base.update(overrides)
    return Interaction(**base)


def test_cost_estimate_is_additive(detector):
    breakdown = detector.estimate_cost(_interaction(tokens_in=1000, tokens_out=1000, tool_calls=2, retry_count=1))
    assert breakdown.total_cost_inr == pytest.approx(
        breakdown.input_cost_inr
        + breakdown.output_cost_inr
        + breakdown.tool_cost_inr
        + breakdown.retry_cost_inr
    )
    assert breakdown.tool_cost_inr > 0
    assert breakdown.retry_cost_inr > 0


def test_normal_usage_low_risk(detector):
    result = detector.detect(_interaction())
    assert result.cost_risk < 0.5
    assert result.triggered_dimensions == []
    assert isinstance(result, CostResult)


def test_high_token_usage_flagged(detector):
    result = detector.detect(_interaction(tokens_in=5000, tokens_out=8000))
    assert result.cost_risk > 0.4
    assert "tokens_out" in result.triggered_dimensions


def test_excessive_retries_and_tools_flagged(detector):
    result = detector.detect(_interaction(retry_count=6, tool_calls=9))
    assert "retry_count" in result.triggered_dimensions
    assert "tool_calls" in result.triggered_dimensions
    assert result.cost_risk > 0.4


def test_cost_spike_flagged(detector):
    result = detector.detect(
        _interaction(tokens_in=4000, tokens_out=9000, tool_calls=10, retry_count=5)
    )
    assert result.cost_risk >= 0.8
    assert "estimated_cost_inr" in result.triggered_dimensions


def test_latency_anomaly_flagged(detector):
    result = detector.detect(_interaction(latency_ms=12000.0))
    assert "latency_ms" in result.triggered_dimensions


def test_anomaly_indicators_are_transparent(detector):
    result = detector.detect(_interaction(tokens_out=9000))
    for indicator in result.anomaly_indicators:
        assert indicator.baseline >= 0
        assert indicator.ratio >= 0
        assert len(indicator.explanation) > 5


def test_empirical_baseline_fit(config):
    rng = random.Random(config["seed"])
    interactions = generate_interactions(config, rng)
    detector = CostDetector(config=config)
    baseline = CostBaseline.fit(interactions, config, estimate_cost=detector.estimate_cost)
    assert baseline.source == "empirical"
    fitted = CostDetector(config=config, baseline=baseline)
    result = fitted.detect(_interaction())
    assert result.baseline_source == "empirical"
    assert result.confidence >= 0.8


def test_deterministic(detector):
    a = detector.detect(_interaction(tokens_out=9000)).model_dump()
    b = detector.detect(_interaction(tokens_out=9000)).model_dump()
    a.pop("latency_ms")
    b.pop("latency_ms")
    assert a == b


def test_convenience_wrapper():
    assert isinstance(detect_cost(_interaction()), CostResult)


def test_no_ground_truth_fields_used():
    assert "ground_truth_cost_anomaly" not in Interaction.model_fields


# ---------------------------------------------------------------- efficiency (Round 2)

def test_normal_interaction_is_efficient(detector):
    result = detector.detect(_interaction())
    assert result.cost_efficiency_score >= 0.8
    assert result.anomaly_types == []
    assert result.retry_inefficiency == 0.0


def test_retry_spike_hurts_efficiency(detector):
    result = detector.detect(_interaction(retry_count=6))
    assert "RETRY_SPIKE" in result.anomaly_types
    assert result.retry_inefficiency > 0.0
    assert result.cost_efficiency_score < 0.6


def test_token_spike_typed(detector):
    result = detector.detect(_interaction(tokens_in=5000, tokens_out=8000))
    assert "TOKEN_SPIKE" in result.anomaly_types


def test_tool_loop_detected(detector):
    result = detector.detect(_interaction(tool_calls=9, tokens_out=30))
    assert "TOOL_LOOP" in result.anomaly_types


def test_latency_spike_typed(detector):
    result = detector.detect(_interaction(latency_ms=12000.0))
    assert "LATENCY_SPIKE" in result.anomaly_types


def test_efficiency_in_unit_range(detector):
    for kwargs in ({}, {"retry_count": 5}, {"tool_calls": 10}, {"tokens_out": 9000}):
        result = detector.detect(_interaction(**kwargs))
        assert 0.0 <= result.cost_efficiency_score <= 1.0
