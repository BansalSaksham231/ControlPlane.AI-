"""Risk Fusion Engine tests."""

from __future__ import annotations

import pytest

from fusion.engine import RiskFusionEngine, fuse_risk
from fusion.schemas import FusionResult


@pytest.fixture(scope="module")
def engine():
    return RiskFusionEngine()


def test_all_low_risk_stays_low(engine):
    result = engine.fuse_scores(0.05, 0.05, 0.05)
    assert result.overall_risk < 0.15
    assert result.severity_rule_applied is False


def test_normal_moderate_risk(engine):
    result = engine.fuse_scores(0.35, 0.30, 0.20)
    assert 0.2 <= result.overall_risk <= 0.5
    assert isinstance(result, FusionResult)


def test_dominant_dimension_identified(engine):
    result = engine.fuse_scores(0.2, 0.8, 0.1)
    assert result.dominant_dimension == "responsibility"
    assert result.dominant_risk == pytest.approx(0.8)


def test_single_severe_dimension_not_diluted(engine):
    """The brief's canonical example."""
    result = engine.fuse_scores(0.95, 0.05, 0.05)
    naive_average = (0.95 + 0.05 + 0.05) / 3
    assert result.overall_risk > 0.6
    assert result.overall_risk > naive_average
    assert result.overall_risk > result.weighted_only_risk
    assert result.severity_rule_applied is True


def test_severity_floor_applied_for_extreme_dimension(engine):
    result = engine.fuse_scores(0.97, 0.0, 0.0)
    assert result.severity_floor_applied is True
    assert result.overall_risk >= 0.8


def test_multiple_high_dimensions_block_range(engine):
    result = engine.fuse_scores(0.95, 0.95, 0.2)
    assert result.overall_risk >= 0.85


def test_confidence_penalised_by_disagreement(engine):
    agree = engine.fuse_scores(0.6, 0.6, 0.6, 0.8, 0.8, 0.8)
    disagree = engine.fuse_scores(0.95, 0.05, 0.05, 0.8, 0.8, 0.8)
    assert disagree.confidence < agree.confidence


def test_risk_breakdown_sums_to_weighted_only(engine):
    result = engine.fuse_scores(0.4, 0.6, 0.3)
    total = sum(c.weighted_contribution for c in result.risk_breakdown)
    assert total == pytest.approx(result.weighted_only_risk, abs=1e-3)


def test_explanation_is_readable(engine):
    result = engine.fuse_scores(0.95, 0.1, 0.1)
    assert "performance" in result.explanation.lower()
    assert len(result.explanation) > 40
    assert ">" not in result.explanation  # no raw "risk > threshold" style


def test_accepts_result_objects():
    class _P:
        performance_risk = 0.9
        confidence = 0.7

    class _R:
        overall_responsibility_risk = 0.1
        confidence = 0.6

    class _C:
        cost_risk = 0.1
        confidence = 0.8

    result = fuse_risk(_P(), _R(), _C())
    assert result.performance_risk == pytest.approx(0.9)
    assert result.dominant_dimension == "performance"


def test_deterministic(engine):
    a = engine.fuse_scores(0.7, 0.3, 0.5).model_dump()
    b = engine.fuse_scores(0.7, 0.3, 0.5).model_dump()
    assert a == b
