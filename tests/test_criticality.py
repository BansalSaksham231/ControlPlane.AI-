"""Claim / Action Criticality engine tests."""

from __future__ import annotations

import pytest

from criticality.engine import CriticalityEngine, assess_criticality
from criticality.schemas import CriticalityAssessment
from data.schemas import ActionType, Application, Interaction, ModelName, UserType
from detectors.performance.detector import PerformanceDetector


@pytest.fixture(scope="module")
def engine() -> CriticalityEngine:
    return CriticalityEngine()


def _interaction(action_type, amount=0.0, entities=1, response="r") -> Interaction:
    return Interaction(
        interaction_id="INT-CRIT-1",
        timestamp="2026-08-21T12:00:00",
        application=Application.CUSTOMER_SUPPORT,
        user_type=UserType.EXTERNAL_CUSTOMER,
        model=ModelName.GPT_4O_MINI,
        session_id="S1",
        prompt="p",
        context="c",
        response=response,
        tokens_in=10,
        tokens_out=10,
        latency_ms=100.0,
        tool_calls=0,
        retry_count=0,
        action_type=action_type,
        action_amount_inr=amount,
        affected_entities=entities,
    )


def test_ranges(engine):
    result = engine.assess(_interaction(ActionType.REFUND, amount=250_000, entities=40))
    assert isinstance(result, CriticalityAssessment)
    assert 0.0 <= result.action_criticality <= 1.0
    for factor in result.factors:
        assert 0.0 <= factor.value <= 1.0
        assert 0.0 <= factor.weighted_contribution <= 1.0


def test_information_is_low_criticality(engine):
    result = engine.assess(_interaction(ActionType.INFORMATION))
    assert result.band == "low"
    assert result.action_criticality < 0.4


def test_large_financial_action_is_high_criticality(engine):
    low = engine.assess(_interaction(ActionType.REFUND, amount=1_000))
    high = engine.assess(_interaction(ActionType.REFUND, amount=480_000))
    assert high.action_criticality > low.action_criticality
    assert "HIGH_FINANCIAL_IMPACT" in high.reason_codes


def test_irreversible_action_reason_code(engine):
    result = engine.assess(_interaction(ActionType.EXTERNAL_COMMUNICATION))
    assert "IRREVERSIBLE_ACTION" in result.reason_codes


def test_blast_radius_scales_with_entities(engine):
    small = engine.assess(_interaction(ActionType.RECOMMENDATION, entities=1))
    large = engine.assess(_interaction(ActionType.RECOMMENDATION, entities=900))
    large_br = next(f for f in large.factors if f.factor == "blast_radius")
    small_br = next(f for f in small.factors if f.factor == "blast_radius")
    assert large_br.value > small_br.value
    assert "HIGH_BLAST_RADIUS" in large.reason_codes


def test_claim_text_criticality_from_money_and_verbs(engine):
    money, signals = engine.text_criticality("Approve a refund of 480000 for this account.")
    assert money > 0.5
    assert any("action_verb" in s for s in signals)
    plain, _ = engine.text_criticality("The office is open on weekdays.")
    assert plain < money


def test_per_claim_criticality_uses_performance(engine):
    detector = PerformanceDetector()
    perf = detector.detect(
        "Approve a refund of 480000 rupees. The office is in the city centre.",
        "Refund policy details.",
    )
    result = engine.assess(_interaction(ActionType.REFUND, amount=100), perf)
    assert result.claim_criticalities
    money_claim = max(result.claim_criticalities, key=lambda c: c.criticality)
    assert "480000" in money_claim.claim
    assert result.max_claim_criticality > 0.0


def test_amplification_gated_below_moderate(engine):
    # Low criticality -> risk is not amplified at all.
    assert engine.amplify_performance_risk(0.5, 0.2) == pytest.approx(0.5)
    # High criticality -> a moderate risk is pushed up.
    assert engine.amplify_performance_risk(0.5, 0.9) > 0.5
    # A near-zero risk barely moves even at max criticality.
    assert engine.amplify_performance_risk(0.03, 1.0) < 0.1


def test_no_ground_truth_used():
    assert "ground_truth_hallucination" not in Interaction.model_fields
    result = assess_criticality(_interaction(ActionType.REFUND, amount=50_000))
    assert 0.0 <= result.action_criticality <= 1.0


def test_deterministic(engine):
    a = engine.assess(_interaction(ActionType.REFUND, amount=50_000, entities=5)).model_dump()
    b = engine.assess(_interaction(ActionType.REFUND, amount=50_000, entities=5)).model_dump()
    assert a == b
