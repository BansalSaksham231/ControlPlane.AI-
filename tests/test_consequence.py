"""Consequence Engine tests."""

from __future__ import annotations

import random

import pytest

from consequence.engine import ConsequenceEngine, assess_consequence
from data.generator import compute_consequence_factors, generate_evaluation_cases
from data.schemas import ActionType, Application, Interaction, ModelName, UserType
from settings import load_settings


@pytest.fixture(scope="module")
def config():
    return load_settings()


@pytest.fixture(scope="module")
def engine(config):
    return ConsequenceEngine(config=config)


def _interaction(action_type, amount=0.0, entities=1) -> Interaction:
    return Interaction(
        interaction_id="INT-Q-1",
        timestamp="2026-08-21T12:00:00",
        application=Application.CUSTOMER_SUPPORT,
        user_type=UserType.EXTERNAL_CUSTOMER,
        model=ModelName.GPT_4O_MINI,
        session_id="S1",
        prompt="p",
        context="c",
        response="r",
        tokens_in=10,
        tokens_out=10,
        latency_ms=100.0,
        tool_calls=0,
        retry_count=0,
        action_type=action_type,
        action_amount_inr=amount,
        affected_entities=entities,
    )


def test_factor_ranges(engine):
    result = engine.assess(_interaction(ActionType.REFUND, amount=250_000, entities=40))
    for value in (
        result.factors.financial_impact,
        result.factors.reversibility,
        result.factors.sensitivity,
        result.factors.blast_radius,
        result.factors.action_automation,
        result.consequence_score,
    ):
        assert 0.0 <= value <= 1.0


def test_weighted_calculation_matches_manual(engine, config):
    result = engine.assess(_interaction(ActionType.REFUND, amount=100_000, entities=10))
    weights = config["consequence_weights"]
    manual = (
        result.factors.financial_impact * weights["financial_impact"]
        + result.factors.reversibility * weights["reversibility"]
        + result.factors.sensitivity * weights["sensitivity"]
        + result.factors.blast_radius * weights["blast_radius"]
        + result.factors.action_automation * weights["action_automation"]
    )
    assert result.consequence_score == pytest.approx(manual, abs=1e-3)


def test_low_consequence_information(engine):
    result = engine.assess(_interaction(ActionType.INFORMATION))
    assert result.severity_band == "low"
    assert result.consequence_score < 0.4


def test_high_consequence_large_financial_action(engine):
    low = engine.assess(_interaction(ActionType.REFUND, amount=1_000, entities=1))
    high = engine.assess(_interaction(ActionType.REFUND, amount=500_000, entities=1))
    assert high.consequence_score > low.consequence_score
    assert high.factors.financial_impact == 1.0


def test_blast_radius_scales_with_entities(engine):
    small = engine.assess(_interaction(ActionType.EXTERNAL_COMMUNICATION, entities=1))
    large = engine.assess(_interaction(ActionType.EXTERNAL_COMMUNICATION, entities=500))
    assert large.factors.blast_radius > small.factors.blast_radius


def test_dominant_factors_reported(engine):
    result = engine.assess(_interaction(ActionType.ACCOUNT_CANCELLATION))
    assert result.dominant_factors
    assert len(result.explanation) > 30


def test_agrees_with_generator_reference(engine, config):
    """Engine must stay numerically consistent with the evaluation dataset."""
    for action, amount, entities in [
        (ActionType.REFUND, 12_345.0, 42),
        (ActionType.EXTERNAL_COMMUNICATION, 0.0, 350),
        (ActionType.INFORMATION, 0.0, 1),
        (ActionType.RECOMMENDATION, 0.0, 800),
    ]:
        reference = compute_consequence_factors(
            action, amount, entities, config["consequence_weights"]
        )
        got = engine.assess(_interaction(action, amount=amount, entities=entities)).factors
        assert got.consequence_score == pytest.approx(reference.consequence_score, abs=1e-3)
        assert got.reversibility == pytest.approx(reference.reversibility, abs=1e-3)


def test_deterministic(engine):
    a = engine.assess(_interaction(ActionType.REFUND, amount=50_000, entities=5)).model_dump()
    b = engine.assess(_interaction(ActionType.REFUND, amount=50_000, entities=5)).model_dump()
    assert a == b


def test_does_not_read_ground_truth(config):
    rng = random.Random(config["seed"])
    cases = generate_evaluation_cases(config, rng)
    row = cases[0]
    interaction = Interaction.model_validate({k: row[k] for k in Interaction.model_fields})
    # assess only takes an Interaction; ground-truth keys are not on it.
    result = assess_consequence(interaction, config)
    assert 0.0 <= result.consequence_score <= 1.0
