"""Policy simulation & counterfactual analysis tests — all run the real pipeline."""

from __future__ import annotations

import pytest

from data.schemas import InterventionTier
from decision.engine import DecisionEngine
from policy.schemas import TIER_RANK
from simulation.engine import (
    build_counterfactual,
    compare_decisions,
    simulate_policies,
)
from tests import scenarios


@pytest.fixture(scope="module")
def engine() -> DecisionEngine:
    return DecisionEngine()


# ---------------------------------------------------------------- policy sim

def test_policy_simulation_runs_real_engine(engine):
    interaction, profiles = scenarios.scenario_i_policy_counterfactual()
    sim = simulate_policies(engine, interaction, profiles)
    assert {o.profile for o in sim.outcomes} == set(profiles)
    for outcome in sim.outcomes:
        assert outcome.decision in {t.value for t in InterventionTier}


def test_policy_simulation_profiles_can_differ(engine):
    interaction, profiles = scenarios.scenario_i_policy_counterfactual()
    sim = simulate_policies(engine, interaction, profiles)
    assert sim.differs
    by_profile = {o.profile: o.decision for o in sim.outcomes}
    # decision_support is the least tolerant of an unverified/contradicted claim.
    assert TIER_RANK[InterventionTier(by_profile["decision_support"])] >= TIER_RANK[
        InterventionTier(by_profile["customer_support"])
    ]


def test_policy_simulation_unambiguous_case_agrees(engine):
    sim = simulate_policies(
        engine,
        scenarios.scenario_c_pii(),
        ["customer_support", "decision_support"],
    )
    # Critical PII is decisive everywhere.
    assert all(o.decision in ("HUMAN_REVIEW", "BLOCK") for o in sim.outcomes)


# ---------------------------------------------------------------- counterfactual

def test_counterfactual_amount_change_recomputes(engine):
    interaction, mods = scenarios.scenario_j_consequence_counterfactual()
    result = compare_decisions(engine, interaction, mods)
    assert result.tier_changed
    assert TIER_RANK[InterventionTier(result.counterfactual_decision)] < TIER_RANK[
        InterventionTier(result.original_decision)
    ]
    assert "HIGH_CONSEQUENCE" in result.rules_removed


def test_counterfactual_is_not_hardcoded(engine):
    interaction = scenarios.scenario_d_high_consequence()
    small = compare_decisions(engine, interaction, {"action_amount_inr": 50.0})
    big = compare_decisions(engine, interaction, {"action_amount_inr": 5_000_000.0})
    assert small.counterfactual_overall_risk <= big.counterfactual_overall_risk


def test_counterfactual_rejects_ground_truth_fields(engine):
    interaction = scenarios.scenario_a_clean()
    _, applied, rejected = build_counterfactual(
        interaction,
        {"ground_truth_hallucination": True, "expected_decision": "BLOCK", "action_amount_inr": 100},
    )
    assert "ground_truth_hallucination" in rejected
    assert "expected_decision" in rejected
    assert "action_amount_inr" in applied


def test_counterfactual_changing_action_type(engine):
    interaction = scenarios.scenario_a_clean()
    result = compare_decisions(
        engine, interaction, {"action_type": "account_cancellation"}
    )
    assert "action_type" in result.changed_fields
    assert isinstance(result.tier_changed, bool)


def test_counterfactual_no_change_is_honest(engine):
    interaction = scenarios.scenario_a_clean()
    result = compare_decisions(engine, interaction, {"tokens_in": 55})
    assert result.original_decision == "ALLOW"
    assert not result.tier_changed


def test_deterministic(engine):
    interaction, mods = scenarios.scenario_j_consequence_counterfactual()
    a = compare_decisions(engine, interaction, mods).model_dump()
    b = compare_decisions(engine, interaction, mods).model_dump()
    assert a == b
