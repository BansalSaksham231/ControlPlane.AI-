"""
Full end-to-end integration test.

    Interaction -> Performance -> Responsibility -> Cost -> Fusion
                -> Consequence -> Session -> Policy -> Decision -> FinalDecision

Asserts the pipeline produces a structurally valid decision for every
demo scenario, and that the audit / feedback path works on top of it.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from api.service import ControlPlaneService
from data.schemas import ConsequenceFactors, FinalDecision, InterventionTier
from decision.engine import DecisionEngine
from decision.schemas import DecisionTrace
from tests import scenarios

FIXED_TS = datetime(2026, 8, 21, 12, 0, 0)


@pytest.fixture(scope="module")
def engine() -> DecisionEngine:
    return DecisionEngine()


@pytest.mark.parametrize("name,factory", list(scenarios.ALL_SINGLE_TURN.items()))
def test_pipeline_produces_valid_final_decision(engine, name, factory):
    trace = engine.evaluate(factory(), timestamp=FIXED_TS, record_session=False)
    assert isinstance(trace, DecisionTrace)

    fd = trace.final_decision
    assert isinstance(fd, FinalDecision)
    assert fd.decision in InterventionTier

    for value in (fd.performance_risk, fd.responsibility_risk, fd.cost_risk, fd.overall_risk, fd.decision_confidence):
        assert 0.0 <= value <= 1.0

    assert isinstance(fd.consequence, ConsequenceFactors)
    assert 0.0 <= fd.consequence.consequence_score <= 1.0
    for factor in (
        fd.consequence.financial_impact,
        fd.consequence.reversibility,
        fd.consequence.sensitivity,
        fd.consequence.blast_radius,
        fd.consequence.action_automation,
    ):
        assert 0.0 <= factor <= 1.0

    assert isinstance(fd.explanation, str) and len(fd.explanation) > 20
    assert isinstance(fd.triggered_rules, list)
    assert fd.timestamp == FIXED_TS

    # every pipeline stage is represented in the trace
    for stage in (trace.performance, trace.responsibility, trace.cost, trace.fusion, trace.consequence, trace.policy):
        assert stage is not None
    assert trace.stage_latency_ms
    assert trace.latency_ms >= 0.0


def test_expected_scenario_decisions(engine):
    """The headline behaviours the demo relies on."""
    def decide(factory):
        return engine.evaluate(factory(), timestamp=FIXED_TS, record_session=False).final_decision.decision

    assert decide(scenarios.scenario_a_clean) == InterventionTier.ALLOW
    assert decide(scenarios.scenario_b_hallucination) in (
        InterventionTier.VERIFY, InterventionTier.HUMAN_REVIEW, InterventionTier.BLOCK
    )
    assert decide(scenarios.scenario_c_pii) in (InterventionTier.HUMAN_REVIEW, InterventionTier.BLOCK)
    assert decide(scenarios.scenario_e_multi_risk) in (InterventionTier.HUMAN_REVIEW, InterventionTier.BLOCK)
    assert decide(scenarios.scenario_f_cost_anomaly) != InterventionTier.BLOCK


def test_multi_turn_escalation():
    from session.manager import SessionManager

    sm = SessionManager()
    engine = DecisionEngine(session_manager=sm)
    tiers = [
        engine.evaluate(turn, timestamp=FIXED_TS).final_decision.decision
        for turn in scenarios.scenario_g_multi_turn()
    ]
    from policy.schemas import TIER_RANK

    assert TIER_RANK[tiers[-1]] > TIER_RANK[tiers[0]]
    assert any(t in (InterventionTier.HUMAN_REVIEW, InterventionTier.BLOCK) for t in tiers)


def test_service_check_audit_feedback_roundtrip():
    service = ControlPlaneService(fit_cost_baseline=False)
    interaction = scenarios.scenario_c_pii()
    trace = service.check(interaction, timestamp=FIXED_TS)

    audit = service.get_audit(interaction.interaction_id)
    assert audit is not None
    assert audit["decision"] == trace.final_decision.decision.value
    # audit summary is redacted
    assert "karan.mehta@example-test.com" not in str(audit)

    record = service.submit_feedback(
        interaction_id=interaction.interaction_id,
        system_decision=None,
        reviewer_decision="VERIFY",
        reason="second opinion",
    )
    assert record.system_decision == trace.final_decision.decision
    assert service.feedback_summary()["total"] == 1


def test_no_ground_truth_reaches_detectors():
    from data.schemas import Interaction

    leak = {
        "ground_truth_hallucination", "ground_truth_pii", "ground_truth_toxicity",
        "ground_truth_bias", "ground_truth_cost_anomaly", "expected_decision", "final_outcome",
    }
    assert leak.isdisjoint(Interaction.model_fields)


def test_parallel_detectors_produce_identical_results():
    """Independent detectors can run concurrently in production without
    changing the outcome (all detectors are deterministic). With the
    verification router this exercises the legacy orchestration path."""
    sequential = DecisionEngine(parallel_detectors=False, use_verification_router=False)
    parallel = DecisionEngine(parallel_detectors=True, use_verification_router=False)

    for factory in scenarios.ALL_SINGLE_TURN.values():
        interaction = factory()
        s = sequential.evaluate(interaction, timestamp=FIXED_TS, record_session=False)
        p = parallel.evaluate(interaction, timestamp=FIXED_TS, record_session=False)

        def _stable(trace):
            d = trace.model_dump(mode="json")
            d.pop("detectors_parallel", None)
            _strip(d)
            return d

        def _strip(obj):
            if isinstance(obj, dict):
                for key in list(obj):
                    if key == "latency" or key.endswith("_ms"):
                        obj.pop(key, None)
                    else:
                        _strip(obj[key])
            elif isinstance(obj, list):
                for v in obj:
                    _strip(v)

        assert _stable(s) == _stable(p)


def test_criticality_amplifies_performance_risk_in_pipeline():
    """A risky claim on a high-criticality action fuses to more risk than
    the same claim on a trivial action."""
    from data.schemas import ActionType

    engine = DecisionEngine()
    low = scenarios.scenario_b_hallucination()
    high = low.model_copy(
        update={
            "action_type": ActionType.REFUND,
            "action_amount_inr": 480_000.0,
            "response": low.response + " Approve the refund of 480000 now.",
        }
    )
    low_trace = engine.evaluate(low, timestamp=FIXED_TS, record_session=False)
    high_trace = engine.evaluate(high, timestamp=FIXED_TS, record_session=False)
    assert (
        high_trace.criticality_weighted_performance_risk
        >= low_trace.criticality_weighted_performance_risk
    )
    assert high_trace.criticality.action_criticality > low_trace.criticality.action_criticality


def test_trace_carries_reason_codes_and_criticality():
    engine = DecisionEngine()
    trace = engine.evaluate(
        scenarios.scenario_e_multi_risk(), timestamp=FIXED_TS, record_session=False
    )
    assert trace.criticality is not None
    assert trace.final_decision.reason_codes
    assert trace.criticality_weighted_performance_risk >= 0.0
    summary = trace.audit_summary()
    assert "reason_codes" in summary
    assert "action_criticality" in summary
