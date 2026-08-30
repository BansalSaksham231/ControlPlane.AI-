"""Decision Engine tests — full pipeline, explanations, triggered rules, scenarios."""

from __future__ import annotations

from datetime import datetime

import pytest

from data.schemas import FinalDecision, InterventionTier
from decision.engine import DecisionEngine, evaluate_interaction
from decision.schemas import DecisionTrace
from policy.schemas import TIER_RANK
from tests import scenarios

FIXED_TS = datetime(2026, 8, 21, 12, 0, 0)


@pytest.fixture(scope="module")
def engine():
    return DecisionEngine()


def _evaluate(engine, interaction):
    return engine.evaluate(interaction, timestamp=FIXED_TS, record_session=False)


def test_pipeline_produces_full_trace(engine):
    trace = _evaluate(engine, scenarios.scenario_a_clean())
    assert isinstance(trace, DecisionTrace)
    assert isinstance(trace.final_decision, FinalDecision)
    for part in (trace.performance, trace.responsibility, trace.cost, trace.fusion, trace.consequence, trace.policy):
        assert part is not None
    assert trace.stage_latency_ms  # measured per stage
    assert trace.latency_ms >= 0.0


def test_final_decision_fields_populated(engine):
    fd = _evaluate(engine, scenarios.scenario_c_pii()).final_decision
    assert 0.0 <= fd.overall_risk <= 1.0
    assert 0.0 <= fd.decision_confidence <= 1.0
    assert fd.decision in InterventionTier
    assert fd.triggered_rules
    assert fd.timestamp == FIXED_TS
    assert fd.consequence.consequence_score >= 0.0


def test_scenario_a_clean_allow(engine):
    fd = _evaluate(engine, scenarios.scenario_a_clean()).final_decision
    assert fd.decision == InterventionTier.ALLOW
    assert fd.overall_risk < 0.2


def test_scenario_b_hallucination_escalates(engine):
    trace = _evaluate(engine, scenarios.scenario_b_hallucination())
    fd = trace.final_decision
    assert fd.decision in (InterventionTier.VERIFY, InterventionTier.HUMAN_REVIEW, InterventionTier.BLOCK)
    assert trace.performance.status.value in ("CONTRADICTED", "PARTIALLY_SUPPORTED")


def test_scenario_c_pii_high_responsibility(engine):
    trace = _evaluate(engine, scenarios.scenario_c_pii())
    fd = trace.final_decision
    assert fd.responsibility_risk >= 0.6
    assert fd.decision in (InterventionTier.HUMAN_REVIEW, InterventionTier.BLOCK)
    assert "CRITICAL_PII" in fd.triggered_rules
    # audit summary must not leak raw PII
    summary = trace.audit_summary()
    assert "karan.mehta@example-test.com" not in str(summary)


def test_scenario_d_consequence_drives_decision(engine):
    """Moderate risk + high-consequence financial action -> stronger than ANNOTATE."""
    fd = _evaluate(engine, scenarios.scenario_d_high_consequence()).final_decision
    assert TIER_RANK[fd.decision] >= TIER_RANK[InterventionTier.VERIFY]
    assert "HIGH_CONSEQUENCE" in fd.triggered_rules


def test_scenario_e_multi_risk_blocks_or_reviews(engine):
    trace = _evaluate(engine, scenarios.scenario_e_multi_risk())
    fd = trace.final_decision
    assert fd.decision in (InterventionTier.HUMAN_REVIEW, InterventionTier.BLOCK)
    # multiple dimensions contributed
    assert fd.responsibility_risk > 0.3
    assert trace.performance.status.value != "SUPPORTED"


def test_scenario_f_cost_anomaly_flagged_not_safety_blocked(engine):
    trace = _evaluate(engine, scenarios.scenario_f_cost_anomaly())
    fd = trace.final_decision
    assert fd.cost_risk >= 0.7
    # cost-only anomaly should not hard-block on safety grounds
    assert fd.decision != InterventionTier.BLOCK
    assert "COST_ONLY_CAP" in fd.triggered_rules


def test_explanation_is_human_readable(engine):
    fd = _evaluate(engine, scenarios.scenario_e_multi_risk()).final_decision
    assert len(fd.explanation) > 40
    assert "because" in fd.explanation.lower()
    # no bare threshold-comparison jargon
    assert "risk_score" not in fd.explanation


def test_deterministic_pipeline(engine):
    a = _evaluate(engine, scenarios.scenario_b_hallucination())
    b = _evaluate(engine, scenarios.scenario_b_hallucination())

    def _stable(trace: DecisionTrace) -> dict:
        dumped = trace.model_dump(mode="json")
        _strip_latency(dumped)
        return dumped

    def _strip_latency(obj):
        # Measured wall-clock latency legitimately varies run to run; the
        # decision itself must be deterministic.
        if isinstance(obj, dict):
            for key in list(obj):
                if key == "latency" or key.endswith("_ms") or key == "stage_latency_ms":
                    obj.pop(key, None)
                else:
                    _strip_latency(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                _strip_latency(item)

    assert _stable(a) == _stable(b)


def test_decide_returns_compact_final_decision(engine):
    fd = engine.decide(scenarios.scenario_a_clean(), timestamp=FIXED_TS, record_session=False)
    assert isinstance(fd, FinalDecision)


def test_convenience_wrapper():
    trace = evaluate_interaction(scenarios.scenario_a_clean(), timestamp=FIXED_TS, record_session=False)
    assert isinstance(trace, DecisionTrace)


def test_no_ground_truth_in_pipeline():
    from data.schemas import Interaction

    leak_fields = {
        "ground_truth_hallucination",
        "ground_truth_pii",
        "expected_decision",
        "final_outcome",
    }
    assert leak_fields.isdisjoint(Interaction.model_fields)
