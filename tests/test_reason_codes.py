"""Structured reason-code tests — deterministic and traceable to evidence."""

from __future__ import annotations

from datetime import datetime

import pytest

from common.reason_codes import DESCRIPTIONS, ReasonCode, describe
from decision.engine import DecisionEngine
from tests import scenarios

TS = datetime(2026, 8, 21, 12, 0, 0)


@pytest.fixture(scope="module")
def engine() -> DecisionEngine:
    return DecisionEngine()


def _codes(engine, factory):
    return engine.evaluate(
        factory(), timestamp=TS, record_session=False
    ).final_decision.reason_codes


def test_every_reason_code_has_a_description():
    for code in ReasonCode:
        assert code in DESCRIPTIONS
        assert len(DESCRIPTIONS[code]) > 10


def test_clean_response_has_no_reason_codes(engine):
    assert _codes(engine, scenarios.scenario_a_clean) == []


def test_contradiction_reason_code(engine):
    codes = _codes(engine, scenarios.scenario_b_hallucination)
    assert "CONTRADICTED_EVIDENCE" in codes or "HIGH_PERFORMANCE_RISK" in codes


def test_pii_reason_codes(engine):
    codes = _codes(engine, scenarios.scenario_c_pii)
    assert "CRITICAL_PII" in codes
    assert "PII_EXPOSURE" in codes


def test_high_consequence_reason_codes(engine):
    codes = _codes(engine, scenarios.scenario_d_high_consequence)
    assert "HIGH_CONSEQUENCE" in codes
    assert "HIGH_FINANCIAL_IMPACT" in codes


def test_multi_risk_reason_code(engine):
    codes = _codes(engine, scenarios.scenario_e_multi_risk)
    assert "MULTI_RISK" in codes
    assert "CRITICAL_PII" in codes


def test_cost_anomaly_reason_codes(engine):
    codes = _codes(engine, scenarios.scenario_f_cost_anomaly)
    assert any(c in codes for c in ("COST_SPIKE", "RETRY_ANOMALY", "LATENCY_ANOMALY"))


def test_reason_codes_are_deterministic(engine):
    a = _codes(engine, scenarios.scenario_e_multi_risk)
    b = _codes(engine, scenarios.scenario_e_multi_risk)
    assert a == b


def test_reason_codes_traceable_to_policy_and_criticality(engine):
    trace = engine.evaluate(
        scenarios.scenario_e_multi_risk(), timestamp=TS, record_session=False
    )
    fd_codes = set(trace.final_decision.reason_codes)
    source_codes = set(trace.policy.reason_codes) | set(trace.criticality.reason_codes)
    # Everything on the decision came from a real sub-component (or the
    # cost/session cross-cutting mappers).
    cross_cutting = {"MULTI_RISK", "SESSION_ESCALATION", "COST_SPIKE", "RETRY_ANOMALY",
                     "TOOL_LOOP", "LATENCY_ANOMALY", "HIGH_PERFORMANCE_RISK"}
    assert fd_codes <= source_codes | cross_cutting


def test_describe_helper():
    assert describe("CRITICAL_PII") == DESCRIPTIONS[ReasonCode.CRITICAL_PII]
    assert describe("NOT_A_CODE") == "NOT_A_CODE"


def test_decision_drivers_recorded(engine):
    trace = engine.evaluate(
        scenarios.scenario_d_high_consequence(), timestamp=TS, record_session=False
    )
    assert trace.decision_drivers
    assert any(d.rule == "HIGH_CONSEQUENCE" for d in trace.decision_drivers)
    for driver in trace.decision_drivers:
        assert len(driver.detail) > 5
