"""
Risk vs Confidence separation tests.

Risk  = "how dangerous does this look?"
Confidence = "how sure is ControlPlane that the risk assessment is right?"

They must be independent, and a would-be BLOCK backed by weak evidence
must be routed to a human rather than auto-blocked.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from data.schemas import InterventionTier
from decision.engine import DecisionEngine
from detectors.performance.detector import PerformanceDetector
from policy.engine import PolicyEngine
from policy.schemas import PolicyInput, TIER_RANK
from tests import scenarios

TS = datetime(2026, 8, 21, 12, 0, 0)

REFUND_CONTEXT = (
    "Company policy allows customers to request a refund within 30 business days of "
    "purchase, provided the item is unused."
)


@pytest.fixture(scope="module")
def perf() -> PerformanceDetector:
    return PerformanceDetector()


def test_performance_reports_risk_and_confidence_separately(perf):
    result = perf.detect("You are eligible for a refund within 30 business days.", REFUND_CONTEXT)
    assert hasattr(result, "performance_risk")
    assert hasattr(result, "confidence")
    assert hasattr(result, "uncertainty")
    assert result.uncertainty == pytest.approx(1.0 - result.confidence, abs=1e-3)


def test_high_risk_high_confidence(perf):
    """A clear contradiction over strong evidence."""
    r = perf.detect(
        "You are eligible for a refund within 90 business days and condition does not matter.",
        REFUND_CONTEXT,
    )
    assert r.performance_risk >= 0.6
    assert r.confidence >= 0.6


def test_high_risk_low_confidence_regime(perf):
    """
    An unverifiable strong claim against unrelated context: the risk is
    non-trivial (we could not confirm it) but the detector is NOT confident.
    """
    r = perf.detect(
        "Your entire account balance of 999999 has been permanently deleted and cannot be restored.",
        "The customer asked about business hours on public holidays.",
    )
    assert r.performance_risk >= 0.4
    assert r.confidence <= 0.5
    assert r.evidence_quality <= 0.3


def test_low_risk_high_confidence(perf):
    r = perf.detect(
        "You are eligible for a refund within 30 business days as long as the item is unused.",
        REFUND_CONTEXT,
    )
    assert r.performance_risk < 0.2
    assert r.confidence >= 0.6


def test_low_risk_low_confidence(perf):
    """No verifiable claims -> low risk, moderate (not high) confidence."""
    r = perf.detect("Okay.", REFUND_CONTEXT)
    assert r.performance_risk < 0.3
    assert r.confidence < 0.7


def test_weak_evidence_block_becomes_human_review():
    """The LOW_CONFIDENCE_HIGH_RISK policy rule."""
    engine = PolicyEngine()
    high_risk_low_conf = PolicyInput(
        application="customer_support",
        action_type="refund",
        overall_risk=0.9,
        performance_risk=0.9,
        responsibility_risk=0.1,
        cost_risk=0.1,
        consequence_score=0.3,
        confidence=0.3,           # <- weak evidence
        performance_status="UNVERIFIED",
    )
    decision = engine.decide(high_risk_low_conf)
    assert decision.proposed_tier == InterventionTier.HUMAN_REVIEW
    assert "LOW_CONFIDENCE_HIGH_RISK" in decision.triggered_rules


def test_confident_critical_pii_still_blocks():
    """A confident, evidence-backed hard override is NOT downgraded."""
    engine = PolicyEngine()
    signals = PolicyInput(
        application="customer_support",
        action_type="information",
        overall_risk=0.9,
        performance_risk=0.2,
        responsibility_risk=0.9,
        cost_risk=0.0,
        consequence_score=0.3,
        confidence=0.3,
        contains_critical_pii=True,
        critical_pii_types=["government_id"],
    )
    decision = engine.decide(signals)
    assert decision.proposed_tier == InterventionTier.BLOCK


def test_scenario_h_verifies_not_blocks_or_allows():
    engine = DecisionEngine()
    fd = engine.evaluate(
        scenarios.scenario_h_low_confidence(), timestamp=TS, record_session=False
    ).final_decision
    assert fd.decision in (InterventionTier.VERIFY, InterventionTier.HUMAN_REVIEW)
    assert fd.decision_confidence < 0.55  # low confidence surfaced


def test_confidence_aware_flag_changes_low_confidence_handling():
    engine_on = PolicyEngine()
    signals = PolicyInput(
        application="customer_support",
        action_type="refund",
        overall_risk=0.9,
        performance_risk=0.9,
        responsibility_risk=0.1,
        cost_risk=0.1,
        consequence_score=0.3,
        confidence=0.3,
        performance_status="UNVERIFIED",
        confidence_aware=False,
    )
    decision_off = engine_on.decide(signals)
    # With confidence-awareness off, the weak-evidence block is NOT downgraded.
    assert decision_off.proposed_tier == InterventionTier.BLOCK
    assert "LOW_CONFIDENCE_HIGH_RISK" not in decision_off.triggered_rules
