"""Policy Engine tests — tiers, application profiles, consequence-awareness."""

from __future__ import annotations

import pytest

from data.schemas import InterventionTier
from policy.engine import PolicyEngine, run_policy
from policy.schemas import PolicyInput


@pytest.fixture(scope="module")
def engine():
    return PolicyEngine()


def _signals(**overrides) -> PolicyInput:
    base = dict(
        application="customer_support",
        action_type="information",
        overall_risk=0.1,
        performance_risk=0.1,
        responsibility_risk=0.1,
        cost_risk=0.1,
        consequence_score=0.1,
        confidence=0.8,
        performance_status="SUPPORTED",
    )
    base.update(overrides)
    return PolicyInput(**base)


def test_allow_for_low_risk(engine):
    d = engine.decide(_signals())
    assert d.proposed_tier == InterventionTier.ALLOW
    assert d.requires_human_review is False


def test_annotate_for_moderate_risk(engine):
    d = engine.decide(_signals(overall_risk=0.35, performance_risk=0.35))
    assert d.proposed_tier == InterventionTier.ANNOTATE


def test_verify_for_unverified_evidence(engine):
    d = engine.decide(
        _signals(overall_risk=0.5, performance_risk=0.55, performance_status="UNVERIFIED")
    )
    assert d.proposed_tier == InterventionTier.VERIFY


def test_contradiction_forces_at_least_verify(engine):
    d = engine.decide(
        _signals(
            overall_risk=0.3,
            performance_risk=0.74,
            performance_status="CONTRADICTED",
            consequence_score=0.1,
        )
    )
    assert d.proposed_tier == InterventionTier.VERIFY
    assert "PERFORMANCE_CONTRADICTION" in d.triggered_rules


def test_high_consequence_escalates_same_risk(engine):
    low_cons = engine.decide(
        _signals(overall_risk=0.55, performance_risk=0.7, performance_status="PARTIALLY_SUPPORTED", consequence_score=0.1)
    )
    high_cons = engine.decide(
        _signals(overall_risk=0.55, performance_risk=0.7, performance_status="PARTIALLY_SUPPORTED", consequence_score=0.9)
    )
    from policy.schemas import TIER_RANK

    assert TIER_RANK[high_cons.proposed_tier] > TIER_RANK[low_cons.proposed_tier]
    assert "HIGH_CONSEQUENCE" in high_cons.triggered_rules


def test_critical_pii_blocks_in_customer_support(engine):
    d = engine.decide(
        _signals(
            overall_risk=0.6,
            responsibility_risk=0.85,
            contains_critical_pii=True,
            critical_pii_types=["full_contact_profile"],
        )
    )
    assert d.proposed_tier == InterventionTier.BLOCK
    assert "CRITICAL_PII" in d.triggered_rules


def test_severe_toxicity_blocks(engine):
    d = engine.decide(_signals(overall_risk=0.5, responsibility_risk=0.7, toxicity_risk=0.95))
    assert d.proposed_tier == InterventionTier.BLOCK
    assert "SEVERE_TOXICITY" in d.triggered_rules


def test_extreme_multi_dimension_blocks(engine):
    d = engine.decide(
        _signals(
            overall_risk=0.9,
            performance_risk=0.95,
            responsibility_risk=0.95,
            performance_status="CONTRADICTED",
            consequence_score=0.8,
        )
    )
    assert d.proposed_tier == InterventionTier.BLOCK


def test_application_profiles_differ(engine):
    """Same signals, different application -> different tolerance."""
    signals = dict(
        overall_risk=0.5,
        performance_risk=0.74,
        performance_status="CONTRADICTED",
        consequence_score=0.2,
    )
    cs = engine.decide(_signals(application="customer_support", **signals))
    ds = engine.decide(_signals(application="decision_support", **signals))
    ika = engine.decide(_signals(application="internal_knowledge_assistant", **signals))
    from policy.schemas import TIER_RANK

    # decision_support is the least risk-tolerant, internal KB the most.
    assert TIER_RANK[ds.proposed_tier] >= TIER_RANK[cs.proposed_tier]
    assert TIER_RANK[cs.proposed_tier] >= TIER_RANK[ika.proposed_tier]


def test_low_confidence_prefers_oversight(engine):
    confident = engine.decide(_signals(overall_risk=0.3, performance_risk=0.3, confidence=0.8))
    unsure = engine.decide(_signals(overall_risk=0.3, performance_risk=0.3, confidence=0.2))
    from policy.schemas import TIER_RANK

    assert TIER_RANK[unsure.proposed_tier] >= TIER_RANK[confident.proposed_tier]


def test_rule_trace_present_and_explained(engine):
    d = engine.decide(
        _signals(overall_risk=0.6, performance_risk=0.74, performance_status="CONTRADICTED", consequence_score=0.9)
    )
    assert len(d.rule_trace) >= 2
    assert all(len(entry.detail) > 5 for entry in d.rule_trace)
    assert "because" in d.explanation.lower()


def test_thresholds_come_from_config():
    """No numeric policy thresholds are hard-coded in the engine module."""
    import inspect

    import policy.engine as engine_module

    source = inspect.getsource(engine_module)
    # allow small structural constants only; no 0.xx risk thresholds
    assert "overall_risk <= 0." not in source
    assert "consequence_score >= 0." not in source


def test_deterministic(engine):
    a = engine.decide(_signals(overall_risk=0.6, performance_risk=0.7)).model_dump()
    b = engine.decide(_signals(overall_risk=0.6, performance_risk=0.7)).model_dump()
    assert a == b


def test_convenience_wrapper():
    d = run_policy(_signals())
    assert d.proposed_tier == InterventionTier.ALLOW
