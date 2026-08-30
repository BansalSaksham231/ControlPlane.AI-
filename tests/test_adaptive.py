"""
Phase 10 — Adaptive Guardrails: recommendations, counterfactual, approval.

Recommendations are RECOMMENDATION ONLY. Approval => APPROVED_FOR_EVALUATION.
Nothing here writes config/settings.yaml.
"""

from __future__ import annotations

import copy
import pathlib
import random

import pytest

from adaptive.report import build_adaptive_governance_report
from adaptive.schemas import RecommendationStatus, RecommendationType
from adaptive.service import AdaptiveGovernanceService, RecommendationNotFound
from api.service import ControlPlaneService
from data.generator import generate_interactions
from data.schemas import InterventionTier
from incident.report import build_incident_intelligence
from settings import load_settings
from tests import scenarios

_AD_DIR = pathlib.Path(__file__).resolve().parent.parent / "adaptive"
_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")


@pytest.fixture(scope="module")
def svc():
    s = ControlPlaneService(fit_cost_baseline=True)
    s.populate_operational_demo(200)
    for t in [t for t in s.all_traces() if t.final_decision.decision is InterventionTier.BLOCK][:4]:
        s.record_governance_action(t.interaction_id, action="ACKNOWLEDGE")
        s.record_governance_action(
            t.interaction_id, action="MODIFY_DECISION",
            comment="hr only", reviewer_decision="HUMAN_REVIEW",
        )
    return s


@pytest.fixture(scope="module")
def report(svc):
    return svc.adaptive_report()


@pytest.fixture(scope="module")
def cf_selection():
    """One real calibration bridge run (slow ~2 min) — shared by counterfactual tests."""
    from adaptive.counterfactual import run_threshold_counterfactual

    return run_threshold_counterfactual(minimum_recall=0.90, minimum_precision=0.0)


# ------------------------------------------------------------------ recommendations


def test_recommendation_generation(report):
    assert report.recommendations
    for r in report.recommendations:
        assert r.rationale and r.proposed_change
        assert r.evidence or r.type is RecommendationType.NO_ACTION
        assert "NOT APPLIED" in r.disclaimer or "never" in r.disclaimer
        assert r.status in (
            RecommendationStatus.RECOMMENDED_FOR_REVIEW,
            RecommendationStatus.SIMULATED,
            RecommendationStatus.APPROVED,
            RecommendationStatus.REJECTED,
        )
    assert report.production_configuration_status == "UNCHANGED"


def test_no_recommendation_when_no_patterns():
    rep = build_adaptive_governance_report(build_incident_intelligence([], [], []))
    assert len(rep.recommendations) == 1
    assert rep.recommendations[0].type is RecommendationType.NO_ACTION


def test_recommendation_ids_are_deterministic(svc):
    a = svc.adaptive_report()
    b = svc.adaptive_report()
    assert [r.recommendation_id for r in a.recommendations] == [
        r.recommendation_id for r in b.recommendations
    ]
    assert a.model_dump(mode="json")["recommendations"] == b.model_dump(mode="json")["recommendations"]


def test_review_policy_carries_governance_signal_framing(report):
    policy_recs = [r for r in report.recommendations if r.type is RecommendationType.REVIEW_POLICY]
    assert policy_recs
    joined = " ".join(r.rationale for r in policy_recs)
    assert "governance signal" in joined and "not evidence" in joined.lower()


# ------------------------------------------------------------------ counterfactual (slow)


def test_counterfactual_candidate_evaluation(svc, cf_selection):
    from adaptive.counterfactual import selection_to_evaluation

    ev = selection_to_evaluation(cf_selection)
    # safety fields are populated and bounded
    for v in (ev.current_recall, ev.candidate_recall, ev.current_precision, ev.candidate_precision):
        assert v is None or 0.0 <= v <= 1.0
    assert isinstance(ev.safety_passed, bool)
    assert ev.safety_constraints  # SafetyConstraints applied FIRST
    # decision distributions are present
    assert sum(ev.current_decision_distribution.values()) > 0


def test_threshold_recommendation_is_enriched_by_counterfactual(svc, cf_selection):
    rep = build_adaptive_governance_report(
        svc.incident_intelligence(), calibration_selection=cf_selection
    )
    thr = [r for r in rep.recommendations if r.type is RecommendationType.REVIEW_VERIFICATION_THRESHOLD]
    assert thr, "decision_support (100% DEEP) should yield a threshold recommendation"
    r = thr[0]
    assert r.simulation_result is not None
    assert r.expected_tradeoff
    if r.simulation_result.safety_passed:
        assert r.candidate_configuration is not None
        assert r.status is RecommendationStatus.SIMULATED


def test_counterfactual_safety_first_no_candidate_under_strict_constraints():
    from adaptive.counterfactual import (
        run_threshold_counterfactual,
        selection_to_evaluation,
    )

    strict = run_threshold_counterfactual(minimum_recall=0.999, minimum_precision=0.99)
    ev = selection_to_evaluation(strict)
    assert ev.safety_passed is False
    assert ev.candidate_configuration is None
    assert "NOT relaxed" in strict.selection_reason or "NO_ELIGIBLE" in strict.status


def test_counterfactual_does_not_modify_config(cf_selection):
    before = copy.deepcopy(load_settings())
    from adaptive.counterfactual import selection_to_evaluation

    selection_to_evaluation(cf_selection)
    assert load_settings() == before
    assert load_settings()["verification"]["deep_verification_risk_threshold"] == 0.35


# ------------------------------------------------------------------ approval gate


def test_approve_sets_approved_for_evaluation(svc):
    rec = next(r for r in svc.adaptive_recommendations() if r.type is not RecommendationType.NO_ACTION)
    updated = svc.adaptive_approve(rec.recommendation_id, actor="r1", comment="ok")
    assert updated.status is RecommendationStatus.APPROVED
    assert updated.approval is not None
    assert updated.approval.decision == "APPROVED_FOR_EVALUATION"
    assert "does NOT apply" in updated.approval.disclaimer
    # persisted across a fresh report
    again = svc.adaptive_get_recommendation(rec.recommendation_id)
    assert again.status is RecommendationStatus.APPROVED


def test_reject_sets_rejected(svc):
    recs = [r for r in svc.adaptive_recommendations() if r.type is not RecommendationType.NO_ACTION]
    target = recs[-1]
    updated = svc.adaptive_reject(target.recommendation_id, actor="r2", comment="no")
    assert updated.status is RecommendationStatus.REJECTED


def test_approve_unknown_id_raises(svc):
    with pytest.raises(RecommendationNotFound):
        svc.adaptive_approve("REC-DOES-NOT-EXIST")


def test_approval_never_changes_production_config():
    s = ControlPlaneService(fit_cost_baseline=False)
    s.populate_operational_demo(60)
    before = copy.deepcopy(load_settings())
    rec = next(r for r in s.adaptive_recommendations() if r.type is not RecommendationType.NO_ACTION)
    s.adaptive_approve(rec.recommendation_id)
    assert load_settings() == before
    assert load_settings()["verification"]["deep_verification_risk_threshold"] == 0.35


def test_no_deploy_or_apply_status_exists():
    for member in RecommendationStatus:
        assert "DEPLOY" not in member.value.upper()
        assert member.value.upper() != "APPLIED_TO_PRODUCTION"
    assert RecommendationStatus.APPROVED.value == "APPROVED_FOR_EVALUATION"


# ------------------------------------------------------------------ safety guards


def test_adaptive_source_never_writes_settings_or_ground_truth():
    for path in _AD_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in ("yaml.dump", "yaml.safe_dump", ".write_text(", ".write(",
                      'open(str', '"w")', "'w')", "import evaluation", "from evaluation",
                      "ground_truth_", "expected_decision", "final_outcome"):
            assert token not in text, f"{path.name}: {token}"


def test_adaptive_does_not_orchestrate_detectors():
    banned = (
        "DecisionEngine(", "PerformanceDetector(", "ResponsibilityDetector(",
        "CostDetector(", "VerificationRouter(", "RiskFusionEngine(", "PolicyEngine(",
        ".detect(", ".evaluate(", ".fuse_scores(", ".route(", ".assess(",
    )
    for path in _AD_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name}: {token}"


def test_report_has_no_raw_pii(report):
    blob = report.model_dump_json()
    for needle in _KNOWN_PII:
        assert needle not in blob
    assert "matched_text" not in blob and "ground_truth" not in blob


def test_full_report_production_config_unchanged(svc):
    before = copy.deepcopy(load_settings())
    svc.adaptive_report()
    assert load_settings() == before


# ------------------------------------------------------------------ scenarios A-J


def test_core_scenarios_unchanged():
    from evaluation.evaluation import build_engine

    engine = build_engine()
    got = {
        k: engine.evaluate(f(), timestamp=None, record_session=False).final_decision.decision.value
        for k, f in {
            "A": scenarios.scenario_a_clean, "B": scenarios.scenario_b_hallucination,
            "C": scenarios.scenario_c_pii, "D": scenarios.scenario_d_high_consequence,
            "E": scenarios.scenario_e_multi_risk,
        }.items()
    }
    assert got == {"A": "ALLOW", "B": "VERIFY", "C": "BLOCK", "D": "VERIFY", "E": "BLOCK"}
