"""
Phase 11 — one-click enterprise demo + What-If / policy playground.

Both features run over the REAL pipeline / calibration machinery. The
counterfactual is expensive, so the slow calls are cached in module-scoped
fixtures (one PASS-oriented run, one deliberately-infeasible run).
"""

from __future__ import annotations

import copy

import pytest

from api.service import ControlPlaneService
from enterprise.schemas import EnterpriseDemoResult, WhatIfResult
from settings import load_settings

_DEEP_THRESHOLD = 0.35  # production value that must never move


@pytest.fixture(scope="module")
def demo_service():
    return ControlPlaneService(fit_cost_baseline=True)


@pytest.fixture(scope="module")
def demo_result(demo_service) -> EnterpriseDemoResult:
    return demo_service.enterprise_demo(with_counterfactual=True)


@pytest.fixture(scope="module")
def whatif_pass() -> WhatIfResult:
    return ControlPlaneService(fit_cost_baseline=True).enterprise_whatif(
        control="deep_verification_risk_threshold", minimum_recall=0.90
    )


@pytest.fixture(scope="module")
def whatif_infeasible() -> WhatIfResult:
    return ControlPlaneService(fit_cost_baseline=True).enterprise_whatif(
        minimum_recall=0.999, minimum_precision=0.99
    )


# ------------------------------------------------------------------ demo


def test_demo_is_bounded_and_has_nine_steps(demo_result):
    assert len(demo_result.steps) == 9
    assert [s.step for s in demo_result.steps] == list(range(1, 10))
    assert [s.title for s in demo_result.steps] == [
        "AI TRAFFIC", "RISK DETECTION", "FAST / DEEP VERIFICATION", "CONTROL DECISION",
        "INCIDENT INTELLIGENCE", "PATTERN / DRIFT DETECTION", "GOVERNANCE RECOMMENDATION",
        "COUNTERFACTUAL SAFETY CHECK", "HUMAN APPROVAL",
    ]


def test_demo_uses_real_pipeline_metrics(demo_service, demo_result):
    # step-1 traffic snapshot, then the demo seeds a bounded amount of extra
    # higher-risk traffic — so the headline count is <= the final trace count.
    assert demo_result.interactions == demo_result.steps[0].metrics["interactions"]
    assert 0 < demo_result.interactions <= len(demo_service.all_traces())
    intel = demo_service.incident_intelligence()
    assert demo_result.incidents == intel.total_incidents
    assert demo_result.potential_drift == intel.drift.potential_drift_count
    assert demo_result.counterfactual_safety in ("PASS", "FAIL", "NO_CANDIDATE", "NOT_RUN")


def test_demo_approval_is_evaluation_only(demo_result):
    assert demo_result.approval_status in (None, "APPROVED_FOR_EVALUATION")
    assert demo_result.production_configuration_status == "UNCHANGED"
    joined = " ".join(demo_result.notes).lower()
    assert "no deployment path" in joined
    assert "never modified" in joined or "unchanged" in joined


def test_demo_does_not_change_production_config():
    s = ControlPlaneService(fit_cost_baseline=True)
    before = copy.deepcopy(load_settings())
    s.enterprise_demo(with_counterfactual=True)
    assert load_settings() == before
    assert load_settings()["verification"]["deep_verification_risk_threshold"] == _DEEP_THRESHOLD


def test_demo_is_deterministic():
    a = ControlPlaneService(fit_cost_baseline=True).enterprise_demo(with_counterfactual=False)
    b = ControlPlaneService(fit_cost_baseline=True).enterprise_demo(with_counterfactual=False)

    def _strip(n):
        if isinstance(n, dict):
            return {k: _strip(v) for k, v in n.items()
                    if not (k.endswith("_ms") or k == "generated_at")}
        if isinstance(n, list):
            return [_strip(x) for x in n]
        return n

    assert _strip(a.model_dump(mode="json")) == _strip(b.model_dump(mode="json"))


def test_demo_no_raw_pii(demo_result):
    blob = demo_result.model_dump_json()
    for needle in ("ACC-227763", "karan.mehta@example-test.com", "matched_text",
                   "ground_truth", "expected_decision"):
        assert needle not in blob


# ------------------------------------------------------------------ what-if


def test_whatif_current_vs_candidate_rows(whatif_pass):
    assert isinstance(whatif_pass, WhatIfResult)
    metrics = {m.metric for m in whatif_pass.metrics}
    assert {"recall", "precision", "false_positive_rate", "missed_risk_rate",
            "fast_rate", "deep_rate", "human_review_rate", "average_latency_ms"} == metrics
    assert whatif_pass.control == "deep_verification_risk_threshold"
    assert whatif_pass.current_configuration
    assert whatif_pass.safety_constraints
    assert whatif_pass.safety_status in ("PASS", "NO_CANDIDATE")
    assert whatif_pass.interpretation


def test_whatif_safety_first_and_candidate_presence(whatif_pass):
    if whatif_pass.safety_status == "PASS":
        assert whatif_pass.candidate_configuration is not None
        assert whatif_pass.candidate_decision_distribution
    else:
        assert whatif_pass.candidate_configuration is None


def test_whatif_interpretation_is_derived_from_metrics(whatif_pass):
    assert whatif_pass.interpretation
    if whatif_pass.safety_status != "PASS":
        return
    recall = next(m for m in whatif_pass.metrics if m.metric == "recall")
    missed = next(m for m in whatif_pass.metrics if m.metric == "missed_risk_rate")
    worse = (recall.candidate < recall.current) or (missed.candidate > missed.current)
    assert ("WORSENS" in whatif_pass.interpretation) == worse


def test_whatif_infeasible_yields_no_candidate(whatif_infeasible):
    assert whatif_infeasible.safety_status == "NO_CANDIDATE"
    assert whatif_infeasible.candidate_configuration is None
    assert "No safe configuration recommended" in whatif_infeasible.interpretation
    assert "NOT relaxed" in whatif_infeasible.interpretation


def test_whatif_rejects_unknown_control():
    s = ControlPlaneService(fit_cost_baseline=False)
    with pytest.raises(ValueError):
        s.enterprise_whatif(control="not_a_real_control")


def test_whatif_does_not_modify_config():
    s = ControlPlaneService(fit_cost_baseline=True)
    before = copy.deepcopy(load_settings())
    s.enterprise_whatif(minimum_recall=0.90)
    assert load_settings() == before
    assert load_settings()["verification"]["deep_verification_risk_threshold"] == _DEEP_THRESHOLD


def test_whatif_disclaimer_mentions_simulation(whatif_pass):
    assert "SIMULATION" in whatif_pass.disclaimer
    assert "does NOT modify" in whatif_pass.disclaimer
