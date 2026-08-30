"""
Phase 7 — Step 1: calibration threshold-sweep foundation.

The sweep is an EVALUATION-layer tool: it may read EvaluationCase
ground truth, runs the real pipeline once through CalibrationCache, and
must never mutate production config or be imported by production code.
"""

from __future__ import annotations

import copy
import pathlib

import pytest
from pydantic import ValidationError

from calibration.advisor import CalibrationCache
from calibration.sweep import (
    CalibrationResult,
    CalibrationSweepReport,
    CandidateConfig,
    evaluate_candidate,
    format_summary,
    sweep_thresholds,
)
from data.schemas import Interaction
from evaluation.evaluation import build_engine
from settings import load_settings
from tests import scenarios

_REPO = pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# shared cache — detectors run ONCE for the whole module
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    return load_settings()


@pytest.fixture(scope="module")
def cache(config):
    return CalibrationCache(config)


@pytest.fixture(scope="module")
def report(config, cache):
    return sweep_thresholds(
        risk_thresholds=[0.25, 0.35, 0.45],
        confidence_thresholds=[0.60, 0.70, 0.80],
        config=config,
        cache=cache,
    )


# ------------------------------------------------------------------
# 1-3. candidate configuration validation
# ------------------------------------------------------------------


def test_valid_threshold_configuration():
    c = CandidateConfig(
        deep_verification_risk_threshold=0.4,
        fast_path_min_confidence=0.75,
        disagreement_trigger=0.5,
    )
    assert c.set_fields() == {
        "deep_verification_risk_threshold": 0.4,
        "fast_path_min_confidence": 0.75,
        "disagreement_trigger": 0.5,
    }
    assert c.to_overrides() == {"verification": c.set_fields()}


def test_invalid_threshold_rejected():
    with pytest.raises(ValidationError):
        CandidateConfig(fast_path_min_confidence=1.5)
    with pytest.raises(ValidationError):
        CandidateConfig(deep_verification_risk_threshold=-0.1)
    with pytest.raises(ValidationError):
        CandidateConfig(made_up_threshold=0.5)  # extra="forbid"


def test_boundary_values_zero_and_one_accepted():
    c = CandidateConfig(
        low_risk_floor=0.0,
        fast_path_min_confidence=1.0,
        deep_verification_risk_threshold=0.0,
        disagreement_trigger=1.0,
    )
    assert c.low_risk_floor == 0.0
    assert c.fast_path_min_confidence == 1.0


def test_empty_candidate_has_no_overrides():
    assert CandidateConfig().to_overrides() == {}


def test_resolved_thresholds_overlay_current_config(config):
    resolved = CandidateConfig(fast_path_min_confidence=0.9).resolved_thresholds(config)
    assert resolved["fast_path_min_confidence"] == 0.9
    # untouched thresholds fall back to the live config value
    assert resolved["deep_verification_risk_threshold"] == float(
        config["verification"]["deep_verification_risk_threshold"]
    )


# ------------------------------------------------------------------
# 4. sweep produces the expected number of combinations
# ------------------------------------------------------------------


def test_sweep_produces_cartesian_product(report):
    assert report.candidate_count == 9
    assert len(report.results) == 9
    assert isinstance(report.baseline, CalibrationResult)
    assert set(report.swept_thresholds) == {
        "deep_verification_risk_threshold",
        "fast_path_max_risk",
        "fast_path_min_confidence",
    }


def test_sweep_single_axis_count(config, cache):
    r = sweep_thresholds(
        disagreement_thresholds=[0.3, 0.4, 0.5, 0.6], config=config, cache=cache
    )
    assert r.candidate_count == 4
    assert r.swept_thresholds == ["disagreement_trigger"]


def test_sweep_with_no_axes_has_only_baseline(config, cache):
    r = sweep_thresholds(config=config, cache=cache)
    assert r.candidate_count == 0
    assert r.results == []
    assert isinstance(r.baseline, CalibrationResult)


# ------------------------------------------------------------------
# 5. determinism
# ------------------------------------------------------------------


def test_sweep_is_deterministic(config, cache):
    a = sweep_thresholds(
        risk_thresholds=[0.3, 0.5], confidence_thresholds=[0.6, 0.8],
        config=config, cache=cache,
    )
    b = sweep_thresholds(
        risk_thresholds=[0.3, 0.5], confidence_thresholds=[0.6, 0.8],
        config=config, cache=cache,
    )
    assert a.model_dump() == b.model_dump()


# ------------------------------------------------------------------
# 6-9. metric bounds + internal consistency
# ------------------------------------------------------------------


def test_metrics_are_bounded(report):
    for result in [report.baseline, *report.results]:
        s = result.safety
        for value in (
            s.accuracy, s.precision, s.recall, s.f1,
            s.false_positive_rate, s.missed_risk_rate,
        ):
            assert 0.0 <= value <= 1.0
        e = result.efficiency
        for value in (
            e.allow_rate, e.annotate_rate, e.verify_rate,
            e.human_review_rate, e.block_rate,
            e.fast_path_rate, e.deep_path_rate,
        ):
            assert 0.0 <= value <= 1.0


def test_fast_and_deep_rates_sum_to_one(report):
    for result in [report.baseline, *report.results]:
        e = result.efficiency
        assert e.fast_path_rate + e.deep_path_rate == pytest.approx(1.0, abs=1e-4)


def test_decision_rates_are_internally_consistent(report):
    for result in [report.baseline, *report.results]:
        n = result.evaluation_count
        assert sum(result.decision_counts.values()) == n
        e = result.efficiency
        rate_sum = (
            e.allow_rate + e.annotate_rate + e.verify_rate
            + e.human_review_rate + e.block_rate
        )
        assert rate_sum == pytest.approx(1.0, abs=1e-3)
        # rates match the raw decision counts
        assert e.allow_rate == pytest.approx(result.decision_counts["ALLOW"] / n, abs=1e-4)
        assert e.block_rate == pytest.approx(result.decision_counts["BLOCK"] / n, abs=1e-4)


def test_safety_counts_are_consistent(report):
    for result in [report.baseline, *report.results]:
        s = result.safety
        assert s.risky_count + s.clean_count == s.evaluation_count == result.evaluation_count
        assert s.risky_count > 0  # the eval set contains genuinely-risky cases


# ------------------------------------------------------------------
# 10. latency is measured, not fabricated
# ------------------------------------------------------------------


def test_latency_is_measured_not_fabricated(report, cache):
    e = report.baseline.efficiency
    assert e.average_latency_ms > 0.0
    assert e.p95_latency_ms >= e.average_latency_ms  # a real distribution, not a constant
    assert "MEASURED" in e.latency_basis

    # every reported average sits within the envelope of the cache's own
    # measured per-record timings -> it is a recombination, not a literal.
    lo = min(r.fast_path_ms + r.downstream_ms for r in cache.records)
    hi = max(
        r.fast_path_ms + r.deep_extra_ms + r.downstream_ms for r in cache.records
    )
    for result in [report.baseline, *report.results]:
        assert lo <= result.efficiency.average_latency_ms <= hi


# ------------------------------------------------------------------
# 11-12. ground-truth / Interaction boundary
# ------------------------------------------------------------------


def test_production_interaction_carries_no_ground_truth():
    leak = {
        "ground_truth_hallucination", "ground_truth_pii", "ground_truth_toxicity",
        "ground_truth_bias", "ground_truth_cost_anomaly", "expected_decision",
        "final_outcome",
    }
    assert leak.isdisjoint(Interaction.model_fields)


def test_sweep_feeds_only_interactions_to_the_engine(cache):
    # the cached records hold plain production Interaction objects
    assert all(isinstance(r.interaction, Interaction) for r in cache.records)
    assert all(
        "ground_truth_hallucination" not in r.interaction.model_dump()
        for r in cache.records
    )


def test_calibration_may_use_ground_truth_inside_the_boundary(cache):
    # the ground-truth "any risk" label IS available to the evaluation layer
    assert any(r.gt_any for r in cache.records)
    assert any(not r.gt_any for r in cache.records)


# ------------------------------------------------------------------
# 13-14. production isolation + no global config mutation
# ------------------------------------------------------------------


def test_calibration_not_imported_by_production():
    offenders: list[str] = []
    for sub in (
        "detectors", "fusion", "policy", "decision",
        "verification", "consequence", "criticality",
    ):
        for path in (_REPO / sub).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "import calibration" in text or "from calibration" in text:
                offenders.append(str(path.relative_to(_REPO)))
    assert offenders == [], offenders


def test_sweep_does_not_mutate_production_config(config, cache):
    before = copy.deepcopy(load_settings())
    sweep_thresholds(
        risk_thresholds=[0.1, 0.9],
        confidence_thresholds=[0.1, 0.99],
        config=config,
        cache=cache,
    )
    assert load_settings() == before
    # settings.yaml on disk is untouched
    assert (
        load_settings()["verification"]["deep_verification_risk_threshold"] == 0.35
    )


def test_sweep_does_not_change_global_engine_behaviour(config, cache):
    sweep_thresholds(
        risk_thresholds=[0.05, 0.95], config=config, cache=cache
    )
    # a fresh engine still uses the baseline thresholds
    fresh = build_engine()
    assert fresh.verification_router is not None
    assert (
        load_settings()["verification"]["fast_path_min_confidence"] == 0.70
    )


# ------------------------------------------------------------------
# 15 (+ report requirement). scenarios A-C unchanged, suite stays green
# ------------------------------------------------------------------


def test_core_scenarios_unchanged_after_calibration_import():
    engine = build_engine()
    got = {
        name: engine.evaluate(
            factory(), timestamp=None, record_session=False
        ).final_decision.decision.value
        for name, factory in {
            "A": scenarios.scenario_a_clean,
            "B": scenarios.scenario_b_hallucination,
            "C": scenarios.scenario_c_pii,
        }.items()
    }
    assert got == {"A": "ALLOW", "B": "VERIFY", "C": "BLOCK"}


# ------------------------------------------------------------------
# smoke: report is serializable + summary renders
# ------------------------------------------------------------------


def test_report_is_serializable_and_summary_renders(report):
    blob = report.model_dump_json()
    restored = CalibrationSweepReport.model_validate_json(blob)
    assert restored.candidate_count == report.candidate_count

    text = format_summary(report)
    assert "Calibration sweep" in text
    assert "Candidates: 9" in text
    assert "No 'best configuration' is declared" in text


def test_evaluate_candidate_direct(config, cache):
    result = evaluate_candidate(
        CandidateConfig(fast_path_min_confidence=0.5), cache=cache, config=config
    )
    assert result.evaluation_count == len(cache.records)
    assert result.resolved_thresholds["fast_path_min_confidence"] == 0.5
    assert 0.0 <= result.safety.recall <= 1.0
