"""
Phase 7 — Step 2: safety-constrained configuration selection.

Safety is a hard gate applied BEFORE the efficiency objective. The
selected candidate is a recommendation for evaluation only — never an
applied production change.
"""

from __future__ import annotations

import copy
import pathlib

import pytest
from pydantic import ValidationError

from calibration.advisor import CalibrationCache
from calibration.select import (
    ConfigurationSelection,
    EfficiencyObjective,
    SafetyConstraints,
    format_selection,
    select_configuration,
)
from calibration.sweep import (
    CalibrationResult,
    CalibrationSweepReport,
    CandidateConfig,
    EfficiencyMetrics,
    SafetyMetrics,
    sweep_thresholds,
)
from evaluation.evaluation import build_engine
from settings import load_settings
from tests import scenarios

_REPO = pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# synthetic CalibrationResult builder (for precise constraint tests)
# ------------------------------------------------------------------


def _result(
    *,
    recall: float,
    precision: float,
    fpr: float = 0.10,
    missed: float = 0.02,
    human_review: float = 0.10,
    fast: float = 0.50,
    latency: float = 5.0,
    tag: float = 0.5,
) -> CalibrationResult:
    cfg = CandidateConfig(fast_path_min_confidence=tag)
    return CalibrationResult(
        configuration=cfg,
        resolved_thresholds={"fast_path_min_confidence": tag},
        evaluation_count=150,
        decision_counts={"ALLOW": 100, "ANNOTATE": 0, "VERIFY": 30, "HUMAN_REVIEW": 15, "BLOCK": 5},
        safety=SafetyMetrics(
            evaluation_count=150,
            risky_count=60,
            clean_count=90,
            accuracy=0.8,
            precision=precision,
            recall=recall,
            f1=0.8,
            false_positive_rate=fpr,
            missed_risk_rate=missed,
        ),
        efficiency=EfficiencyMetrics(
            allow_rate=1 - fast if fast < 1 else 0.0,
            annotate_rate=0.0,
            verify_rate=max(0.0, fast - human_review),
            human_review_rate=human_review,
            block_rate=0.0,
            fast_path_rate=fast,
            deep_path_rate=round(1 - fast, 4),
            average_latency_ms=latency,
            p95_latency_ms=latency + 3,
        ),
    )


def _report(baseline: CalibrationResult, results: list[CalibrationResult]) -> CalibrationSweepReport:
    return CalibrationSweepReport(
        evaluation_count=150,
        candidate_count=len(results),
        swept_thresholds=["fast_path_min_confidence"],
        baseline=baseline,
        results=results,
    )


_BASE = _result(recall=0.98, precision=0.80, human_review=0.23, fast=0.15, latency=2.8, tag=0.0)


# ------------------------------------------------------------------
# constraint object validation
# ------------------------------------------------------------------


def test_constraints_validate_range():
    SafetyConstraints(minimum_recall=0.0, minimum_precision=1.0)
    with pytest.raises(ValidationError):
        SafetyConstraints(minimum_recall=1.5, minimum_precision=0.5)
    with pytest.raises(ValidationError):
        SafetyConstraints(minimum_recall=0.9, minimum_precision=0.5, maximum_false_positive_rate=-0.1)
    with pytest.raises(ValidationError):
        SafetyConstraints(minimum_recall=0.9, minimum_precision=0.5, made_up=0.3)


# ------------------------------------------------------------------
# 1-5. safety gate
# ------------------------------------------------------------------


def test_candidate_satisfying_constraints_is_eligible():
    good = _result(recall=0.97, precision=0.75, tag=0.6)
    sel = select_configuration(
        _report(_BASE, [good]),
        SafetyConstraints(minimum_recall=0.95, minimum_precision=0.70),
    )
    assert sel.eligible_candidate_count == 1
    assert sel.candidate_evaluations[0].eligible is True
    assert sel.candidate_evaluations[0].violations == []


def test_candidate_failing_minimum_recall_is_rejected():
    bad = _result(recall=0.80, precision=0.90, tag=0.6)
    sel = select_configuration(
        _report(_BASE, [bad]),
        SafetyConstraints(minimum_recall=0.95, minimum_precision=0.50),
    )
    assert sel.eligible_candidate_count == 0
    assert any("recall" in v for v in sel.candidate_evaluations[0].violations)
    assert sel.status == "NO_ELIGIBLE_CANDIDATE"


def test_candidate_failing_minimum_precision_is_rejected():
    bad = _result(recall=0.99, precision=0.40, tag=0.6)
    sel = select_configuration(
        _report(_BASE, [bad]),
        SafetyConstraints(minimum_recall=0.90, minimum_precision=0.70),
    )
    assert sel.eligible_candidate_count == 0
    assert any("precision" in v for v in sel.candidate_evaluations[0].violations)


def test_candidate_failing_maximum_fpr_is_rejected():
    bad = _result(recall=0.99, precision=0.80, fpr=0.65, tag=0.6)
    sel = select_configuration(
        _report(_BASE, [bad]),
        SafetyConstraints(
            minimum_recall=0.90, minimum_precision=0.50, maximum_false_positive_rate=0.55
        ),
    )
    assert sel.eligible_candidate_count == 0
    assert any("false_positive_rate" in v for v in sel.candidate_evaluations[0].violations)


def test_candidate_failing_maximum_missed_risk_is_rejected():
    bad = _result(recall=0.85, precision=0.80, missed=0.15, tag=0.6)
    sel = select_configuration(
        _report(_BASE, [bad]),
        SafetyConstraints(
            minimum_recall=0.0, minimum_precision=0.0, maximum_missed_risk_rate=0.10
        ),
    )
    assert sel.eligible_candidate_count == 0
    assert any("missed_risk_rate" in v for v in sel.candidate_evaluations[0].violations)


def test_candidate_failing_maximum_human_review_is_rejected():
    bad = _result(recall=0.99, precision=0.85, human_review=0.40, tag=0.6)
    sel = select_configuration(
        _report(_BASE, [bad]),
        SafetyConstraints(
            minimum_recall=0.90, minimum_precision=0.50, maximum_human_review_rate=0.30
        ),
    )
    assert sel.eligible_candidate_count == 0
    assert any("human_review_rate" in v for v in sel.candidate_evaluations[0].violations)


# ------------------------------------------------------------------
# 6-8. efficiency objective selects the right candidate
# ------------------------------------------------------------------


@pytest.fixture
def three_eligible():
    # all three pass a lenient safety gate; they differ on efficiency
    a = _result(recall=0.96, precision=0.72, human_review=0.30, fast=0.20, latency=8.0, tag=0.3)
    b = _result(recall=0.96, precision=0.72, human_review=0.10, fast=0.55, latency=5.0, tag=0.5)
    c = _result(recall=0.96, precision=0.72, human_review=0.20, fast=0.80, latency=3.0, tag=0.8)
    return _report(_BASE, [a, b, c])


def test_min_human_review_selects_lowest_human_review(three_eligible):
    sel = select_configuration(
        three_eligible,
        SafetyConstraints(minimum_recall=0.95, minimum_precision=0.70),
        EfficiencyObjective.MIN_HUMAN_REVIEW,
    )
    assert sel.status == "SELECTED"
    assert sel.selected_result.efficiency.human_review_rate == 0.10
    assert sel.selected_configuration.fast_path_min_confidence == 0.5


def test_max_fast_path_selects_highest_fast_path(three_eligible):
    sel = select_configuration(
        three_eligible,
        SafetyConstraints(minimum_recall=0.95, minimum_precision=0.70),
        EfficiencyObjective.MAX_FAST_PATH,
    )
    assert sel.selected_result.efficiency.fast_path_rate == 0.80


def test_min_latency_selects_lowest_latency(three_eligible):
    sel = select_configuration(
        three_eligible,
        SafetyConstraints(minimum_recall=0.95, minimum_precision=0.70),
        EfficiencyObjective.MIN_LATENCY,
    )
    assert sel.selected_result.efficiency.average_latency_ms == 3.0


# ------------------------------------------------------------------
# 9. safety BEFORE efficiency
# ------------------------------------------------------------------


def test_safety_is_applied_before_efficiency():
    # the most efficient candidate is unsafe; it must NOT be selected
    unsafe_but_fast = _result(
        recall=0.50, precision=0.30, human_review=0.01, fast=0.99, latency=1.0, tag=0.9
    )
    safe_but_slow = _result(
        recall=0.97, precision=0.80, human_review=0.25, fast=0.10, latency=9.0, tag=0.2
    )
    sel = select_configuration(
        _report(_BASE, [unsafe_but_fast, safe_but_slow]),
        SafetyConstraints(minimum_recall=0.95, minimum_precision=0.70),
        EfficiencyObjective.MAX_FAST_PATH,
    )
    assert sel.eligible_candidate_count == 1
    assert sel.selected_configuration.fast_path_min_confidence == 0.2
    assert sel.selected_result.efficiency.fast_path_rate == 0.10  # the SAFE one


# ------------------------------------------------------------------
# 10. no eligible candidates -> structured no-selection
# ------------------------------------------------------------------


def test_no_eligible_candidates_returns_structured_result():
    bad = _result(recall=0.60, precision=0.40, tag=0.6)
    sel = select_configuration(
        _report(_BASE, [bad]),
        SafetyConstraints(minimum_recall=0.95, minimum_precision=0.90),
    )
    assert sel.status == "NO_ELIGIBLE_CANDIDATE"
    assert sel.selected_configuration is None
    assert sel.selected_result is None
    assert "NOT relaxed" in sel.selection_reason
    assert sel.baseline_result is not None  # baseline still reported


def test_empty_candidate_list_is_handled():
    sel = select_configuration(
        _report(_BASE, []),
        SafetyConstraints(minimum_recall=0.9, minimum_precision=0.5),
    )
    assert sel.status == "NO_ELIGIBLE_CANDIDATE"
    assert sel.total_candidate_count == 0
    assert "NOT relaxed" in sel.selection_reason


# ------------------------------------------------------------------
# 11-12. determinism + tie-breaking
# ------------------------------------------------------------------


def test_selection_is_deterministic(three_eligible):
    c = SafetyConstraints(minimum_recall=0.95, minimum_precision=0.70)
    a = select_configuration(three_eligible, c, EfficiencyObjective.MIN_LATENCY)
    b = select_configuration(three_eligible, c, EfficiencyObjective.MIN_LATENCY)
    assert a.model_dump() == b.model_dump()


def test_ties_are_broken_deterministically():
    # two candidates identical on the objective and every documented tie-break
    # except original order -> the earlier one wins, every time.
    x = _result(recall=0.96, precision=0.75, human_review=0.10, fast=0.5, latency=5.0, tag=0.31)
    y = _result(recall=0.96, precision=0.75, human_review=0.10, fast=0.5, latency=5.0, tag=0.32)
    rep = _report(_BASE, [x, y])
    c = SafetyConstraints(minimum_recall=0.95, minimum_precision=0.70)
    for _ in range(5):
        sel = select_configuration(rep, c, EfficiencyObjective.MIN_HUMAN_REVIEW)
        assert sel.selected_configuration.fast_path_min_confidence == 0.31

    # a higher-recall tie candidate wins over the objective-equal one
    z = _result(recall=0.99, precision=0.75, human_review=0.10, fast=0.5, latency=5.0, tag=0.99)
    sel = select_configuration(
        _report(_BASE, [x, z]), c, EfficiencyObjective.MIN_HUMAN_REVIEW
    )
    assert sel.selected_configuration.fast_path_min_confidence == 0.99


# ------------------------------------------------------------------
# 13-15. governance / isolation
# ------------------------------------------------------------------


def test_selection_does_not_write_settings(three_eligible):
    before = copy.deepcopy(load_settings())
    select_configuration(
        three_eligible,
        SafetyConstraints(minimum_recall=0.9, minimum_precision=0.5),
        EfficiencyObjective.MIN_HUMAN_REVIEW,
    )
    assert load_settings() == before
    assert load_settings()["verification"]["deep_verification_risk_threshold"] == 0.35


def test_result_distinguishes_baseline_from_selected(three_eligible):
    sel = select_configuration(
        three_eligible,
        SafetyConstraints(minimum_recall=0.9, minimum_precision=0.5),
        EfficiencyObjective.MIN_HUMAN_REVIEW,
    )
    assert "Recommended for evaluation" in sel.disposition
    assert "NOT applied" in sel.disposition
    assert "Applied" not in sel.selection_reason
    assert sel.baseline_result is not sel.selected_result
    assert "synthetic evaluation set" in sel.selection_reason


def test_production_modules_do_not_import_calibration():
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


# ------------------------------------------------------------------
# real 150-case smoke test + scenario guard
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_sweep():
    cfg = load_settings()
    cache = CalibrationCache(cfg)
    return sweep_thresholds(
        risk_thresholds=[0.35, 0.60, 0.85],
        confidence_thresholds=[0.30, 0.60, 0.90],
        config=cfg,
        cache=cache,
    )


def test_real_smoke_selection(real_sweep):
    sel = select_configuration(
        real_sweep,
        SafetyConstraints(minimum_recall=0.95, minimum_precision=0.0),
        EfficiencyObjective.MIN_LATENCY,
    )
    assert sel.total_candidate_count == 9
    assert 0 <= sel.eligible_candidate_count <= 9
    text = format_selection(sel)
    assert "BASELINE" in text and "Reason:" in text
    if sel.status == "SELECTED":
        # every documented safety gate holds for the winner
        assert sel.selected_result.safety.recall >= 0.95
        # and it is at least as fast as the baseline on the objective, or the
        # reason explicitly says nothing beat the baseline
        assert (
            sel.selected_result.efficiency.average_latency_ms
            <= real_sweep.baseline.efficiency.average_latency_ms + 1e-9
            or "No tested candidate improves" in sel.selection_reason
        )


def test_core_scenarios_unchanged():
    engine = build_engine()
    got = {
        k: engine.evaluate(f(), timestamp=None, record_session=False).final_decision.decision.value
        for k, f in {
            "A": scenarios.scenario_a_clean,
            "B": scenarios.scenario_b_hallucination,
            "C": scenarios.scenario_c_pii,
        }.items()
    }
    assert got == {"A": "ALLOW", "B": "VERIFY", "C": "BLOCK"}


def test_selection_report_round_trips(three_eligible):
    sel = select_configuration(
        three_eligible,
        SafetyConstraints(minimum_recall=0.9, minimum_precision=0.5),
        EfficiencyObjective.MAX_FAST_PATH,
    )
    restored = ConfigurationSelection.model_validate_json(sel.model_dump_json())
    assert restored.selected_configuration == sel.selected_configuration
