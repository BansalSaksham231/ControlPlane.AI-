"""Evaluation framework tests — metrics, confusion matrix, report, ablation, no leakage."""

from __future__ import annotations

import pytest

from evaluation.ablation import run_ablation
from evaluation.evaluation import EvaluationReport, evaluate, load_evaluation_dataset
from evaluation.metrics import (
    binary_metrics,
    confusion_matrix,
    distribution,
    threshold_sweep,
)


# ---------------------------------------------------------------- metrics

def test_binary_metrics_basic():
    m = binary_metrics([True, True, False, False], [True, False, False, False])
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert m["f1"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["false_positive_rate"] == 0.0


def test_binary_metrics_length_mismatch():
    with pytest.raises(ValueError):
        binary_metrics([True], [True, False])


def test_confusion_matrix():
    cm = confusion_matrix(
        ["ALLOW", "BLOCK", "ALLOW"], ["ALLOW", "ALLOW", "ALLOW"], labels=["ALLOW", "BLOCK"]
    )
    assert cm["labels"] == ["ALLOW", "BLOCK"]
    assert cm["matrix"][0][0] == 2      # ALLOW->ALLOW
    assert cm["matrix"][1][0] == 1      # BLOCK->ALLOW
    assert cm["exact_match"] == 2


def test_threshold_sweep_monotonic_support():
    truth = [True, False, True, False, True]
    scores = [0.9, 0.1, 0.6, 0.3, 0.8]
    sweep = threshold_sweep(truth, scores, [0.2, 0.5, 0.7])
    assert [row["threshold"] for row in sweep] == [0.2, 0.5, 0.7]
    # higher threshold -> fewer predicted positives
    positives = [row["tp"] + row["fp"] for row in sweep]
    assert positives == sorted(positives, reverse=True)


def test_distribution():
    assert distribution(["a", "a", "b"]) == {"a": 2, "b": 1}


# ---------------------------------------------------------------- dataset

def test_dataset_is_production_shaped():
    dataset = load_evaluation_dataset()
    assert len(dataset) == 150
    interaction, ground_truth = dataset[0]
    from data.schemas import Interaction

    assert isinstance(interaction, Interaction)
    # ground truth is separate, never on the interaction
    for key in ("ground_truth_hallucination", "expected_decision", "final_outcome"):
        assert key in ground_truth
        assert not hasattr(interaction, key)


def test_dataset_deterministic():
    a = load_evaluation_dataset()
    b = load_evaluation_dataset()
    assert [i.model_dump(mode="json") for i, _ in a] == [
        i.model_dump(mode="json") for i, _ in b
    ]


# ---------------------------------------------------------------- report

@pytest.fixture(scope="module")
def report() -> EvaluationReport:
    return evaluate()


def test_report_shape(report):
    assert report.n_cases == 150
    for block in (report.pii, report.toxicity, report.bias, report.cost_anomaly):
        assert set(block) >= {"precision", "recall", "f1"}
    assert 0.0 <= report.abstention_rate <= 1.0
    assert report.coverage_rate + report.abstention_rate <= 1.0 + 1e-9


def test_report_responsibility_detectors_reasonable(report):
    # These heuristics separate the synthetic set cleanly.
    assert report.pii["recall"] >= 0.8
    assert report.toxicity["recall"] >= 0.8
    assert report.cost_anomaly["recall"] >= 0.8
    assert report.pii["false_positive_rate"] <= 0.1


def test_report_performance_no_false_contradictions(report):
    # UNVERIFIED must never be scored as a hallucination hit.
    assert report.performance_hallucination["false_positive_rate"] <= 0.05


def test_report_has_intervention_distribution(report):
    assert sum(report.intervention_distribution.values()) == 150
    assert 0.0 <= report.human_review_rate <= 1.0


def test_report_confusion_matrix_vs_baseline(report):
    cm = report.decision_confusion_vs_baseline
    assert cm["total"] == 150
    assert len(cm["labels"]) == 5


def test_report_is_deterministic():
    a = evaluate().model_dump()
    b = evaluate().model_dump()
    a.pop("mean_latency_ms")
    b.pop("mean_latency_ms")
    assert a == b


# ---------------------------------------------------------------- ablation

def test_ablation_runs():
    report = run_ablation()
    names = {m.mode for m in report.modes}
    assert {"performance_only", "responsibility_only", "cost_only", "fused_only", "full_pipeline"} <= names
    full = next(m for m in report.modes if m.mode == "full_pipeline")
    assert full.tier_distribution is not None
    assert sum(full.tier_distribution.values()) == report.n_cases


def test_ablation_full_pipeline_beats_best_flat_score():
    """
    After the Round-2 evidence-weighted upgrade the flat performance score
    is itself better calibrated, so the interesting comparison is *catch
    quality*: the layered pipeline catches materially more genuinely-risky
    cases than any single flat score, while still not escalating clean
    traffic to a human.
    """
    report = run_ablation()
    full = next(m for m in report.modes if m.mode == "full_pipeline")
    flat_modes = [m for m in report.modes if m.threshold is not None]
    best_flat = max(flat_modes, key=lambda m: m.caught_risky["f1"])

    assert full.caught_risky["recall"] > best_flat.caught_risky["recall"]
    assert full.caught_risky["f1"] >= best_flat.caught_risky["f1"]
    assert full.clean_human_escalation_rate <= 0.05
    # proportionate: more than one tier is actually used
    assert sum(1 for v in full.tier_distribution.values() if v > 0) >= 3


def test_ablation_conclusion_text():
    report = run_ablation()
    assert "pipeline" in report.conclusion.lower()
    assert len(report.conclusion) > 80
