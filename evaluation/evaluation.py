"""
Evaluation framework.

Runs the ControlPlane pipeline over the synthetic ``EvaluationCase``
dataset and scores it against ground truth. Ground truth is read ONLY
here — never inside a detector. The pipeline receives production-shaped
``Interaction`` objects built strictly from ``Interaction.model_fields``.

Important evaluation note: ``expected_decision`` in the dataset is itself a
simple heuristic baseline produced by the generator, not an oracle. The
decision-confusion numbers therefore compare two policies, and are
reported as such.
"""

from __future__ import annotations

import random
from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from data.generator import (
    EVAL_EXTRA_COLUMNS,
    generate_evaluation_cases,
    generate_interactions,
)
from data.schemas import Interaction, InterventionTier
from decision.engine import DecisionEngine
from detectors.cost.baseline import CostBaseline
from detectors.cost.detector import CostDetector
from evaluation.metrics import (
    binary_metrics,
    confusion_matrix,
    distribution,
    rate,
    threshold_sweep,
)
from settings import load_settings

_EVAL_TS = datetime(2026, 8, 21, 12, 0, 0)
_TIER_VALUES = [t.value for t in InterventionTier]


class EvaluationReport(BaseModel):
    n_cases: int
    risk_binary_threshold: float

    performance_hallucination: dict[str, float]   # = contradiction detection metrics
    contradiction_precision: float
    contradiction_recall: float
    pii: dict[str, float]
    toxicity: dict[str, float]
    bias: dict[str, float]
    responsibility_overall: dict[str, float]
    cost_anomaly: dict[str, float]
    any_risk_catch: dict[str, float]

    performance_status_distribution: dict[str, int]
    abstention_rate: float
    coverage_rate: float
    verification_coverage: float

    intervention_distribution: dict[str, int]
    allow_rate: float
    verify_rate: float
    human_review_rate: float
    block_rate: float

    # --- Round 2 upgrade: risk vs confidence calibration ---
    risk_confidence_buckets: dict[str, int]
    high_risk_low_confidence_rate: float
    high_risk_high_confidence_rate: float

    criticality_band_distribution: dict[str, int]
    reason_code_frequency: dict[str, int]

    decision_confusion_vs_baseline: dict[str, Any]

    threshold_sweeps: dict[str, list[dict[str, float]]]

    mean_latency_ms: float
    notes: list[str]


def load_evaluation_dataset(
    config: dict[str, Any] | None = None,
) -> list[tuple[Interaction, dict[str, Any]]]:
    """Deterministically rebuild the evaluation dataset (mirrors generator.run sequencing)."""
    cfg = config if config is not None else load_settings()
    rng = random.Random(cfg["seed"])
    generate_interactions(cfg, rng)  # advance rng exactly as run() does
    rows = generate_evaluation_cases(cfg, rng)

    dataset: list[tuple[Interaction, dict[str, Any]]] = []
    for row in rows:
        interaction = Interaction.model_validate(
            {key: row[key] for key in Interaction.model_fields}
        )
        ground_truth = {key: row[key] for key in EVAL_EXTRA_COLUMNS}
        dataset.append((interaction, ground_truth))
    return dataset


def build_engine(
    config: dict[str, Any] | None = None,
    *,
    session_manager: Any | None = None,
    confidence_aware: bool = True,
) -> DecisionEngine:
    """A decision engine with the cost baseline fitted from synthetic traffic."""
    cfg = config if config is not None else load_settings()
    rng = random.Random(cfg["seed"])
    interactions = generate_interactions(cfg, rng)
    baseline = CostBaseline.fit(
        interactions, cfg, estimate_cost=CostDetector(cfg).estimate_cost
    )
    return DecisionEngine(
        cfg,
        cost_baseline=baseline,
        session_manager=session_manager,
        confidence_aware=confidence_aware,
    )


def evaluate(
    config: dict[str, Any] | None = None,
    *,
    limit: int | None = None,
) -> EvaluationReport:
    cfg = config if config is not None else load_settings()
    threshold = float(cfg["evaluation"]["risk_binary_threshold"])
    dataset = load_evaluation_dataset(cfg)
    if limit is not None:
        dataset = dataset[:limit]
    engine = build_engine(cfg)

    perf_risk: list[float] = []
    pii_risk: list[float] = []
    tox_risk: list[float] = []
    bias_risk: list[float] = []
    resp_risk: list[float] = []
    cost_risk: list[float] = []

    gt_hallucination: list[bool] = []
    gt_pii: list[bool] = []
    gt_tox: list[bool] = []
    gt_bias: list[bool] = []
    gt_any: list[bool] = []
    gt_any_responsibility: list[bool] = []
    gt_cost: list[bool] = []

    perf_status: list[str] = []
    perf_contradicted: list[bool] = []
    predicted_tiers: list[str] = []
    baseline_tiers: list[str] = []
    intervened_when_risky: list[bool] = []
    latencies: list[float] = []

    overall_risks: list[float] = []
    decision_confidences: list[float] = []
    criticality_bands: list[str] = []
    reason_code_counter: Counter[str] = Counter()

    for interaction, gt in dataset:
        trace = engine.evaluate(interaction, timestamp=_EVAL_TS, record_session=False)
        fd = trace.final_decision
        overall_risks.append(fd.overall_risk)
        decision_confidences.append(fd.decision_confidence)
        criticality_bands.append(trace.criticality.band)
        reason_code_counter.update(fd.reason_codes)

        perf_risk.append(trace.performance.performance_risk)
        pii_risk.append(trace.responsibility.pii_risk)
        tox_risk.append(trace.responsibility.toxicity_risk)
        bias_risk.append(trace.responsibility.bias_risk)
        resp_risk.append(trace.responsibility.overall_responsibility_risk)
        cost_risk.append(trace.cost.cost_risk)
        perf_status.append(trace.performance.status.value)
        perf_contradicted.append(trace.performance.status.value == "CONTRADICTED")

        any_flag = bool(
            gt["ground_truth_hallucination"]
            or gt["ground_truth_pii"]
            or gt["ground_truth_toxicity"]
            or gt["ground_truth_bias"]
            or gt["ground_truth_cost_anomaly"]
        )
        gt_hallucination.append(bool(gt["ground_truth_hallucination"]))
        gt_pii.append(bool(gt["ground_truth_pii"]))
        gt_tox.append(bool(gt["ground_truth_toxicity"]))
        gt_bias.append(bool(gt["ground_truth_bias"]))
        gt_cost.append(bool(gt["ground_truth_cost_anomaly"]))
        gt_any.append(any_flag)
        gt_any_responsibility.append(
            bool(
                gt["ground_truth_pii"]
                or gt["ground_truth_toxicity"]
                or gt["ground_truth_bias"]
            )
        )

        predicted_tiers.append(fd.decision.value)
        baseline_tiers.append(str(gt["expected_decision"]))
        if any_flag:
            intervened_when_risky.append(fd.decision != InterventionTier.ALLOW)
        latencies.append(trace.latency_ms)

    n = len(dataset)

    def _bin(scores: list[float], truth: list[bool]) -> dict[str, float]:
        return binary_metrics(truth, [s >= threshold for s in scores])

    unverified = sum(1 for s in perf_status if s == "UNVERIFIED")
    covered = sum(1 for s in perf_status if s in ("SUPPORTED", "CONTRADICTED"))

    intervention_counts = distribution(predicted_tiers)
    human_review = sum(
        intervention_counts.get(t, 0)
        for t in (InterventionTier.HUMAN_REVIEW.value, InterventionTier.BLOCK.value)
    )
    blocks = intervention_counts.get(InterventionTier.BLOCK.value, 0)
    allows = intervention_counts.get(InterventionTier.ALLOW.value, 0)
    verifies = intervention_counts.get(InterventionTier.VERIFY.value, 0)

    contradiction_metrics = binary_metrics(gt_hallucination, perf_contradicted)

    # risk-vs-confidence calibration buckets
    r_thr = float(cfg["evaluation"]["risk_bucket_threshold"])
    c_thr = float(cfg["evaluation"]["confidence_bucket_threshold"])
    buckets = {
        "high_risk_high_confidence": 0,
        "high_risk_low_confidence": 0,
        "low_risk_high_confidence": 0,
        "low_risk_low_confidence": 0,
    }
    for risk_v, conf_v in zip(overall_risks, decision_confidences):
        hi_r = risk_v >= r_thr
        hi_c = conf_v >= c_thr
        key = f"{'high' if hi_r else 'low'}_risk_{'high' if hi_c else 'low'}_confidence"
        buckets[key] += 1
    high_risk = buckets["high_risk_high_confidence"] + buckets["high_risk_low_confidence"]

    return EvaluationReport(
        n_cases=n,
        risk_binary_threshold=threshold,
        performance_hallucination=contradiction_metrics,
        contradiction_precision=contradiction_metrics["precision"],
        contradiction_recall=contradiction_metrics["recall"],
        pii=_bin(pii_risk, gt_pii),
        toxicity=_bin(tox_risk, gt_tox),
        bias=_bin(bias_risk, gt_bias),
        responsibility_overall=_bin(resp_risk, gt_any_responsibility),
        cost_anomaly=_bin(cost_risk, gt_cost),
        any_risk_catch=binary_metrics(
            gt_any, [t != InterventionTier.ALLOW.value for t in predicted_tiers]
        ),
        performance_status_distribution=distribution(perf_status),
        abstention_rate=rate(unverified, n),
        coverage_rate=rate(covered, n),
        verification_coverage=rate(covered, n),
        intervention_distribution=intervention_counts,
        allow_rate=rate(allows, n),
        verify_rate=rate(verifies, n),
        human_review_rate=rate(human_review, n),
        block_rate=rate(blocks, n),
        risk_confidence_buckets=buckets,
        high_risk_low_confidence_rate=rate(buckets["high_risk_low_confidence"], high_risk or 1),
        high_risk_high_confidence_rate=rate(buckets["high_risk_high_confidence"], high_risk or 1),
        criticality_band_distribution=distribution(criticality_bands),
        reason_code_frequency=dict(reason_code_counter.most_common()),
        decision_confusion_vs_baseline=confusion_matrix(
            baseline_tiers, predicted_tiers, labels=_TIER_VALUES
        ),
        threshold_sweeps={
            "performance_vs_hallucination": threshold_sweep(gt_hallucination, perf_risk),
            "pii": threshold_sweep(gt_pii, pii_risk),
            "cost": threshold_sweep(gt_cost, cost_risk),
        },
        mean_latency_ms=round(sum(latencies) / n, 3) if n else 0.0,
        notes=[
            "expected_decision is the generator's heuristic baseline, not an oracle; "
            "decision_confusion_vs_baseline compares two policies.",
            "Detectors never read ground-truth fields; only evaluation does.",
            "Claim-level ground truth is not present in the synthetic dataset, so "
            "'performance_hallucination' reports RESPONSE-level contradiction "
            "precision/recall as the closest available proxy.",
            "Heuristic detectors: metrics reflect synthetic data and are not a "
            "production safety guarantee.",
            "PII / toxicity / bias precision-recall is measured against synthetic "
            "cases built from a small set of fixed templates whose surface forms "
            "(email/phone/account-ID shapes, lexicon phrases) are exactly what the "
            "corresponding detector's regex/lexicon was written to match. A "
            "1.00/1.00 score here shows the detector correctly recognises the "
            "formats it targets, not that it generalises to disguised, "
            "paraphrased, or novel-format real-world PII/toxicity/bias — unlike "
            "the hallucination detector's contradiction recall, this number is "
            "not evidence of real-world robustness.",
        ],
    )


if __name__ == "__main__":
    import json

    report = evaluate()
    print(json.dumps(report.model_dump(), indent=2, default=str))
