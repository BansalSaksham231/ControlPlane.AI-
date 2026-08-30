"""
Ablation / architecture-value experiments.

Demonstrates that the layered architecture — multi-dimensional detection
+ fusion + consequence + policy — produces better, more proportionate
oversight than any single flat risk score.

Each "mode" is scored on the same synthetic evaluation set:

    performance_only        intervene if performance_risk    >= t
    responsibility_only     intervene if responsibility_risk >= t
    cost_only               intervene if cost_risk           >= t
    fused_only              intervene if fused overall_risk  >= t
    fused_plus_consequence  intervene if blend(fused, consequence) >= t
    full_pipeline           the real policy engine's tier != ALLOW

"intervene" = anything other than ALLOW. We report how well each mode
catches genuinely risky cases (any ground-truth flag) and how often it
disturbs fully-clean traffic (alert fatigue), plus — for the full
pipeline only — the distribution across the five tiers, which the flat
modes cannot produce at all.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from data.schemas import InterventionTier
from evaluation.evaluation import build_engine, load_evaluation_dataset
from evaluation.metrics import binary_metrics, distribution, rate
from settings import load_settings

_ABLATION_TS = datetime(2026, 8, 21, 12, 0, 0)


class AblationMode(BaseModel):
    mode: str
    threshold: float | None
    intervention_rate: float
    caught_risky: dict[str, float]        # binary metrics vs "any ground-truth risk"
    clean_disturbed_rate: float           # any intervention on fully-clean cases

    # A flat score can only say "flag / don't flag" -> a flag means a human
    # must look. The full pipeline can additionally route to soft tiers
    # (ANNOTATE / VERIFY) that do NOT need a human.
    human_escalation_rate: float
    clean_human_escalation_rate: float
    risky_missed_by_escalation: int

    tier_distribution: dict[str, int] | None = None
    comment: str


class AblationReport(BaseModel):
    n_cases: int
    threshold: float
    modes: list[AblationMode]
    conclusion: str


def run_ablation(
    config: dict[str, Any] | None = None,
    *,
    threshold: float = 0.5,
) -> AblationReport:
    cfg = config if config is not None else load_settings()
    dataset = load_evaluation_dataset(cfg)
    engine = build_engine(cfg)
    # Same pipeline (same fitted cost baseline), but with the
    # confidence-aware policy rules disabled — isolates the value of
    # reasoning about risk + confidence together.
    engine_no_conf = build_engine(cfg, confidence_aware=False)

    perf: list[float] = []
    resp: list[float] = []
    cost: list[float] = []
    fused: list[float] = []
    consequence: list[float] = []
    full_tier: list[str] = []
    no_conf_tier: list[str] = []
    gt_any: list[bool] = []
    gt_clean: list[bool] = []

    for interaction, gt in dataset:
        trace = engine.evaluate(interaction, timestamp=_ABLATION_TS, record_session=False)
        perf.append(trace.performance.performance_risk)
        resp.append(trace.responsibility.overall_responsibility_risk)
        cost.append(trace.cost.cost_risk)
        fused.append(trace.fusion.overall_risk)
        consequence.append(trace.consequence.consequence_score)
        full_tier.append(trace.final_decision.decision.value)
        no_conf_tier.append(
            engine_no_conf.evaluate(
                interaction, timestamp=_ABLATION_TS, record_session=False
            ).final_decision.decision.value
        )

        any_flag = bool(
            gt["ground_truth_hallucination"]
            or gt["ground_truth_pii"]
            or gt["ground_truth_toxicity"]
            or gt["ground_truth_bias"]
            or gt["ground_truth_cost_anomaly"]
        )
        gt_any.append(any_flag)
        gt_clean.append(not any_flag)

    n = len(dataset)
    blend = [0.6 * f + 0.4 * c for f, c in zip(fused, consequence)]

    clean_total = sum(gt_clean)
    risky_total = sum(gt_any)

    def _flat_mode(name: str, scores: list[float], comment: str) -> AblationMode:
        intervened = [s >= threshold for s in scores]
        disturbed = sum(1 for i, clean in zip(intervened, gt_clean) if i and clean)
        # For a flat score, any flag == "a human must look".
        missed = sum(1 for i, risky in zip(intervened, gt_any) if risky and not i)
        return AblationMode(
            mode=name,
            threshold=threshold,
            intervention_rate=rate(sum(intervened), n),
            caught_risky=binary_metrics(gt_any, intervened),
            clean_disturbed_rate=rate(disturbed, clean_total),
            human_escalation_rate=rate(sum(intervened), n),
            clean_human_escalation_rate=rate(disturbed, clean_total),
            risky_missed_by_escalation=missed,
            comment=comment,
        )

    modes = [
        _flat_mode("performance_only", perf, "Blind to PII, toxicity, bias and cost."),
        _flat_mode("responsibility_only", resp, "Blind to hallucination and cost."),
        _flat_mode("cost_only", cost, "Blind to every safety dimension."),
        _flat_mode("fused_only", fused, "Sees all dimensions but ignores consequence and context."),
        _flat_mode(
            "fused_plus_consequence",
            blend,
            "Adds consequence weighting; still a single flat cut-off, no per-application policy.",
        ),
    ]

    def _pipeline_mode(name: str, tiers: list[str], comment: str) -> AblationMode:
        intervened = [t != InterventionTier.ALLOW.value for t in tiers]
        escalated = [
            t in (InterventionTier.HUMAN_REVIEW.value, InterventionTier.BLOCK.value)
            for t in tiers
        ]
        disturbed = sum(1 for i, clean in zip(intervened, gt_clean) if i and clean)
        clean_escalated = sum(1 for e, clean in zip(escalated, gt_clean) if e and clean)
        missed = sum(1 for e, risky in zip(escalated, gt_any) if risky and not e)
        return AblationMode(
            mode=name,
            threshold=None,
            intervention_rate=rate(sum(intervened), n),
            caught_risky=binary_metrics(gt_any, intervened),
            clean_disturbed_rate=rate(disturbed, clean_total),
            human_escalation_rate=rate(sum(escalated), n),
            clean_human_escalation_rate=rate(clean_escalated, clean_total),
            risky_missed_by_escalation=missed,
            tier_distribution=distribution(tiers),
            comment=comment,
        )

    modes.append(
        _pipeline_mode(
            "risk_only_no_confidence",
            no_conf_tier,
            "Full pipeline but the confidence-aware policy rules are OFF: risk is "
            "acted on without asking how sure we are.",
        )
    )
    modes.append(
        _pipeline_mode(
            "full_pipeline",
            full_tier,
            "Multi-dimensional detection + fusion + consequence + criticality + "
            "confidence-aware per-application policy. The only mode that assigns a "
            "proportionate tier instead of a binary flag.",
        )
    )

    best_flat = max(
        (m for m in modes if m.threshold is not None),
        key=lambda m: m.caught_risky["f1"],
    )
    full = next(m for m in modes if m.mode == "full_pipeline")
    no_conf = next(m for m in modes if m.mode == "risk_only_no_confidence")
    block_hr_full = full.tier_distribution.get("BLOCK", 0) + full.tier_distribution.get(
        "HUMAN_REVIEW", 0
    )
    block_hr_noconf = no_conf.tier_distribution.get("BLOCK", 0) + no_conf.tier_distribution.get(
        "HUMAN_REVIEW", 0
    )
    if block_hr_noconf != block_hr_full:
        conf_line = (
            f"Confidence-awareness also matters: turning it off routes "
            f"{block_hr_noconf} cases to BLOCK/HUMAN_REVIEW vs {block_hr_full} with it "
            f"on — high-risk / low-confidence cases sent to a human for verification "
            f"instead of an automatic block."
        )
    else:
        conf_line = (
            f"Confidence-awareness makes no difference on THIS synthetic set "
            f"({block_hr_full} BLOCK/HUMAN_REVIEW either way): the lexical NLI produces "
            f"confident contradictions and low-risk unverifieds, so the "
            f"'high-risk / low-confidence' regime the LOW_CONFIDENCE_HIGH_RISK rule "
            f"targets is rare here (it is unit-tested directly)."
        )
    conclusion = (
        f"Best flat single-score mode ('{best_flat.mode}') catches risky cases at "
        f"F1={best_flat.caught_risky['f1']:.2f} (recall "
        f"{best_flat.caught_risky['recall']:.2f}) — it misses much because it sees "
        f"only one dimension. The full pipeline reaches F1="
        f"{full.caught_risky['f1']:.2f} (recall {full.caught_risky['recall']:.2f}) "
        f"while escalating {full.clean_human_escalation_rate:.0%} of clean traffic to "
        f"a human, and assigns a proportionate tier rather than a binary flag. "
        + conf_line
    )

    return AblationReport(
        n_cases=n, threshold=threshold, modes=modes, conclusion=conclusion
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run_ablation().model_dump(), indent=2, default=str))
