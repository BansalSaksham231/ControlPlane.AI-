"""
Governance -> Calibration bridge (Phase 9, Step 5).

Consumes governance analytics and (optionally) an existing
``calibration.select.ConfigurationSelection`` to produce
``GovernanceRecommendation`` objects.

It NEVER modifies the production configuration file. The disposition is
always ``RECOMMENDED_FOR_EVALUATION`` or ``REVIEW_REQUIRED`` — never ``APPLIED``.
Calibration math is delegated to ``calibration.sweep`` / ``calibration.select``
— none of it is duplicated here.
"""

from __future__ import annotations

from typing import Any

from governance.schemas import (
    ApplicationComparison,
    GovernanceConfig,
    GovernanceOverview,
    GovernanceRecommendation,
    GovernanceSignalSummary,
    RecommendationDisposition,
    RecommendationType,
)

__all__ = ["build_recommendations", "run_calibration_bridge"]


def run_calibration_bridge(
    config: dict[str, Any] | None = None,
    *,
    minimum_recall: float = 0.90,
    minimum_precision: float = 0.0,
):
    """
    Execute the existing calibration sweep + safety-constrained selection.
    Returns a ``calibration.select.ConfigurationSelection``. Heavy (runs the
    evaluation pipeline once) — call only when a quantified recommendation
    is explicitly requested.
    """
    from calibration.select import (
        EfficiencyObjective,
        SafetyConstraints,
        select_configuration,
    )
    from calibration.sweep import sweep_thresholds
    from settings import load_settings

    cfg = config or load_settings()
    cc = cfg.get("calibration", {})
    sweep = sweep_thresholds(
        risk_thresholds=[float(v) for v in cc.get("grid_risk", [0.35, 0.60, 0.85])],
        confidence_thresholds=[float(v) for v in cc.get("grid_confidence", [0.30, 0.60, 0.90])],
        config=cfg,
    )
    return select_configuration(
        sweep,
        SafetyConstraints(minimum_recall=minimum_recall, minimum_precision=minimum_precision),
        EfficiencyObjective.MIN_HUMAN_REVIEW,
    )


def _threshold_recommendation_from_selection(selection: Any) -> GovernanceRecommendation | None:
    """Map a ConfigurationSelection onto a threshold recommendation."""
    if selection is None or getattr(selection, "status", "") != "SELECTED":
        return None
    base = selection.baseline_result
    sel = selection.selected_result
    tradeoff = (
        f"human-review rate {base.efficiency.human_review_rate:.2f} -> "
        f"{sel.efficiency.human_review_rate:.2f}; "
        f"recall {base.safety.recall:.2f} -> {sel.safety.recall:.2f}; "
        f"FAST-path {base.efficiency.fast_path_rate:.0%} -> {sel.efficiency.fast_path_rate:.0%}; "
        f"avg latency {base.efficiency.average_latency_ms:.2f}ms -> "
        f"{sel.efficiency.average_latency_ms:.2f}ms"
    )
    return GovernanceRecommendation(
        recommendation_type=RecommendationType.REVIEW_THRESHOLD,
        application=None,
        rationale=(
            "Human-oversight rate is elevated system-wide while reviewer disagreement "
            "is low. calibration.select found a safety-passing verification-threshold "
            "configuration with lower human-review workload. " + selection.selection_reason
        ),
        evidence={
            "objective": selection.objective.value,
            "eligible_candidates": selection.eligible_candidate_count,
            "total_candidates": selection.total_candidate_count,
        },
        current_configuration={k: round(v, 4) for k, v in base.resolved_thresholds.items()},
        candidate_configuration={k: round(v, 4) for k, v in sel.resolved_thresholds.items()},
        expected_tradeoff=tradeoff,
        safety_constraints=selection.safety_constraints.as_dict(),
        disposition=RecommendationDisposition.RECOMMENDED_FOR_EVALUATION,
        points_to=["calibration.sweep", "calibration.select"],
    )


def build_recommendations(
    overview: GovernanceOverview,
    comparison: ApplicationComparison,
    signals: GovernanceSignalSummary,
    *,
    config: GovernanceConfig | None = None,
    calibration_selection: Any | None = None,
) -> list[GovernanceRecommendation]:
    config = config or GovernanceConfig()
    recs: list[GovernanceRecommendation] = []

    system_oversight = overview.decisions.human_oversight_rate or 0.0
    system_override = overview.reviewer_disagreement.override_rate
    low_disagreement = system_override is None or system_override < 0.20

    # ---- system-wide threshold review (calibration-backed when available) ----
    if system_oversight > config.high_human_review_rate and low_disagreement:
        quantified = _threshold_recommendation_from_selection(calibration_selection)
        if quantified is not None:
            recs.append(quantified)
        else:
            recs.append(
                GovernanceRecommendation(
                    recommendation_type=RecommendationType.REVIEW_THRESHOLD,
                    application=None,
                    rationale=(
                        f"Human-oversight rate is {system_oversight:.0%} (threshold "
                        f"{config.high_human_review_rate:.0%}) with low reviewer "
                        "disagreement. The FAST/DEEP verification thresholds are "
                        "candidates for calibration."
                    ),
                    evidence={
                        "human_oversight_rate": round(system_oversight, 4),
                        "reviewer_override_rate": system_override,
                        "deep_rate": overview.verification.deep_rate,
                    },
                    disposition=RecommendationDisposition.REVIEW_REQUIRED,
                    points_to=["calibration.sweep", "calibration.select"],
                )
            )

    # ---- per-application policy review (reviewer disagreement) ----
    for a in comparison.applications:
        if a.volume < config.min_application_volume:
            continue
        override = a.reviewer_override_rate
        if override is not None and override > config.high_override_rate:
            recs.append(
                GovernanceRecommendation(
                    recommendation_type=RecommendationType.REVIEW_POLICY,
                    application=a.application,
                    rationale=(
                        f"Reviewers disagreed with {override:.0%} of reviewed automated "
                        f"decisions for '{a.application}' "
                        f"(block rate {a.block_rate or 0:.0%}). The policy profile for "
                        "this application should be reviewed by a human — this is a "
                        "governance signal, not a correctness finding."
                    ),
                    evidence={
                        "reviewer_override_rate": round(override, 4),
                        "block_rate": a.block_rate,
                        "human_oversight_rate": a.human_oversight_rate,
                        "volume": a.volume,
                    },
                    disposition=RecommendationDisposition.REVIEW_REQUIRED,
                    points_to=["policy profile: " + a.application, "investigation"],
                )
            )

    # ---- per-application detector-coverage review (low confidence) ----
    for a in comparison.applications:
        if a.volume < config.min_application_volume:
            continue
        if (a.low_confidence_rate or 0.0) > config.low_confidence_rate:
            recs.append(
                GovernanceRecommendation(
                    recommendation_type=RecommendationType.REVIEW_DETECTOR,
                    application=a.application,
                    rationale=(
                        f"{a.low_confidence_rate:.0%} of '{a.application}' decisions are "
                        "low-confidence, which usually indicates missing / weak evidence "
                        "rather than wrong decisions. Detector coverage for this traffic "
                        "is worth a review."
                    ),
                    evidence={
                        "low_confidence_rate": round(a.low_confidence_rate, 4),
                        "deep_rate": a.deep_rate,
                    },
                    disposition=RecommendationDisposition.REVIEW_REQUIRED,
                    points_to=["detectors/performance", "detectors/responsibility"],
                )
            )

    if not recs:
        recs.append(
            GovernanceRecommendation(
                recommendation_type=RecommendationType.NO_ACTION,
                application=None,
                rationale=(
                    "No governance threshold was crossed. Human-oversight, reviewer "
                    "disagreement, confidence and routing are within configured bounds."
                ),
                evidence={
                    "human_oversight_rate": overview.decisions.human_oversight_rate,
                    "reviewer_override_rate": overview.reviewer_disagreement.override_rate,
                },
                disposition=RecommendationDisposition.NO_ACTION,
            )
        )

    _type_rank = {
        RecommendationType.REVIEW_POLICY: 0,
        RecommendationType.REVIEW_THRESHOLD: 1,
        RecommendationType.REVIEW_APPLICATION: 2,
        RecommendationType.REVIEW_DETECTOR: 3,
        RecommendationType.NO_ACTION: 4,
    }
    recs.sort(key=lambda r: (_type_rank[r.recommendation_type], r.application or ""))
    return recs
