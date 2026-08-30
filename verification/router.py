"""
VerificationRouter — decides how much verification effort an AI response
deserves.

    responses
        -> cheap checks (responsibility + cost)   always
        -> SHALLOW performance pass               always (top_k = 1)
        -> preliminary risk + confidence
        -> consequence + criticality (existing engines)
        -> disagreement / uncertainty score
        -> FAST  or  DEEP  (config-driven)
        -> if DEEP: full performance pass (top_k = 3)
        -> detector outputs + VerificationReport

The deep pass is NEVER run before the routing decision — that would
defeat the purpose. FAST cases reuse the shallow result.

All thresholds live in ``config/settings.yaml`` under ``verification``.
Everything here is deterministic; latency is measured, never fabricated.
"""

from __future__ import annotations

from statistics import pstdev
from typing import Any

from common.timing import Stopwatch, clamp01
from consequence.engine import ConsequenceEngine
from criticality.engine import CriticalityEngine
from data.schemas import Interaction
from detectors.cost.detector import CostDetector
from detectors.performance.schemas import ClaimStatus, PerformanceResult
from detectors.responsibility.detector import ResponsibilityDetector
from fusion.engine import RiskFusionEngine
from settings import load_settings
from verification.backend import LexicalDeepVerifier, VerificationBackend
from verification.cascade_router import CascadeRouter
from verification.cascade_router import Interaction as _CascadeInteraction
from verification.cascade_router import RouteVerdict
from verification.routing_models import build_default_embedding_mf_router
from verification.schemas import (
    DisagreementBreakdown,
    VerificationPath,
    VerificationReport,
)


_INFORMATIONAL_ACTIONS = frozenset(
    {"", "none", "information", "informational", "info", "answer", "chat"}
)


def _to_cascade_interaction(interaction: Interaction) -> _CascadeInteraction:
    """Adapt a production ``data.schemas.Interaction`` to the cascade's frozen view."""
    action_type = getattr(interaction.action_type, "value", str(interaction.action_type))
    amount = float(getattr(interaction, "action_amount_inr", 0.0) or 0.0)
    # ``affected_entities`` on an informational response is "who the answer is
    # about", not "who an action touches" — only forward it for a genuine
    # side-effecting action, so the cascade does not treat every answer that
    # mentions a customer as an action.
    is_actioned = action_type not in _INFORMATIONAL_ACTIONS or amount > 0.0
    return _CascadeInteraction(
        interaction_id=getattr(interaction, "interaction_id", "") or "",
        application=getattr(interaction.application, "value", str(interaction.application)),
        prompt=getattr(interaction, "prompt", "") or "",
        response=getattr(interaction, "response", "") or "",
        context=getattr(interaction, "context", "") or "",
        action_type=action_type,
        action_amount=amount,
        affected_entities=(
            int(getattr(interaction, "affected_entities", 0) or 0) if is_actioned else 0
        ),
        tool_calls=int(getattr(interaction, "tool_calls", 0) or 0),
        retry_count=int(getattr(interaction, "retry_count", 0) or 0),
    )

ROUTER_NAME = "verification_router"

# Disagreement-score component weights (documented; sum to 1.0).
_W_RISK_SPREAD = 0.35
_W_WEAK_EVIDENCE = 0.25
_W_NEUTRAL_RATE = 0.15
_W_LOW_CONFIDENCE = 0.15
_W_MISSING_EVIDENCE = 0.10
_RISK_SPREAD_SCALE = 0.35  # pstdev of 3 dimension risks rarely exceeds this
_LOW_CONF_ANCHOR = 0.60


class VerificationRouter:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        responsibility: ResponsibilityDetector | None = None,
        cost: CostDetector | None = None,
        criticality: CriticalityEngine | None = None,
        consequence: ConsequenceEngine | None = None,
        fusion: RiskFusionEngine | None = None,
        backend: VerificationBackend | None = None,
        cascade: CascadeRouter | None = None,
    ) -> None:
        settings = config if config is not None else load_settings()
        vcfg = settings.get("verification", {}) or {}

        self.enabled = bool(vcfg.get("enabled", True))
        self.fast_path_max_risk = float(vcfg.get("fast_path_max_risk", 0.35))
        self.fast_path_min_confidence = float(vcfg.get("fast_path_min_confidence", 0.70))
        self.low_risk_floor = float(vcfg.get("low_risk_floor", 0.15))
        self.deep_risk_threshold = float(vcfg.get("deep_verification_risk_threshold", 0.35))
        self.deep_consequence_threshold = float(
            vcfg.get("deep_verification_consequence_threshold", 0.60)
        )
        self.deep_extreme_factor = float(vcfg.get("deep_verification_extreme_factor", 0.85))
        self.deep_criticality_threshold = float(
            vcfg.get("deep_verification_criticality_threshold", 0.60)
        )
        self.disagreement_trigger = float(vcfg.get("disagreement_trigger", 0.45))
        self.always_deep_on_missing_evidence = bool(
            vcfg.get("always_deep_on_missing_evidence", True)
        )
        # Deterministic semantic bypass. When a critical outbound-PII hard
        # boundary is present the PolicyEngine will BLOCK regardless of the
        # grounding score, so the semantic pass (claim extraction / TF-IDF /
        # NLI) is skipped. Off by default: turning it on changes the reported
        # performance_risk for those interactions (never the decision).
        self.bypass_semantics_on_hard_boundary = bool(
            vcfg.get("bypass_semantics_on_hard_boundary", True)
        )
        # Tiered Cascade Router.
        #   cascade_shadow_mode: true   -> verdict is telemetry only (legacy
        #                                  _deep_reasons logic drives FAST/DEEP)
        #   cascade_shadow_mode: false  -> AUTHORITATIVE: the ML-driven cascade
        #                                  decides FAST vs DEEP, subject to the
        #                                  Tier-0.5 deterministic overrides in
        #                                  route() (which can only force DEEP).
        self.cascade_shadow = bool(vcfg.get("cascade_shadow_mode", False))
        # The ML-driven Tier-1 model: a matrix-factorisation router over a
        # hashing-embedding of the interaction. Injectable for tests / a
        # differently-trained checkpoint.
        self.cascade = cascade or CascadeRouter(
            classifier=build_default_embedding_mf_router()
        )
        self.parallel_fast_checks = bool(vcfg.get("parallel_fast_checks", False))
        self.shallow_top_k = int(vcfg.get("shallow_top_k", 1))
        self.deep_top_k = int(vcfg.get("deep_top_k", 3))

        self.responsibility = responsibility or ResponsibilityDetector(settings)
        self.cost = cost or CostDetector(settings)
        self.criticality = criticality or CriticalityEngine(settings)
        self.consequence = consequence or ConsequenceEngine(settings)
        self.fusion = fusion or RiskFusionEngine(settings)
        self.backend = backend or LexicalDeepVerifier(settings)

    # ------------------------------------------------------------------

    def route(
        self,
        interaction: Interaction,
        watch: Stopwatch | None = None,
        stage_latency: dict[str, float] | None = None,
    ) -> tuple[PerformanceResult, Any, Any, VerificationReport]:
        watch = watch or Stopwatch()
        stage_latency = stage_latency if stage_latency is not None else {}

        # -------- fast path: cheap checks + shallow performance --------
        with watch.stage("fast_path"):
            with watch.stage("responsibility"):
                responsibility = self.responsibility.detect(interaction)
            with watch.stage("cost"):
                cost = self.cost.detect(interaction)

            bypass_reason = (
                self._deterministic_hard_boundary(responsibility)
                if self.bypass_semantics_on_hard_boundary
                else None
            )

            cascade_decision = None
            if bypass_reason is None:
                with watch.stage("performance_shallow"):
                    perf_shallow = self.backend.verify(interaction, depth="shallow")
                consequence = self.consequence.assess(interaction)
                crit_prelim = self.criticality.assess(interaction, perf_shallow)
                prelim_risk, prelim_confidence = self._light_fuse(
                    perf_shallow, responsibility, cost, crit_prelim
                )
                disagreement = self._disagreement(
                    perf_shallow, responsibility, cost, crit_prelim, prelim_confidence
                )

                # ---- Tier-0.5 Deterministic Override -------------------
                # Business-logic safety gates the ML router structurally
                # cannot see (it is not fed consequence / criticality /
                # evidence quality / preliminary risk). Any of these forces
                # DEEP regardless of the cascade's semantic prediction.
                tier_0_5 = self._deep_reasons(
                    prelim_risk, prelim_confidence, consequence, crit_prelim,
                    disagreement, perf_shallow.evidence_available,
                )

                # ---- Tier 1: the ML-driven cascade router --------------
                with watch.stage("cascade"):
                    cascade_decision = self.cascade.route(
                        _to_cascade_interaction(interaction)
                    )
                stage_latency["cascade_ms"] = watch.get("cascade")
                cascade_deep = (
                    cascade_decision.verdict is RouteVerdict.ROUTE_TO_DEEP
                )

                if self.cascade_shadow:
                    # legacy: cascade verdict is telemetry only
                    reasons = tier_0_5
                else:
                    # AUTHORITATIVE: DEEP iff a Tier-0.5 gate fired OR the
                    # cascade escalated. The cascade owns the FAST decision
                    # for interactions that are deterministically clean.
                    reasons = list(tier_0_5)
                    if cascade_deep and cascade_decision.reason_code not in {
                        "DETERMINISTIC_HARD_BOUNDARY"
                    }:
                        reasons.append("TIER1_CASCADE_COMPLEXITY")
            else:
                # Deterministic hard boundary: skip claim extraction / TF-IDF /
                # NLI. The interaction still reaches the ResponsibilityDetector
                # (already run above) and the PolicyEngine, which blocks on the
                # critical-PII override.
                perf_shallow = self.backend.verify(
                    interaction, depth="deep",
                    bypass_semantics=True, bypass_reason=bypass_reason,
                )
                crit_prelim = self.criticality.assess(interaction, perf_shallow)
                prelim_risk, prelim_confidence = self._light_fuse(
                    perf_shallow, responsibility, cost, crit_prelim
                )
                disagreement = self._disagreement(
                    perf_shallow, responsibility, cost, crit_prelim, prelim_confidence
                )
                reasons = ["DETERMINISTIC_HARD_BOUNDARY"]

        fast_ms = watch.get("fast_path")
        stage_latency["fast_path_ms"] = fast_ms
        stage_latency["responsibility_ms"] = watch.get("responsibility")
        stage_latency["cost_ms"] = watch.get("cost")
        stage_latency["performance_shallow_ms"] = watch.get("performance_shallow")

        path = VerificationPath.DEEP if (reasons or not self.enabled) else VerificationPath.FAST

        # -------- deep path: only if justified, and not already bypassed --------
        deep_ms = 0.0
        if path is VerificationPath.DEEP and bypass_reason is None:
            with watch.stage("deep_path"):
                performance = self.backend.verify(interaction, depth="deep")
            deep_ms = watch.get("deep_path")
        else:
            performance = perf_shallow
        stage_latency["deep_path_ms"] = deep_ms
        stage_latency["performance_ms"] = (
            watch.get("performance_shallow") + deep_ms
        )

        crit_final = self.criticality.assess(interaction, performance)
        final_risk, final_confidence = self._light_fuse(
            performance, responsibility, cost, crit_final
        )

        deep_forced = path is VerificationPath.DEEP and not (
            {"HIGH_PRELIMINARY_RISK", "LOW_CONFIDENCE", "DETECTOR_DISAGREEMENT"}
            & set(reasons)
        )

        cascade = self._cascade_telemetry(cascade_decision, path)

        report = VerificationReport(
            verification_path=path,
            deep_trigger_reasons=reasons,
            reason_for_deep_verification="; ".join(reasons),
            deep_was_forced=deep_forced,
            preliminary_risk=round(prelim_risk, 4),
            preliminary_confidence=round(prelim_confidence, 4),
            final_risk=round(final_risk, 4),
            final_confidence=round(final_confidence, 4),
            disagreement_score=round(disagreement.score, 4),
            disagreement_breakdown=disagreement,
            evidence_available=performance.evidence_available,
            fast_path_latency_ms=round(fast_ms, 3),
            deep_path_latency_ms=round(deep_ms, 3),
            total_verification_latency_ms=round(fast_ms + deep_ms, 3),
            shallow_top_k=self.shallow_top_k,
            deep_top_k=self.deep_top_k,
            semantics_bypassed=bypass_reason is not None,
            bypass_reason=bypass_reason or "",
            cascade_verdict=cascade["verdict"],
            cascade_reason_code=cascade["reason_code"],
            cascade_tier=cascade["tier"],
            predicted_complexity_score=cascade["score"],
            cascade_threshold=cascade["threshold"],
            cascade_agrees_with_router=cascade["agrees"],
            explanation=(
                self._explain(path, reasons, prelim_risk, prelim_confidence)
                if bypass_reason is None
                else (
                    f"DEEP path, semantic verification bypassed: {bypass_reason}. "
                    "Claim extraction / retrieval / NLI were skipped because the "
                    "policy layer will block on the deterministic responsibility "
                    "override regardless of the grounding score."
                )
            ),
        )
        return performance, responsibility, cost, report

    # ------------------------------------------------------------------

    def _light_fuse(self, performance, responsibility, cost, criticality) -> tuple[float, float]:
        """Preliminary/final risk + confidence using the SAME fusion engine
        the decision engine uses (so the numbers are consistent)."""
        crit_input = max(
            criticality.action_criticality, criticality.max_claim_criticality
        )
        amplified = self.criticality.amplify_performance_risk(
            performance.performance_risk, crit_input
        )
        fused = self.fusion.fuse_scores(
            amplified,
            responsibility.overall_responsibility_risk,
            cost.cost_risk,
            performance_confidence=performance.confidence,
            responsibility_confidence=responsibility.confidence,
            cost_confidence=cost.confidence,
        )
        return fused.overall_risk, fused.confidence

    def _disagreement(
        self, performance, responsibility, cost, criticality, prelim_confidence
    ) -> DisagreementBreakdown:
        crit_input = max(
            criticality.action_criticality, criticality.max_claim_criticality
        )
        amplified = self.criticality.amplify_performance_risk(
            performance.performance_risk, crit_input
        )
        risks = [
            amplified,
            responsibility.overall_responsibility_risk,
            cost.cost_risk,
        ]
        risk_spread = clamp01(pstdev(risks) / _RISK_SPREAD_SCALE)
        weak_evidence = clamp01(1.0 - performance.evidence_quality)

        total_claims = len(performance.claim_results)
        neutral = sum(
            1 for c in performance.claim_results if c.status == ClaimStatus.NEUTRAL
        )
        neutral_rate = (neutral / total_claims) if total_claims else 0.0

        missing_evidence = 0.0 if performance.evidence_available else 1.0
        low_confidence = clamp01(
            (_LOW_CONF_ANCHOR - prelim_confidence) / _LOW_CONF_ANCHOR
        )

        score = clamp01(
            _W_RISK_SPREAD * risk_spread
            + _W_WEAK_EVIDENCE * weak_evidence
            + _W_NEUTRAL_RATE * neutral_rate
            + _W_LOW_CONFIDENCE * low_confidence
            + _W_MISSING_EVIDENCE * missing_evidence
        )
        return DisagreementBreakdown(
            risk_spread=round(risk_spread, 4),
            weak_evidence=round(weak_evidence, 4),
            neutral_rate=round(neutral_rate, 4),
            missing_evidence=round(missing_evidence, 4),
            low_confidence=round(low_confidence, 4),
            score=round(score, 4),
        )

    def _deep_reasons(
        self, prelim_risk, prelim_confidence, consequence, criticality,
        disagreement, evidence_available,
    ) -> list[str]:
        reasons: list[str] = []
        if (
            prelim_risk >= self.deep_risk_threshold
            or prelim_risk > self.fast_path_max_risk
        ):
            reasons.append("HIGH_PRELIMINARY_RISK")
        if (
            prelim_confidence < self.fast_path_min_confidence
            and prelim_risk >= self.low_risk_floor
        ):
            reasons.append("LOW_CONFIDENCE")

        factors = consequence.factors
        extreme_factor = max(factors.financial_impact, factors.reversibility)
        if (
            consequence.consequence_score >= self.deep_consequence_threshold
            or extreme_factor >= self.deep_extreme_factor
        ):
            reasons.append("HIGH_CONSEQUENCE")
        if criticality.action_criticality >= self.deep_criticality_threshold:
            reasons.append("HIGH_CRITICALITY")
        if disagreement.score >= self.disagreement_trigger:
            reasons.append("DETECTOR_DISAGREEMENT")
        if not evidence_available and self.always_deep_on_missing_evidence:
            reasons.append("MISSING_EVIDENCE")
        return reasons

    @staticmethod
    def _cascade_telemetry(cascade_decision, path) -> dict[str, Any]:
        """
        Shape the cascade's :class:`RoutingDecision` into the plain dict the
        VerificationReport records. ``cascade_decision`` is ``None`` only on the
        deterministic-hard-boundary bypass path (the cascade is not consulted
        there — the boundary already forces DEEP).
        """
        if cascade_decision is None:
            return {
                "verdict": "ROUTE_TO_DEEP", "reason_code": "DETERMINISTIC_HARD_BOUNDARY",
                "tier": "FAST_PATH_DETERMINISTIC", "score": None,
                "threshold": None, "agrees": path is VerificationPath.DEEP,
            }
        router_deep = path is VerificationPath.DEEP
        return {
            "verdict": cascade_decision.verdict.value,
            "reason_code": cascade_decision.reason_code,
            "tier": cascade_decision.tier_reached.name,
            "score": cascade_decision.predicted_complexity_score,
            "threshold": cascade_decision.applied_threshold,
            "agrees": (
                cascade_decision.verdict is RouteVerdict.ROUTE_TO_DEEP
            ) == router_deep,
        }

    @staticmethod
    def _deterministic_hard_boundary(responsibility) -> str | None:
        """
        Return a short reason string when the responsibility layer has already
        found a *deterministic* hard boundary that the PolicyEngine will block
        on no matter what the grounding score is — currently critical outbound
        PII. In that case the semantic verification pass is pure waste.

        Returns ``None`` when there is no such boundary (the normal case).
        """
        if getattr(responsibility, "contains_critical_pii", False):
            types = getattr(responsibility, "critical_pii_types", None) or ["PII"]
            return "CRITICAL_PII:" + ",".join(sorted(types))
        return None

    @staticmethod
    def _explain(path, reasons, prelim_risk, prelim_confidence) -> str:
        if path is VerificationPath.FAST:
            return (
                f"FAST path: preliminary risk {prelim_risk:.2f} at confidence "
                f"{prelim_confidence:.2f} — low risk, high confidence, low "
                "consequence and no detector disagreement, so a full verification "
                "pass is not justified."
            )
        pretty = {
            "HIGH_PRELIMINARY_RISK": "preliminary risk is high",
            "LOW_CONFIDENCE": "preliminary confidence is low",
            "HIGH_CONSEQUENCE": "the action has high consequence if wrong",
            "HIGH_CRITICALITY": "the action is highly critical",
            "DETECTOR_DISAGREEMENT": "the detector signals disagree",
            "MISSING_EVIDENCE": "no usable evidence was retrieved",
        }
        because = "; ".join(pretty.get(r, r) for r in reasons)
        return (
            f"DEEP path: a full verification pass was run because {because} "
            f"(preliminary risk {prelim_risk:.2f}, confidence {prelim_confidence:.2f})."
        )
