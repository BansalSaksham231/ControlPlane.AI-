"""
Decision Engine — the ControlPlane pipeline orchestrator.

    Interaction
      -> Performance / Responsibility / Cost detectors   (each: risk + confidence)
      -> Claim / Action Criticality                      (amplifies performance risk)
      -> Risk Fusion                                     (overall_risk + overall confidence)
      -> Consequence Engine
      -> (optional) Session risk accumulation
      -> Policy Engine                                   (tier + structured reason codes)
      -> FinalDecision  (+ full DecisionTrace for audit)

Risk vs confidence are kept explicitly separate at every stage. The
detectors are independent and could run concurrently in production
(``parallel_detectors=True`` demonstrates this; results are identical).
Ground truth is never touched.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from common.reason_codes import ReasonCode, dedupe
from common.timing import Stopwatch, clamp01
from consequence.engine import ConsequenceEngine
from criticality.engine import CriticalityEngine
from data.schemas import FinalDecision, Interaction, InterventionTier
from decision.schemas import DecisionDriver, DecisionPathStep, DecisionTrace
from detectors.cost.baseline import CostBaseline
from detectors.cost.detector import CostDetector
from detectors.performance.detector import PerformanceDetector
from detectors.responsibility.detector import ResponsibilityDetector
from fusion.engine import RiskFusionEngine
from policy.engine import PolicyEngine
from policy.schemas import PolicyInput
from settings import load_settings
from verification.backend import LexicalDeepVerifier
from verification.router import VerificationRouter

ENGINE_NAME = "decision"


class DecisionEngine:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        cost_baseline: CostBaseline | None = None,
        session_manager: Any | None = None,
        confidence_aware: bool = True,
        parallel_detectors: bool = False,
        use_verification_router: bool = True,
    ) -> None:
        self._config = config if config is not None else load_settings()
        self.performance = PerformanceDetector(self._config)
        self.responsibility = ResponsibilityDetector(self._config)
        self.cost = CostDetector(self._config, baseline=cost_baseline)
        self.criticality = CriticalityEngine(self._config)
        self.fusion = RiskFusionEngine(self._config)
        self.consequence = ConsequenceEngine(self._config)
        self.policy = PolicyEngine(self._config)
        self.session_manager = session_manager
        self.confidence_aware = confidence_aware
        self.parallel_detectors = parallel_detectors
        self.use_verification_router = use_verification_router

        # Progressive Verification (Phase 2). The router shares this
        # engine's detector/engine instances so config is parsed once.
        self.verification_router = VerificationRouter(
            self._config,
            responsibility=self.responsibility,
            cost=self.cost,
            criticality=self.criticality,
            consequence=self.consequence,
            fusion=self.fusion,
            backend=LexicalDeepVerifier(self._config, detector=self.performance),
        )

    # ------------------------------------------------------------------

    def evaluate(
        self,
        interaction: Interaction,
        *,
        timestamp: datetime | None = None,
        record_session: bool = True,
        policy_profile: str | None = None,
    ) -> DecisionTrace:
        watch = Stopwatch()
        stage_latency: dict[str, float] = {}

        verification_report = None
        if self.use_verification_router:
            performance, responsibility, cost, verification_report = (
                self.verification_router.route(interaction, watch, stage_latency)
            )
            verification_path = verification_report.verification_path.value
        else:
            performance, responsibility, cost = self._run_detectors(
                interaction, watch, stage_latency
            )
            verification_path = "DEEP"

        with watch.stage("criticality"):
            criticality = self.criticality.assess(interaction, performance)
        stage_latency["criticality_ms"] = watch.get("criticality")

        # Amplify performance risk by how much this response matters if wrong.
        crit_input = max(
            criticality.action_criticality, criticality.max_claim_criticality
        )
        crit_weighted_perf_risk = self.criticality.amplify_performance_risk(
            performance.performance_risk, crit_input
        )

        with watch.stage("fusion"):
            fusion = self.fusion.fuse_scores(
                crit_weighted_perf_risk,
                responsibility.overall_responsibility_risk,
                cost.cost_risk,
                performance_confidence=performance.confidence,
                responsibility_confidence=responsibility.confidence,
                cost_confidence=cost.confidence,
            )
        stage_latency["fusion_ms"] = watch.get("fusion")

        with watch.stage("consequence"):
            consequence = self.consequence.assess(interaction)
        stage_latency["consequence_ms"] = watch.get("consequence")

        pre_session_risk = fusion.overall_risk
        effective_risk = pre_session_risk
        session_info: dict[str, Any] | None = None
        session_triggered: list[str] = []

        if self.session_manager is not None:
            with watch.stage("session"):
                session_info = self.session_manager.contribution(
                    interaction.session_id, pre_session_risk
                )
            stage_latency["session_ms"] = watch.get("session")
            effective_risk = clamp01(
                session_info.get("adjusted_overall_risk", pre_session_risk)
            )
            if session_info.get("escalated"):
                session_triggered.append("SESSION_RISK_ESCALATION")

        with watch.stage("policy"):
            policy_input = PolicyInput.from_results(
                application=policy_profile
                or getattr(interaction.application, "value", interaction.application),
                action_type=getattr(
                    interaction.action_type, "value", interaction.action_type
                ),
                fusion=fusion,
                consequence=consequence,
                performance=performance,
                responsibility=responsibility,
                criticality=criticality,
                confidence_aware=self.confidence_aware,
            )
            policy_input = policy_input.model_copy(update={"overall_risk": effective_risk})
            policy_decision = self.policy.decide(policy_input)
        stage_latency["policy_ms"] = watch.get("policy")

        tier = policy_decision.proposed_tier
        triggered_rules = list(policy_decision.triggered_rules) + session_triggered
        if (
            consequence.factors.action_automation >= 0.8
            and tier != InterventionTier.ALLOW
            and "AUTOMATED_ACTION" not in triggered_rules
        ):
            triggered_rules.append("AUTOMATED_ACTION")
        if fusion.severity_floor_applied and "SEVERITY_FLOOR" not in triggered_rules:
            triggered_rules.append("SEVERITY_FLOOR")

        reason_codes = self._reason_codes(
            policy_decision, criticality, cost, fusion, session_triggered, tier
        )
        decision_confidence = self._decision_confidence(
            fusion, responsibility, policy_decision, tier
        )
        drivers = [
            DecisionDriver(rule=e.rule, effect=e.effect, detail=e.detail)
            for e in policy_decision.rule_trace
            if e.fired and e.rule != "RISK_BAND"
        ]
        decision_path = self._decision_path(policy_decision)
        explanation = self._explanation(
            tier, policy_decision, performance, responsibility, cost,
            consequence, criticality, fusion, reason_codes, session_info,
        )

        resolved_ts = timestamp or datetime.now(timezone.utc)
        final_decision = FinalDecision(
            performance_risk=fusion.performance_risk,
            responsibility_risk=fusion.responsibility_risk,
            cost_risk=fusion.cost_risk,
            consequence=consequence.factors,
            decision=tier,
            overall_risk=round(effective_risk, 4),
            decision_confidence=round(decision_confidence, 4),
            explanation=explanation,
            triggered_rules=triggered_rules,
            reason_codes=reason_codes,
            verification_path=verification_path,
            timestamp=resolved_ts,
        )

        trace = DecisionTrace(
            interaction_id=interaction.interaction_id,
            timestamp=resolved_ts,
            application=getattr(interaction.application, "value", interaction.application),
            action_type=getattr(interaction.action_type, "value", interaction.action_type),
            model=getattr(interaction.model, "value", interaction.model),
            final_decision=final_decision,
            performance=performance,
            responsibility=responsibility,
            cost=cost,
            criticality=criticality,
            fusion=fusion,
            consequence=consequence,
            policy=policy_decision,
            session=session_info,
            criticality_weighted_performance_risk=round(crit_weighted_perf_risk, 4),
            decision_drivers=drivers,
            verification_path=verification_path,
            verification=verification_report,
            decision_path=decision_path,
            pre_session_overall_risk=round(pre_session_risk, 4),
            latency_ms=watch.total_ms(),
            stage_latency_ms=stage_latency,
            detectors_parallel=self.parallel_detectors,
        )

        if self.session_manager is not None and record_session:
            self.session_manager.record(
                interaction.session_id,
                overall_risk=effective_risk,
                decision=tier.value,
                interaction_id=interaction.interaction_id,
                timestamp=resolved_ts,
                dimension_risks=(
                    fusion.performance_risk,
                    fusion.responsibility_risk,
                    fusion.cost_risk,
                ),
                reason_codes=reason_codes,
                tier_changing_rules=[d.rule for d in drivers],
                pii_entity_keys=self._pii_entity_keys(responsibility),
                critical=self._is_critical_turn(tier, responsibility),
                critical_trigger=self._critical_trigger(tier, responsibility),
            )

        return trace

    def decide(self, interaction: Interaction, **kwargs: Any) -> FinalDecision:
        """Return just the compact ``FinalDecision``."""
        return self.evaluate(interaction, **kwargs).final_decision

    # ------------------------------------------------------------------

    def _run_detectors(self, interaction, watch, stage_latency):
        """
        Run the three independent detectors. They share no state, so in
        production they run concurrently; here that is opt-in and produces
        byte-identical results (all detectors are deterministic).
        """
        if self.parallel_detectors:
            with watch.stage("detectors"):
                with ThreadPoolExecutor(max_workers=3) as pool:
                    f_perf = pool.submit(self.performance.detect, interaction)
                    f_resp = pool.submit(self.responsibility.detect, interaction)
                    f_cost = pool.submit(self.cost.detect, interaction)
                    performance, responsibility, cost = (
                        f_perf.result(), f_resp.result(), f_cost.result()
                    )
            wall = watch.get("detectors")
            stage_latency["detectors_ms"] = wall
            stage_latency["performance_ms"] = performance.latency.total_ms
            stage_latency["responsibility_ms"] = responsibility.latency_ms
            stage_latency["cost_ms"] = cost.latency_ms
            return performance, responsibility, cost

        with watch.stage("performance"):
            performance = self.performance.detect(interaction)
        stage_latency["performance_ms"] = watch.get("performance")
        with watch.stage("responsibility"):
            responsibility = self.responsibility.detect(interaction)
        stage_latency["responsibility_ms"] = watch.get("responsibility")
        with watch.stage("cost"):
            cost = self.cost.detect(interaction)
        stage_latency["cost_ms"] = watch.get("cost")
        return performance, responsibility, cost

    @staticmethod
    def _decision_path(policy_decision) -> list[DecisionPathStep]:
        """
        How the intervention tier was reached, step by step. Built purely
        from the policy rule trace (each entry now carries ``tier_after``).
        """
        steps: list[DecisionPathStep] = []
        prev = InterventionTier.ALLOW
        for entry in policy_decision.rule_trace:
            after = entry.tier_after or prev
            if entry.rule == "RISK_BAND" or (entry.fired and after != prev):
                steps.append(
                    DecisionPathStep(
                        rule=entry.rule,
                        from_tier=prev,
                        to_tier=after,
                        reason=entry.detail[:160],
                    )
                )
            prev = after
        return steps

    # ------------------------------------------------------------------
    # session snapshot signals (PII-safe: redacted keys only)
    # ------------------------------------------------------------------

    @staticmethod
    def _pii_entity_keys(responsibility) -> list[str]:
        findings = getattr(getattr(responsibility, "pii", None), "findings", []) or []
        return sorted(
            {
                f"{getattr(f, 'subtype', 'pii')}:{getattr(f, 'redacted_text', '')}"
                for f in findings
                if getattr(f, "redacted_text", "")
            }
        )

    @staticmethod
    def _is_critical_turn(tier, responsibility) -> bool:
        if tier is InterventionTier.BLOCK:
            return True
        if getattr(responsibility, "contains_critical_pii", False):
            return True
        tox = getattr(getattr(responsibility, "toxicity", None), "findings", []) or []
        return any(getattr(f, "severity", None) == "CRITICAL" for f in tox)

    @classmethod
    def _critical_trigger(cls, tier, responsibility) -> str:
        if getattr(responsibility, "contains_critical_pii", False):
            return "CRITICAL_PII"
        tox = getattr(getattr(responsibility, "toxicity", None), "findings", []) or []
        if any(getattr(f, "severity", None) == "CRITICAL" for f in tox):
            return "SEVERE_TOXICITY"
        if tier is InterventionTier.BLOCK:
            return "BLOCK"
        return ""

    @staticmethod
    def _reason_codes(
        policy_decision, criticality, cost, fusion, session_triggered, tier
    ) -> list[str]:
        codes: list[ReasonCode | str] = []
        if tier != InterventionTier.ALLOW:
            codes.extend(policy_decision.reason_codes)
            codes.extend(criticality.reason_codes)
        # cost anomaly types -> reason codes (informational unless the tier
        # moved because of them).
        cost_map = {
            "RETRY_SPIKE": ReasonCode.RETRY_ANOMALY,
            "TOOL_LOOP": ReasonCode.TOOL_LOOP,
            "LATENCY_SPIKE": ReasonCode.LATENCY_ANOMALY,
            "TOKEN_SPIKE": ReasonCode.COST_SPIKE,
            "COST_PER_SUCCESS_SPIKE": ReasonCode.COST_SPIKE,
        }
        for anomaly in cost.anomaly_types:
            if anomaly in cost_map:
                codes.append(cost_map[anomaly])
        if fusion.multi_risk and tier != InterventionTier.ALLOW:
            codes.append(ReasonCode.MULTI_RISK)
        if session_triggered:
            codes.append(ReasonCode.SESSION_ESCALATION)
        return dedupe(codes)

    @staticmethod
    def _decision_confidence(
        fusion: Any, responsibility: Any, policy_decision: Any, tier: InterventionTier
    ) -> float:
        base = fusion.confidence
        decisive = {"CRITICAL_PII", "SEVERE_TOXICITY"}
        if decisive & set(policy_decision.triggered_rules):
            return clamp01(max(base, responsibility.confidence))
        if tier == InterventionTier.ALLOW:
            return clamp01(max(base, 0.75))
        return clamp01(base)

    @staticmethod
    def _explanation(
        tier, policy_decision, performance, responsibility, cost,
        consequence, criticality, fusion, reason_codes, session_info,
    ) -> str:
        parts = [policy_decision.explanation]

        risk_line = (
            f"Risk {fusion.overall_risk:.2f} at confidence "
            f"{fusion.confidence:.2f} (uncertainty {fusion.uncertainty:.2f})."
        )
        parts.append(risk_line)

        signals: list[str] = []
        if performance.status.value in ("CONTRADICTED", "PARTIALLY_SUPPORTED"):
            signals.append(f"performance: {performance.status.value.lower()}")
        elif performance.status.value == "UNVERIFIED":
            signals.append("performance: claims unverified against evidence")
        if responsibility.overall_responsibility_risk > 0:
            cats = sorted({f.category.value for f in responsibility.findings})
            signals.append(f"responsibility: {', '.join(cats).lower()}")
        if cost.anomaly_types:
            signals.append(f"cost: {', '.join(cost.anomaly_types).lower()}")
        signals.append(
            f"consequence {consequence.severity_band}, "
            f"criticality {criticality.band}"
        )
        parts.append("Key signals: " + "; ".join(signals) + ".")

        if reason_codes:
            parts.append("Reason codes: " + ", ".join(reason_codes) + ".")
        if session_info and session_info.get("escalated"):
            parts.append(session_info.get("explanation", "Session risk is elevated."))
        return " ".join(parts)


def evaluate_interaction(
    interaction: Interaction,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DecisionTrace:
    """Convenience one-shot wrapper (builds a fresh engine each call)."""
    return DecisionEngine(config=config).evaluate(interaction, **kwargs)
