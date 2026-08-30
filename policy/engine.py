"""
Policy Engine.

Maps a fused risk picture + consequence + application + action context
to a *proposed* intervention tier. All thresholds live in
``config/settings.yaml`` under ``policy`` (one profile per application,
falling back to ``policy.default``); the engine itself is a small, fixed
set of composable rules rather than a sprawl of hard-coded conditionals.

Rule model
----------
1.  Risk bands map ``overall_risk`` to a base tier.
2.  Consequence escalation raises the tier when consequence is high — the
    same risk with high consequence lands harder (this is the core idea:
    likelihood + consequence, not likelihood alone).
3.  Dimension rules impose *minimum* tiers (contradiction, unverified
    evidence, high responsibility risk, low confidence).
4.  Hard overrides set the tier directly (critical PII, severe toxicity).
5.  The proposed tier is the most severe outcome of all of the above.
"""

from __future__ import annotations

from typing import Any

from common.reason_codes import ReasonCode, dedupe
from data.schemas import InterventionTier
from policy.schemas import (
    TIER_RANK,
    PolicyDecision,
    PolicyInput,
    RuleTraceEntry,
    tier_escalate,
    tier_max,
)
from settings import load_settings

ENGINE_NAME = "policy"


class PolicyEngine:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        settings = config if config is not None else load_settings()
        self._policy = settings["policy"]

    # ------------------------------------------------------------------

    def profile_for(self, application: str) -> dict[str, Any]:
        default = dict(self._policy["default"])
        default.update(self._policy.get(application, {}))
        return default

    def decide(self, signals: PolicyInput) -> PolicyDecision:
        profile = self.profile_for(signals.application)
        trace: list[RuleTraceEntry] = []
        triggered: list[str] = []

        base_tier = self._risk_band(signals.overall_risk, profile)
        trace.append(
            RuleTraceEntry(
                rule="RISK_BAND",
                fired=True,
                effect=f"base={base_tier.value}",
                detail=(
                    f"overall_risk {signals.overall_risk:.2f} maps to base tier "
                    f"{base_tier.value} for profile '{signals.application}'."
                ),
            )
        )

        tier = base_tier

        # --- consequence escalation ---
        # Fires on the weighted consequence score OR on a single extreme
        # factor (a large financial action / a near-irreversible action is
        # high-consequence even if the blended score is only moderate).
        cons_trigger = float(profile["consequence_escalation_trigger"])
        factor_trigger = float(profile["extreme_factor_trigger"])
        extreme_factor = max(signals.financial_impact, signals.irreversibility)
        if signals.consequence_score >= cons_trigger or extreme_factor >= factor_trigger:
            steps = int(profile["consequence_escalation_steps"])
            escalated = tier_escalate(tier, steps)
            fired = escalated != tier
            if signals.consequence_score >= cons_trigger:
                why = f"consequence_score {signals.consequence_score:.2f} >= {cons_trigger:.2f}"
            else:
                why = (
                    f"an individual consequence factor is extreme "
                    f"({extreme_factor:.2f} >= {factor_trigger:.2f})"
                )
            trace.append(
                RuleTraceEntry(
                    rule="HIGH_CONSEQUENCE",
                    fired=fired,
                    effect=f"{tier.value}->{escalated.value}" if fired else "no-op",
                    detail=f"{why}; escalate {steps} step(s).",
                )
            )
            if fired:
                triggered.append("HIGH_CONSEQUENCE")
            tier = tier_max(tier, escalated)

        # --- minimum-tier dimension rules ---
        tier = self._apply_min_rule(
            tier, trace, triggered,
            condition=(
                signals.action_criticality >= float(profile["criticality_trigger"])
                or signals.financial_impact >= float(profile["extreme_factor_trigger"])
            ),
            rule="HIGH_CRITICALITY_ACTION",
            min_tier=InterventionTier(profile["criticality_min_tier"]),
            detail=(
                f"action criticality {signals.action_criticality:.2f} / financial "
                f"impact {signals.financial_impact:.2f}: a large consequential action "
                "always gets at least a verification pass."
            ),
        )
        tier = self._apply_min_rule(
            tier, trace, triggered,
            condition=signals.performance_status == "CONTRADICTED",
            rule="PERFORMANCE_CONTRADICTION",
            min_tier=InterventionTier(profile["contradiction_min_tier"]),
            detail="response contains a claim that contradicts supplied evidence.",
        )
        tier = self._apply_min_rule(
            tier, trace, triggered,
            condition=(
                signals.performance_status == "UNVERIFIED"
                and signals.performance_risk >= 0.4
            ),
            rule="UNVERIFIED_EVIDENCE",
            min_tier=InterventionTier(profile["unverified_min_tier"]),
            detail="response claims could not be verified against available evidence.",
        )
        tier = self._apply_min_rule(
            tier, trace, triggered,
            condition=signals.responsibility_risk >= float(profile["high_responsibility_trigger"]),
            rule="HIGH_RESPONSIBILITY_RISK",
            min_tier=InterventionTier(profile["high_responsibility_min_tier"]),
            detail=(
                f"responsibility risk {signals.responsibility_risk:.2f} "
                f">= {float(profile['high_responsibility_trigger']):.2f}."
            ),
        )
        if signals.confidence_aware:
            tier = self._apply_min_rule(
                tier, trace, triggered,
                condition=signals.confidence <= float(profile["low_confidence_trigger"]),
                rule="LOW_DETECTOR_CONFIDENCE",
                min_tier=InterventionTier(profile["low_confidence_min_tier"]),
                detail=(
                    f"fused detector confidence {signals.confidence:.2f} "
                    f"<= {float(profile['low_confidence_trigger']):.2f}; prefer human oversight."
                ),
            )

        # --- hard overrides ---
        if signals.contains_critical_pii:
            override = InterventionTier(profile["critical_pii_action"])
            trace.append(
                RuleTraceEntry(
                    rule="CRITICAL_PII",
                    fired=True,
                    effect=f"force>={override.value}",
                    detail=(
                        "response exposes critical PII "
                        f"({', '.join(signals.critical_pii_types) or 'sensitive identifier'})."
                    ),
                )
            )
            triggered.append("CRITICAL_PII")
            tier = tier_max(tier, override)

        tox_trigger = float(profile["toxicity_block_trigger"])
        if signals.toxicity_risk >= tox_trigger:
            trace.append(
                RuleTraceEntry(
                    rule="SEVERE_TOXICITY",
                    fired=True,
                    effect=f"force>={InterventionTier.BLOCK.value}",
                    detail=(
                        f"toxicity risk {signals.toxicity_risk:.2f} >= {tox_trigger:.2f}."
                    ),
                )
            )
            triggered.append("SEVERE_TOXICITY")
            tier = tier_max(tier, InterventionTier.BLOCK)

        # --- cost-only cap ---
        # An operational cost anomaly with no safety signal is an ops
        # concern, not a safety block: cap the tier accordingly.
        safety_rules = {
            "PERFORMANCE_CONTRADICTION",
            "UNVERIFIED_EVIDENCE",
            "HIGH_RESPONSIBILITY_RISK",
            "CRITICAL_PII",
            "SEVERE_TOXICITY",
        }
        cost_only_cap = InterventionTier(profile["cost_only_max_tier"])
        if (
            signals.dominant_dimension == "cost"
            and signals.performance_risk < 0.4
            and signals.responsibility_risk < 0.4
            and not (safety_rules & set(triggered))
            and TIER_RANK[tier] > TIER_RANK[cost_only_cap]
        ):
            trace.append(
                RuleTraceEntry(
                    rule="COST_ONLY_CAP",
                    fired=True,
                    effect=f"{tier.value}->{cost_only_cap.value}",
                    detail=(
                        "cost is the only elevated dimension and no safety rule "
                        f"fired; capped at {cost_only_cap.value} for operational review."
                    ),
                )
            )
            triggered.append("COST_ONLY_CAP")
            tier = cost_only_cap

        # --- risk + confidence rule ---
        # A would-be BLOCK that rests on weak evidence is routed to a human
        # rather than auto-blocked. Hard, evidence-backed overrides
        # (critical PII, severe toxicity) are exempt.
        hard_override = {"CRITICAL_PII", "SEVERE_TOXICITY"} & set(triggered)
        if (
            signals.confidence_aware
            and not hard_override
            and tier == InterventionTier.BLOCK
            and signals.overall_risk >= float(profile["low_confidence_high_risk_risk_trigger"])
            and signals.confidence <= float(profile["low_confidence_high_risk_conf_trigger"])
        ):
            capped = InterventionTier(profile["low_confidence_high_risk_cap"])
            if TIER_RANK[capped] < TIER_RANK[tier]:
                trace.append(
                    RuleTraceEntry(
                        rule="LOW_CONFIDENCE_HIGH_RISK",
                        fired=True,
                        effect=f"{tier.value}->{capped.value}",
                        detail=(
                            f"risk {signals.overall_risk:.2f} is high but detector "
                            f"confidence {signals.confidence:.2f} is low; route to a "
                            "human instead of an automatic block."
                        ),
                    )
                )
                triggered.append("LOW_CONFIDENCE_HIGH_RISK")
                tier = capped

        self._annotate_tiers(trace)
        requires_human = tier in (InterventionTier.HUMAN_REVIEW, InterventionTier.BLOCK)
        reason_codes = self._reason_codes(signals, triggered)
        explanation = self._explain(signals, base_tier, tier, triggered)

        return PolicyDecision(
            application=signals.application,
            proposed_tier=tier,
            base_tier=base_tier,
            requires_human_review=requires_human,
            triggered_rules=triggered,
            reason_codes=reason_codes,
            rule_trace=trace,
            explanation=explanation,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_tiers(trace: list[RuleTraceEntry]) -> None:
        """
        Walk the rule trace in order, tracking the running intervention
        tier, and stamp ``tier_after`` on every entry. This is the single
        source of truth for ``DecisionTrace.decision_path``.
        """
        running = InterventionTier.ALLOW
        for entry in trace:
            effect = entry.effect
            if entry.rule == "RISK_BAND" and effect.startswith("base="):
                running = InterventionTier(effect.split("=", 1)[1])
            elif entry.fired:
                if effect.startswith(("min>=", "force>=")):
                    target = InterventionTier(effect.split(">=", 1)[1].split()[0])
                    running = tier_max(running, target)
                elif "->" in effect:
                    running = InterventionTier(effect.split("->", 1)[1].split()[0])
            entry.tier_after = running

    @staticmethod
    def _reason_codes(signals: PolicyInput, triggered: list[str]) -> list[str]:
        rule_to_code = {
            "PERFORMANCE_CONTRADICTION": ReasonCode.CONTRADICTED_EVIDENCE,
            "UNVERIFIED_EVIDENCE": ReasonCode.LOW_VERIFICATION_COVERAGE,
            "HIGH_RESPONSIBILITY_RISK": ReasonCode.HIGH_RESPONSIBILITY_RISK,
            "CRITICAL_PII": ReasonCode.CRITICAL_PII,
            "SEVERE_TOXICITY": ReasonCode.TOXICITY,
            "HIGH_CONSEQUENCE": ReasonCode.HIGH_CONSEQUENCE,
            "HIGH_CRITICALITY_ACTION": ReasonCode.HIGH_CONSEQUENCE,
            "COST_ONLY_CAP": ReasonCode.COST_SPIKE,
            "LOW_DETECTOR_CONFIDENCE": ReasonCode.LOW_CONFIDENCE_HIGH_RISK,
            "LOW_CONFIDENCE_HIGH_RISK": ReasonCode.LOW_CONFIDENCE_HIGH_RISK,
        }
        codes: list[ReasonCode | str] = []
        for rule in triggered:
            if rule in rule_to_code:
                codes.append(rule_to_code[rule])
        # From responsibility layer directly (PII_EXPOSURE / bias / etc).
        codes.extend(signals.responsibility_reason_codes)
        if signals.performance_risk >= 0.7 and "CONTRADICTED_EVIDENCE" not in [
            c.value if isinstance(c, ReasonCode) else c for c in codes
        ]:
            codes.append(ReasonCode.HIGH_PERFORMANCE_RISK)
        if signals.multi_risk:
            codes.append(ReasonCode.MULTI_RISK)
        return dedupe(codes)

    # ------------------------------------------------------------------

    @staticmethod
    def _risk_band(overall_risk: float, profile: dict[str, Any]) -> InterventionTier:
        if overall_risk <= float(profile["allow_max_risk"]):
            return InterventionTier.ALLOW
        if overall_risk <= float(profile["annotate_max_risk"]):
            return InterventionTier.ANNOTATE
        if overall_risk <= float(profile["verify_max_risk"]):
            return InterventionTier.VERIFY
        if overall_risk <= float(profile["human_review_max_risk"]):
            return InterventionTier.HUMAN_REVIEW
        return InterventionTier.BLOCK

    @staticmethod
    def _apply_min_rule(
        tier: InterventionTier,
        trace: list[RuleTraceEntry],
        triggered: list[str],
        *,
        condition: bool,
        rule: str,
        min_tier: InterventionTier,
        detail: str,
    ) -> InterventionTier:
        if not condition:
            return tier
        new_tier = tier_max(tier, min_tier)
        fired = new_tier != tier
        trace.append(
            RuleTraceEntry(
                rule=rule,
                fired=fired,
                effect=f"min>={min_tier.value}" if fired else f"min>={min_tier.value} (already met)",
                detail=detail,
            )
        )
        if fired:
            triggered.append(rule)
        return new_tier

    @staticmethod
    def _explain(
        signals: PolicyInput,
        base_tier: InterventionTier,
        final_tier: InterventionTier,
        triggered: list[str],
    ) -> str:
        if final_tier == base_tier and not triggered:
            return (
                f"{final_tier.value}: overall risk {signals.overall_risk:.2f} with "
                f"consequence {signals.consequence_score:.2f} falls in the "
                f"{final_tier.value} band for {signals.application}; no additional "
                "policy rules fired."
            )
        reasons = {
            "HIGH_CONSEQUENCE": "the associated action has high consequence if wrong",
            "PERFORMANCE_CONTRADICTION": "the response contradicts supplied evidence",
            "UNVERIFIED_EVIDENCE": "key claims could not be verified against evidence",
            "HIGH_RESPONSIBILITY_RISK": "a high-confidence responsibility signal was raised",
            "LOW_DETECTOR_CONFIDENCE": "detector confidence was low",
            "CRITICAL_PII": "the response exposes critical personal data",
            "SEVERE_TOXICITY": "the response contains severe toxic language",
        }
        because = "; ".join(reasons[r] for r in triggered if r in reasons)
        if not because:
            because = (
                f"the fused risk of {signals.overall_risk:.2f} exceeds the "
                f"acceptable level for {signals.application}"
            )
        return (
            f"{final_tier.value} (base {base_tier.value}) because {because}."
        )


def run_policy(signals: PolicyInput, config: dict[str, Any] | None = None) -> PolicyDecision:
    """Convenience one-shot wrapper."""
    return PolicyEngine(config=config).decide(signals)
