"""
Unified Responsibility Detector.

Runs the PII, toxicity and bias sub-detectors over a production
Interaction's response and fuses them into a single
``ResponsibilityResult``. Ground truth is never consulted.

A conservative fusion rule is used: the overall responsibility risk is
never allowed to fall far below the single most severe dimension, so a
lone high-confidence PII leak still drives a high overall risk even when
toxicity and bias are clean.
"""

from __future__ import annotations

from typing import Any

from common.reason_codes import ReasonCode, dedupe
from common.timing import Stopwatch, clamp01
from data.schemas import ActionType, Interaction
from detectors.responsibility.bias import BiasDetector
from detectors.responsibility.pii import PIIDetector
from detectors.responsibility.schemas import Finding, ResponsibilityResult
from detectors.responsibility.toxicity import ToxicityDetector
from settings import load_settings

DETECTOR_NAME = "responsibility"


class ResponsibilityDetector:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        settings = config if config is not None else load_settings()
        self._weights = settings["responsibility_detector"]["dimension_weights"]
        self._enabled = {
            "pii": settings.get("responsibility", {}).get("pii", {}).get("enabled", True),
            "toxicity": settings.get("responsibility", {}).get("toxicity", {}).get("enabled", True),
            "bias": settings.get("responsibility", {}).get("bias", {}).get("enabled", True),
        }
        self.pii = PIIDetector(settings)
        self.toxicity = ToxicityDetector(settings)
        self.bias = BiasDetector(settings)

    # ------------------------------------------------------------------

    def detect(
        self, interaction: Interaction | str, action_type: str | None = None
    ) -> ResponsibilityResult:
        if isinstance(interaction, Interaction):
            response = interaction.response
            action_value = (
                interaction.action_type.value
                if isinstance(interaction.action_type, ActionType)
                else str(interaction.action_type)
            )
        else:
            response = interaction or ""
            action_value = action_type

        watch = Stopwatch()
        with watch.stage("pii"):
            pii = self.pii.detect(response, action_type=action_value)
        with watch.stage("toxicity"):
            toxicity = self.toxicity.detect(response)
        with watch.stage("bias"):
            bias = self.bias.detect(response)

        pii_risk = pii.pii_risk if self._enabled["pii"] else 0.0
        toxicity_risk = toxicity.toxicity_risk if self._enabled["toxicity"] else 0.0
        bias_risk = bias.bias_signal if self._enabled["bias"] else 0.0

        overall = self._fuse(pii_risk, toxicity_risk, bias_risk)

        findings: list[Finding] = []
        if self._enabled["pii"]:
            findings.extend(pii.findings)
        if self._enabled["toxicity"]:
            findings.extend(toxicity.findings)
        if self._enabled["bias"]:
            findings.extend(bias.findings)

        confidence = self._confidence(pii, toxicity, bias, pii_risk, toxicity_risk, bias_risk)
        critical_types = list(pii.critical_types) if self._enabled["pii"] else []
        explanation = self._explain(
            pii_risk, toxicity_risk, bias_risk, overall, pii, toxicity, bias
        )
        reason_codes = self._reason_codes(
            pii, pii_risk, toxicity_risk, bias_risk, overall
        )

        return ResponsibilityResult(
            pii_risk=round(pii_risk, 4),
            toxicity_risk=round(toxicity_risk, 4),
            bias_risk=round(bias_risk, 4),
            overall_responsibility_risk=round(overall, 4),
            confidence=round(confidence, 4),
            contains_critical_pii=pii.contains_critical_pii and self._enabled["pii"],
            critical_pii_types=list(critical_types),
            findings=findings,
            redacted_response=pii.redacted_response if self._enabled["pii"] else response,
            pii=pii,
            toxicity=toxicity,
            bias=bias,
            explanation=explanation,
            latency_ms=watch.total_ms(),
            reason_codes=reason_codes,
        )

    # ------------------------------------------------------------------

    def _fuse(self, pii_risk: float, toxicity_risk: float, bias_risk: float) -> float:
        weighted = (
            self._weights["pii"] * pii_risk
            + self._weights["toxicity"] * toxicity_risk
            + self._weights["bias"] * bias_risk
        )
        max_dimension = max(pii_risk, toxicity_risk, bias_risk)
        # Conservative floor: a single severe dimension is not diluted away.
        return clamp01(max(weighted, 0.85 * max_dimension))

    @staticmethod
    def _confidence(
        pii, toxicity, bias, pii_risk: float, toxicity_risk: float, bias_risk: float
    ) -> float:
        active = [
            (pii.confidence, pii_risk),
            (toxicity.confidence, toxicity_risk),
            (bias.confidence, bias_risk),
        ]
        firing = [conf for conf, risk in active if risk > 0.0]
        if not firing:
            # Fairly confident that nothing obvious is present, but lexical
            # methods miss subtle cases — cap accordingly.
            return 0.6
        return clamp01(max(firing))

    @staticmethod
    def _reason_codes(
        pii, pii_risk: float, toxicity_risk: float, bias_risk: float, overall: float
    ) -> list[str]:
        codes: list[ReasonCode] = []
        if pii.contains_critical_pii:
            codes.append(ReasonCode.CRITICAL_PII)
        if pii_risk > 0.0:
            codes.append(ReasonCode.PII_EXPOSURE)
        if toxicity_risk >= 0.5:
            codes.append(ReasonCode.TOXICITY)
        if bias_risk > 0.0:
            codes.append(ReasonCode.POTENTIAL_BIAS_SIGNAL)
        if overall >= 0.6:
            codes.append(ReasonCode.HIGH_RESPONSIBILITY_RISK)
        return dedupe(codes)

    @staticmethod
    def _explain(
        pii_risk, toxicity_risk, bias_risk, overall, pii, toxicity, bias
    ) -> str:
        parts: list[str] = []
        if pii_risk > 0:
            parts.append(pii.explanation)
        if toxicity_risk > 0:
            parts.append(toxicity.explanation)
        if bias_risk > 0:
            parts.append(bias.explanation)
        if not parts:
            return (
                "No responsibility concerns detected: response is free of PII, "
                "toxic language and bias signals by the heuristic detectors."
            )
        dominant = max(
            (("PII", pii_risk), ("toxicity", toxicity_risk), ("bias signal", bias_risk)),
            key=lambda t: t[1],
        )[0]
        parts.append(
            f"Overall responsibility risk {overall:.2f}, dominated by {dominant}."
        )
        return " ".join(parts)


def detect_responsibility(
    interaction: Interaction, config: dict[str, Any] | None = None
) -> ResponsibilityResult:
    """Convenience one-shot wrapper."""
    return ResponsibilityDetector(config=config).detect(interaction)
