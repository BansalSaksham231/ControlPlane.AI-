"""
Verification-backend abstraction.

A ``VerificationBackend`` runs a grounding check at a requested depth. The
default, ``LexicalDeepVerifier``, simply delegates to the existing
``PerformanceDetector`` (no logic duplicated) — ``depth="shallow"`` for
the cheap first pass, ``depth="deep"`` for the full pass.

The abstraction exists so a stronger semantic backend (e.g. a small
fine-tuned NLI model) could be substituted later without touching the
router. The default path stays fully offline, deterministic and
laptop-friendly — no external LLM, no API key, no model download.

Deterministic semantic bypass
-----------------------------
``verify(..., bypass_semantics=True)`` returns a canonical ``UNVERIFIED``
result **without running claim extraction, TF-IDF retrieval or NLI**. It is
used only when the router has already found a deterministic hard boundary
(critical outbound PII) that the PolicyEngine will block on regardless of the
grounding score — so the semantic pass cannot change the outcome and is pure
waste. On a real transformer-NLI backend this saves the entire model forward
pass (~1–3 s per doomed interaction); on the lexical backend it still removes
claim extraction + chunking + retrieval + per-claim NLI. The interaction is
still handed to the ResponsibilityDetector and the PolicyEngine downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from data.schemas import Interaction
from detectors.performance.detector import PerformanceDetector
from detectors.performance.schemas import (
    LatencyBreakdown,
    PerformanceResult,
    ResponseStatus,
)

# A bypassed assessment is a genuine "we did not check grounding" state, so it
# reports the system's canonical no-evidence UNVERIFIED risk (``claim_risk_no_
# evidence``, default 0.50) rather than a made-up low value. The bypass
# guarantees the *decision* is unchanged (the responsibility override blocks
# regardless); it does NOT guarantee every cross-dimension reason code is
# unchanged, because performance is no longer independently measured.
_BYPASS_DEFAULT_RISK = 0.50
_BYPASS_CONFIDENCE = 0.30
_BYPASS_METHOD = "deterministic_semantic_bypass"


def _bypass_result(reason: str, performance_risk: float = _BYPASS_DEFAULT_RISK) -> PerformanceResult:
    """A valid ``PerformanceResult`` produced without any semantic work."""
    return PerformanceResult(
        status=ResponseStatus.UNVERIFIED,
        grounding_score=None,
        performance_risk=performance_risk,
        confidence=_BYPASS_CONFIDENCE,
        verification_confidence=0.0,
        uncertainty=1.0 - _BYPASS_CONFIDENCE,
        evidence_quality=0.0,
        claim_results=[],
        evidence_available=False,
        method=_BYPASS_METHOD,
        explanation=(
            "Semantic verification skipped: a deterministic hard boundary "
            f"({reason}) will be adjudicated by the policy layer, so the "
            "grounding score cannot change the decision."
        ),
        latency=LatencyBreakdown(
            claim_extraction_ms=0.0,
            chunking_ms=0.0,
            embedding_ms=0.0,
            retrieval_ms=0.0,
            nli_ms=0.0,
            total_ms=0.0,
        ),
    )


class VerificationBackend(ABC):
    """Runs a grounding check at ``"shallow"`` or ``"deep"`` depth."""

    name: str = "abstract"

    @abstractmethod
    def verify(
        self,
        interaction: Interaction,
        *,
        depth: str,
        bypass_semantics: bool = False,
        bypass_reason: str = "",
    ) -> PerformanceResult:
        raise NotImplementedError


class LexicalDeepVerifier(VerificationBackend):
    """Default backend: the existing lexical performance detector."""

    name = "lexical"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        detector: PerformanceDetector | None = None,
    ) -> None:
        self._detector = detector or PerformanceDetector(config)

    @property
    def detector(self) -> PerformanceDetector:
        return self._detector

    def verify(
        self,
        interaction: Interaction,
        *,
        depth: str = "deep",
        bypass_semantics: bool = False,
        bypass_reason: str = "",
    ) -> PerformanceResult:
        if bypass_semantics:
            return _bypass_result(
                bypass_reason or "deterministic hard boundary",
                performance_risk=getattr(
                    self._detector, "_cr_no_evidence", _BYPASS_DEFAULT_RISK
                ),
            )
        return self._detector.detect(interaction, depth=depth)
