"""
Data contracts for the Performance Detector.

The Performance Detector answers a single question: "How well is this
AI response supported by the available evidence/context?" It never
sees ground truth, never computes consequence factors, and never
decides an intervention tier — those belong to later layers.

Key distinction: UNVERIFIED does NOT mean hallucinated. It means the
available evidence was insufficient to verify the claim one way or
the other — a claim can be UNVERIFIED without being wrong.

This module is intentionally framework-agnostic:
- No pandas / numpy / sklearn / transformers imports.
- No configuration loading.
- No detector implementation logic (no embeddings, no NLI, no file I/O).

It defines data contracts only.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ==================================================
# ENUMS
# ==================================================


class NLILabel(str, Enum):
    """Raw natural-language-inference label for a single claim/evidence pair."""

    ENTAILMENT = "ENTAILMENT"
    CONTRADICTION = "CONTRADICTION"
    NEUTRAL = "NEUTRAL"


class ClaimStatus(str, Enum):
    """Per-claim verification outcome after retrieval + NLI."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NEUTRAL = "NEUTRAL"
    NO_EVIDENCE = "NO_EVIDENCE"


class ResponseStatus(str, Enum):
    """Overall response-level support status, aggregated across claims."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


# ==================================================
# EVIDENCE MATCH
# ==================================================


class EvidenceMatch(BaseModel):
    """A single retrieved evidence chunk considered for a claim."""

    text: str

    similarity: float = Field(ge=0, le=1)

    rank: int = Field(ge=1)


# ==================================================
# CLAIM RESULT
# ==================================================


class ClaimResult(BaseModel):
    """Verification outcome for a single extracted claim from the response."""

    claim: str

    status: ClaimStatus

    nli_label: NLILabel | None = None

    nli_confidence: float | None = Field(default=None, ge=0, le=1)

    top_evidence: list[EvidenceMatch] = Field(default_factory=list)

    explanation: str

    # --- Round 2 upgrade: evidence-weighted per-claim assessment ---
    # Best retrieval similarity found for this claim (0.0 if none retrieved).
    retrieval_similarity: float = Field(default=0.0, ge=0, le=1)
    # Blended strength of the evidence backing (or refuting) this claim:
    # retrieval similarity + NLI confidence + evidence availability.
    evidence_strength: float = Field(default=0.0, ge=0, le=1)
    # How much this claim contributes to response-level performance risk,
    # in [0,1]. A confident contradiction over strong evidence >> a weakly
    # related, low-confidence neutral claim.
    claim_risk: float = Field(default=0.0, ge=0, le=1)
    # Heuristic importance of this claim from its own text (money amounts,
    # action verbs, named entities). Production-visible; no ground truth.
    text_criticality: float = Field(default=0.0, ge=0, le=1)


# ==================================================
# LATENCY
# ==================================================


class LatencyBreakdown(BaseModel):
    """Per-stage timing breakdown for a single performance-detector run."""

    claim_extraction_ms: float = Field(ge=0)
    chunking_ms: float = Field(ge=0)
    embedding_ms: float = Field(ge=0)
    retrieval_ms: float = Field(ge=0)
    nli_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


# ==================================================
# PERFORMANCE RESULT
# ==================================================


class PerformanceResult(BaseModel):
    """
    Full output contract of the Performance Detector for one interaction.

    Semantic definitions
    ---------------------
    grounding_score:
        How strongly the response's claims are supported by available
        evidence. Higher means better-grounded. ``None`` when it
        cannot be meaningfully computed (e.g. no evidence at all).

    performance_risk:
        Estimated probability/severity of performance failure
        (hallucination, contradiction, or unverifiable claims),
        represented on [0,1]. This is a risk estimate, not a
        ground-truth label.

    confidence:
        Confidence in the overall detector assessment itself — how
        much the detector trusts its own ``status``/``performance_risk``
        conclusion.

    verification_confidence:
        How much of the response was actually verifiable with
        sufficient evidence, independent of whether what was
        verified was supported or contradicted.

    evidence_available:
        Whether usable context/evidence was available at all for this
        interaction. When ``False``, claims are more likely to be
        ``UNVERIFIED`` rather than ``CONTRADICTED``.

    method:
        The actual backend used to produce this result, e.g.
        ``"tfidf+lexical_nli"`` or ``"minilm+deberta_nli"``.
    """

    status: ResponseStatus

    grounding_score: float | None = Field(default=None, ge=0, le=1)

    performance_risk: float = Field(ge=0, le=1)

    confidence: float = Field(ge=0, le=1)

    verification_confidence: float = Field(ge=0, le=1)

    # --- Round 2 upgrade: explicit risk / confidence separation ---
    # 1 - confidence, surfaced so callers never have to derive it.
    uncertainty: float = Field(default=0.0, ge=0, le=1)
    # Aggregate quality of the evidence used for this assessment
    # (mean per-claim evidence strength). Low value => low confidence.
    evidence_quality: float = Field(default=0.0, ge=0, le=1)

    claim_results: list[ClaimResult] = Field(default_factory=list)

    evidence_available: bool

    method: str

    explanation: str

    latency: LatencyBreakdown