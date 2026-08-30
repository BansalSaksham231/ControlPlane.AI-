"""
Integrated Performance Detector.

Wires together the four independent building blocks — claim extraction,
context chunking, TF-IDF retrieval, and lexical NLI — into a single
detector that answers one question:

    "How well is this AI response supported by the supplied evidence?"

Critical semantics (do not regress these):

* UNVERIFIED != CONTRADICTED. When there is no usable evidence, claims
  are reported as ``NO_EVIDENCE`` and the response as ``UNVERIFIED`` —
  the detector never pretends an unverifiable response is hallucinated.
* The detector never sees ground truth, never computes consequence
  factors, and never selects an intervention tier.

The backend is deliberately lightweight (TF-IDF + rule-based lexical
NLI) so the system runs on a laptop with no model downloads. The
``EmbeddingBackend`` / ``NLIBackend`` abstractions allow a stronger
transformer backend to be injected later without changing this file.
"""

from __future__ import annotations

import re
from statistics import fmean
from typing import Any

from common.timing import Stopwatch, clamp01
from data.schemas import Interaction
from detectors.performance.chunker import chunk_context, extract_claims
from detectors.performance.embeddings import (
    EmbeddingBackend,
    TfidfEmbeddingBackend,
    rank_documents,
)
from detectors.performance.nli import (
    CoverageNLIBackend,
    LexicalNLIBackend,
    NLIBackend,
)
from detectors.performance.schemas import (
    ClaimResult,
    ClaimStatus,
    EvidenceMatch,
    LatencyBreakdown,
    NLILabel,
    PerformanceResult,
    ResponseStatus,
)
from settings import load_settings

DETECTOR_NAME = "performance"

_MONEY_RE = re.compile(r"(?:₹|\$|\brs\.?\b|\binr\b|\busd\b)\s?[\d,]{2,}", re.IGNORECASE)
_BIG_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{5,}\b")
_ENTITY_RE = re.compile(
    r"\b(account|customer|employee|email|phone|card|ssn|aadhaar|address|refund|payment)\b",
    re.IGNORECASE,
)


def _load_action_verbs(settings: dict[str, Any]) -> set[str]:
    verbs = settings.get("criticality", {}).get("action_verbs", [])
    return {v.lower() for v in verbs} or {
        "approve", "cancel", "delete", "refund", "transfer", "pay", "send",
    }


class PerformanceDetector:
    """Retrieval + NLI grounding detector operating on production Interactions."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        embedding_backend: EmbeddingBackend | None = None,
        nli_backend: NLIBackend | None = None,
    ) -> None:
        settings = config if config is not None else load_settings()
        cfg = settings["performance"]
        self._cfg = cfg

        self.backend_mode: str = cfg.get("backend_mode", "lite")
        self.min_claim_tokens: int = int(cfg.get("min_claim_tokens", 2))
        self.top_k: int = int(cfg.get("top_k_evidence", 3))
        self.retrieval_threshold: float = (
            float(cfg["retrieval_threshold_transformer"])
            if self.backend_mode == "transformer"
            else float(cfg["retrieval_threshold_lite"])
        )
        self.entailment_threshold: float = float(cfg["entailment_threshold"])
        self.contradiction_threshold: float = float(cfg["contradiction_threshold"])
        self.unverified_risk: float = float(cfg["unverified_performance_risk"])
        self.ceiling_base: float = float(cfg["contradiction_ceiling_base"])
        self.ceiling_step: float = float(cfg["contradiction_ceiling_step"])
        self.ceiling_floor: float = float(cfg["contradiction_ceiling_floor"])

        # --- evidence-weighted claim scoring (Round 2 upgrade) ---
        esw = cfg.get("evidence_strength_weights", {})
        self._w_similarity = float(esw.get("similarity", 0.4))
        self._w_nli = float(esw.get("nli_confidence", 0.4))
        self._w_availability = float(esw.get("availability", 0.2))
        self._cr_contra_base = float(cfg.get("claim_risk_contradiction_base", 0.60))
        self._cr_contra_span = float(cfg.get("claim_risk_contradiction_span", 0.37))
        self._cr_no_evidence = float(cfg.get("claim_risk_no_evidence", 0.50))
        self._cr_neutral = float(cfg.get("claim_risk_neutral", 0.42))
        self._cr_supported_max = float(cfg.get("claim_risk_supported_max", 0.08))
        self._text_crit_weight = float(cfg.get("claim_text_criticality_weight", 0.5))
        self._action_verbs = _load_action_verbs(settings)

        # --- progressive verification (Phase 2): evidence breadth per depth ---
        # DEEP is exactly the pre-existing behaviour (``top_k_evidence``).
        # SHALLOW retrieves fewer chunks; it is a strict subset.
        vcfg = settings.get("verification", {}) or {}
        self._shallow_top_k = max(1, int(vcfg.get("shallow_top_k", 1)))
        self._deep_top_k = self.top_k

        self.embeddings: EmbeddingBackend = embedding_backend or TfidfEmbeddingBackend()
        self.nli: NLIBackend = nli_backend or CoverageNLIBackend()
        self._method = self._resolve_method()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def detect(
        self,
        interaction: Interaction | str,
        context: str | None = None,
        *,
        depth: str = "deep",
    ) -> PerformanceResult:
        """
        Assess grounding for one interaction.

        Accepts either an :class:`Interaction` (production input) or a raw
        ``(response, context)`` pair for lightweight component testing.

        ``depth`` controls verification breadth (progressive verification):

        * ``"deep"`` (default, unchanged) — retrieve ``top_k_evidence``
          chunks per claim and run NLI over every competitive chunk.
        * ``"shallow"`` — retrieve only ``verification.shallow_top_k``
          (default 1) chunks per claim, i.e. one NLI call. This is a
          strict computational *subset* of the deep pass: same extraction,
          chunking, thresholds, scoring and aggregation — just less
          retrieval / NLI work. On unambiguous cases it agrees with the
          deep pass; ambiguous cases are what the router escalates to deep.
        """
        if isinstance(interaction, Interaction):
            response_text = interaction.response
            context_text = interaction.context
        else:
            response_text = interaction or ""
            context_text = context or ""

        effective_top_k = (
            self._shallow_top_k if depth == "shallow" else self._deep_top_k
        )

        watch = Stopwatch()

        with watch.stage("claim_extraction"):
            claims = extract_claims(response_text, min_tokens=self.min_claim_tokens)
        with watch.stage("chunking"):
            chunks = chunk_context(context_text)

        evidence_available = len(chunks) > 0
        claim_results: list[ClaimResult] = []

        for claim in claims:
            claim_results.append(
                self._verify_claim(claim, chunks, watch, top_k=effective_top_k)
            )

        result = self._aggregate(claim_results, evidence_available, watch)
        return result

    # ------------------------------------------------------------------
    # per-claim verification
    # ------------------------------------------------------------------

    def _verify_claim(
        self, claim: str, chunks: list[str], watch: Stopwatch, *, top_k: int | None = None
    ) -> ClaimResult:
        top_k = self._deep_top_k if top_k is None else top_k
        text_criticality = self._text_criticality(claim)

        if not chunks:
            return self._build_claim_result(
                claim=claim,
                status=ClaimStatus.NO_EVIDENCE,
                label=None,
                nli_confidence=None,
                top_evidence=[],
                best_similarity=0.0,
                chunks_present=False,
                retrieved_strongly=False,
                text_criticality=text_criticality,
                explanation=(
                    "No context was supplied, so this claim could not be "
                    "checked against any evidence."
                ),
            )

        with watch.stage("embedding"):
            scores = self.embeddings.similarity(claim, chunks)
        with watch.stage("retrieval"):
            ranked = rank_documents(claim, chunks, scores, top_k=top_k)

        top_evidence = [
            EvidenceMatch(text=chunks[index], similarity=clamp01(score), rank=position)
            for position, (index, score) in enumerate(ranked, start=1)
        ]

        best_similarity = ranked[0][1] if ranked else 0.0
        if not ranked or best_similarity < self.retrieval_threshold:
            return self._build_claim_result(
                claim=claim,
                status=ClaimStatus.NO_EVIDENCE,
                label=None,
                nli_confidence=None,
                top_evidence=top_evidence,
                best_similarity=best_similarity,
                chunks_present=True,
                retrieved_strongly=False,
                text_criticality=text_criticality,
                explanation=(
                    "The supplied context contained nothing sufficiently "
                    f"similar to this claim (best similarity {best_similarity:.2f} "
                    f"< retrieval threshold {self.retrieval_threshold:.2f}); "
                    "treated as unverified, not contradicted."
                ),
            )

        label, confidence = self._best_nli(claim, top_evidence, watch)
        status = self._claim_status(label, confidence)
        explanation = self._claim_explanation(status, label, confidence, best_similarity)
        return self._build_claim_result(
            claim=claim,
            status=status,
            label=label,
            nli_confidence=confidence,
            top_evidence=top_evidence,
            best_similarity=best_similarity,
            chunks_present=True,
            retrieved_strongly=True,
            text_criticality=text_criticality,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # evidence-weighted per-claim scoring (Round 2 upgrade)
    # ------------------------------------------------------------------

    def _build_claim_result(
        self,
        *,
        claim: str,
        status: ClaimStatus,
        label: NLILabel | None,
        nli_confidence: float | None,
        top_evidence: list[EvidenceMatch],
        best_similarity: float,
        chunks_present: bool,
        retrieved_strongly: bool,
        text_criticality: float,
        explanation: str,
    ) -> ClaimResult:
        nli_conf = float(nli_confidence) if nli_confidence is not None else 0.0

        if retrieved_strongly:
            availability_signal = 1.0
        elif chunks_present:
            availability_signal = 0.3
        else:
            availability_signal = 0.0

        evidence_strength = clamp01(
            self._w_similarity * clamp01(best_similarity)
            + self._w_nli * nli_conf
            + self._w_availability * availability_signal
        )

        claim_risk = self._claim_risk(status, best_similarity, nli_conf, evidence_strength)

        return ClaimResult(
            claim=claim,
            status=status,
            nli_label=label,
            nli_confidence=None if nli_confidence is None else round(nli_conf, 4),
            top_evidence=top_evidence,
            explanation=explanation,
            retrieval_similarity=round(clamp01(best_similarity), 4),
            evidence_strength=round(evidence_strength, 4),
            claim_risk=round(claim_risk, 4),
            text_criticality=round(text_criticality, 4),
        )

    def _claim_risk(
        self,
        status: ClaimStatus,
        similarity: float,
        nli_conf: float,
        evidence_strength: float,
    ) -> float:
        if status == ClaimStatus.CONTRADICTED:
            # Confident contradiction over strong retrieval -> near-maximal.
            sim_factor = min(1.0, similarity / 0.6)
            raw = self._cr_contra_base + self._cr_contra_span * nli_conf
            return clamp01(raw * (0.75 + 0.25 * sim_factor))
        if status == ClaimStatus.SUPPORTED:
            support_strength = min(1.0, (similarity + nli_conf) / 1.6)
            return clamp01(self._cr_supported_max * (1.0 - 0.9 * support_strength))
        if status == ClaimStatus.NEUTRAL:
            # Evidence retrieved but ambiguous: more risk when that evidence
            # is weak (we really don't know) than when it is rich but silent.
            return clamp01(self._cr_neutral * (0.6 + 0.4 * (1.0 - evidence_strength)))
        # NO_EVIDENCE -> the flat "we could not verify" penalty.
        return clamp01(self._cr_no_evidence)

    def _text_criticality(self, claim: str) -> float:
        score = 0.0
        lowered = (claim or "").lower()
        if _MONEY_RE.search(claim or "") or _BIG_NUMBER_RE.search(claim or ""):
            score += 0.55
        if any(re.search(rf"\b{re.escape(v)}", lowered) for v in self._action_verbs):
            score += 0.35
        if _ENTITY_RE.search(claim or ""):
            score += 0.20
        return clamp01(score)

    def _best_nli(
        self, claim: str, evidence: list[EvidenceMatch], watch: Stopwatch
    ) -> tuple[NLILabel, float]:
        """
        Run NLI for the claim against each retrieved evidence chunk and
        keep the most decisive verdict: a threshold-passing contradiction
        wins, then a threshold-passing entailment, otherwise the
        highest-confidence remaining label.
        """
        contradiction: tuple[NLILabel, float] | None = None
        entailment: tuple[NLILabel, float] | None = None
        fallback: tuple[NLILabel, float] = (NLILabel.NEUTRAL, 0.0)

        # Only adjudicate against evidence chunks whose retrieval similarity
        # is competitive with the best match. A verdict from a weakly-related
        # chunk (e.g. a numeric conflict against an unrelated sentence) must
        # not override the verdict from the genuinely most-relevant chunk.
        best_similarity = evidence[0].similarity if evidence else 0.0
        similarity_gate = max(self.retrieval_threshold, 0.6 * best_similarity)

        with watch.stage("nli"):
            for match in evidence:
                if match.similarity < similarity_gate:
                    continue
                label, confidence = self.nli.predict(match.text, claim)
                if (
                    label == NLILabel.CONTRADICTION
                    and confidence >= self.contradiction_threshold
                ):
                    if contradiction is None or confidence > contradiction[1]:
                        contradiction = (label, confidence)
                elif (
                    label == NLILabel.ENTAILMENT
                    and confidence >= self.entailment_threshold
                ):
                    if entailment is None or confidence > entailment[1]:
                        entailment = (label, confidence)
                elif confidence > fallback[1]:
                    fallback = (label, confidence)

        if contradiction is not None:
            return contradiction
        if entailment is not None:
            return entailment
        return fallback

    def _claim_status(self, label: NLILabel, confidence: float) -> ClaimStatus:
        if label == NLILabel.ENTAILMENT and confidence >= self.entailment_threshold:
            return ClaimStatus.SUPPORTED
        if label == NLILabel.CONTRADICTION and confidence >= self.contradiction_threshold:
            return ClaimStatus.CONTRADICTED
        return ClaimStatus.NEUTRAL

    @staticmethod
    def _claim_explanation(
        status: ClaimStatus,
        label: NLILabel,
        confidence: float,
        similarity: float,
    ) -> str:
        if status == ClaimStatus.SUPPORTED:
            return (
                f"Retrieved evidence entails this claim (NLI confidence "
                f"{confidence:.2f}, retrieval similarity {similarity:.2f})."
            )
        if status == ClaimStatus.CONTRADICTED:
            return (
                f"Retrieved evidence contradicts this claim (NLI confidence "
                f"{confidence:.2f}, retrieval similarity {similarity:.2f})."
            )
        return (
            "Relevant evidence was retrieved but does not clearly support or "
            f"contradict this claim (closest NLI label {label.value}, "
            f"confidence {confidence:.2f}); reported as unverified."
        )

    # ------------------------------------------------------------------
    # response-level aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        claim_results: list[ClaimResult],
        evidence_available: bool,
        watch: Stopwatch,
    ) -> PerformanceResult:
        latency = LatencyBreakdown(
            claim_extraction_ms=watch.get("claim_extraction"),
            chunking_ms=watch.get("chunking"),
            embedding_ms=watch.get("embedding"),
            retrieval_ms=watch.get("retrieval"),
            nli_ms=watch.get("nli"),
            total_ms=watch.total_ms(),
        )

        total = len(claim_results)
        if total == 0:
            return PerformanceResult(
                status=ResponseStatus.UNVERIFIED,
                grounding_score=None,
                performance_risk=0.10,
                confidence=0.55,
                uncertainty=0.45,
                evidence_quality=0.0,
                verification_confidence=0.0,
                claim_results=[],
                evidence_available=evidence_available,
                method=self._method,
                explanation=(
                    "The response makes no verifiable factual claims, so there "
                    "is nothing to ground against. Reported as UNVERIFIED with "
                    "low performance risk."
                ),
                latency=latency,
            )

        supported = sum(1 for c in claim_results if c.status == ClaimStatus.SUPPORTED)
        contradicted = sum(
            1 for c in claim_results if c.status == ClaimStatus.CONTRADICTED
        )
        adjudicated = supported + contradicted
        unverified = total - adjudicated

        verification_confidence = adjudicated / total
        evidence_quality = fmean(c.evidence_strength for c in claim_results)

        grounding_score: float | None
        if adjudicated == 0:
            grounding_score = None
        else:
            grounding_score = supported / adjudicated
            if contradicted > 0:
                ceiling = max(
                    self.ceiling_floor,
                    self.ceiling_base - self.ceiling_step * (contradicted - 1),
                )
                grounding_score = min(grounding_score, ceiling)

        performance_risk = self._response_risk(claim_results, contradicted, supported, total)
        status = self._response_status(total, supported, contradicted)
        confidence = self._detector_confidence(
            claim_results, verification_confidence, adjudicated, evidence_quality
        )
        explanation = self._response_explanation(
            status, total, supported, contradicted, unverified, evidence_available,
            performance_risk, claim_results,
        )

        return PerformanceResult(
            status=status,
            grounding_score=(
                None if grounding_score is None else round(clamp01(grounding_score), 4)
            ),
            performance_risk=round(clamp01(performance_risk), 4),
            confidence=round(clamp01(confidence), 4),
            uncertainty=round(clamp01(1.0 - confidence), 4),
            evidence_quality=round(clamp01(evidence_quality), 4),
            verification_confidence=round(clamp01(verification_confidence), 4),
            claim_results=claim_results,
            evidence_available=evidence_available,
            method=self._method,
            explanation=explanation,
            latency=latency,
        )

    def _response_risk(
        self,
        claim_results: list[ClaimResult],
        contradicted: int,
        supported: int,
        total: int,
    ) -> float:
        # Evidence-weighted: each claim's risk contributes in proportion to
        # how much that claim matters (its text criticality).
        weights = [1.0 + self._text_crit_weight * c.text_criticality for c in claim_results]
        weighted_mean = sum(w * c.claim_risk for w, c in zip(weights, claim_results)) / (
            sum(weights) or 1.0
        )

        if contradicted > 0:
            # A confirmed contradiction floors the response risk regardless
            # of how many benign claims dilute the mean (preserves the
            # "one hallucination is enough" semantics).
            contradiction_floor = min(0.97, 0.62 + 0.12 * contradicted)
            return clamp01(max(weighted_mean, contradiction_floor))
        if supported == total:
            return clamp01(weighted_mean)
        # UNVERIFIED regime — moderate risk, never the risk of a confirmed
        # hallucination. The weighted mean already encodes this.
        return clamp01(max(weighted_mean, 0.05))

    @staticmethod
    def _response_status(
        total: int, supported: int, contradicted: int
    ) -> ResponseStatus:
        if contradicted > 0 and supported == 0:
            return ResponseStatus.CONTRADICTED
        if contradicted > 0:
            return ResponseStatus.PARTIALLY_SUPPORTED
        if supported == total:
            return ResponseStatus.SUPPORTED
        if supported > 0:
            return ResponseStatus.PARTIALLY_SUPPORTED
        return ResponseStatus.UNVERIFIED

    @staticmethod
    def _detector_confidence(
        claim_results: list[ClaimResult],
        verification_confidence: float,
        adjudicated: int,
        evidence_quality: float,
    ) -> float:
        """
        Confidence in the *assessment*, explicitly distinct from the risk.

        Driven by: how much of the response we could adjudicate, how sure
        the NLI verdicts were, and how strong the underlying evidence was.
        """
        if adjudicated == 0:
            # We are moderately sure we *could not* verify — but that says
            # nothing about whether the response is true. Scale gently with
            # evidence quality (retrieved-but-silent > nothing retrieved).
            return clamp01(0.40 + 0.20 * evidence_quality)
        nli_confidences = [
            c.nli_confidence
            for c in claim_results
            if c.nli_confidence is not None
            and c.status in (ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED)
        ]
        mean_nli = fmean(nli_confidences) if nli_confidences else 0.5
        base = (
            0.35
            + 0.25 * verification_confidence
            + 0.20 * mean_nli
            + 0.20 * evidence_quality
        )
        # Lexical system: never claim very high self-confidence.
        return min(0.9, base)

    @staticmethod
    def _response_explanation(
        status: ResponseStatus,
        total: int,
        supported: int,
        contradicted: int,
        unverified: int,
        evidence_available: bool,
        performance_risk: float = 0.0,
        claim_results: list[ClaimResult] | None = None,
    ) -> str:
        parts = [
            f"{total} claim(s) extracted: {supported} supported, "
            f"{contradicted} contradicted, {unverified} unverified."
        ]
        if claim_results:
            riskiest = max(claim_results, key=lambda c: c.claim_risk)
            if riskiest.claim_risk >= 0.5:
                parts.append(
                    f"Riskiest claim (risk {riskiest.claim_risk:.2f}, evidence "
                    f"strength {riskiest.evidence_strength:.2f}): "
                    f"\"{riskiest.claim[:90]}\"."
                )
        if not evidence_available:
            parts.append(
                "No usable context/evidence was available, so unsupported "
                "claims are reported as UNVERIFIED rather than hallucinated."
            )
        if status == ResponseStatus.SUPPORTED:
            parts.append("All extracted claims are supported by the supplied evidence.")
        elif status == ResponseStatus.CONTRADICTED:
            parts.append(
                "One or more claims directly contradict the supplied evidence."
            )
        elif status == ResponseStatus.PARTIALLY_SUPPORTED:
            parts.append(
                "Some claims are supported while others are contradicted or "
                "could not be verified."
            )
        else:  # UNVERIFIED
            parts.append(
                "The available evidence was insufficient to confirm or refute "
                "the response's claims."
            )
        return " ".join(parts)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _resolve_method(self) -> str:
        if isinstance(self.embeddings, TfidfEmbeddingBackend):
            embed_method = "tfidf"
        else:
            embed_method = type(self.embeddings).__name__
        if isinstance(self.nli, CoverageNLIBackend):
            nli_method = "lexical_nli"
        elif isinstance(self.nli, LexicalNLIBackend):
            nli_method = "lexical_nli"
        else:
            nli_method = type(self.nli).__name__
        return f"{embed_method}+{nli_method}"


def detect_performance(
    interaction: Interaction,
    config: dict[str, Any] | None = None,
) -> PerformanceResult:
    """Convenience one-shot wrapper around :class:`PerformanceDetector`."""
    return PerformanceDetector(config=config).detect(interaction)
