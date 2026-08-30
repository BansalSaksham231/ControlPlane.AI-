"""
Integrated Performance Detector tests.

Covers the full response -> claims -> retrieval -> NLI -> aggregation
pipeline and, most importantly, the semantic rule that UNVERIFIED is
never conflated with CONTRADICTED.
"""

from __future__ import annotations

import pytest

from data.schemas import ActionType, Application, Interaction, ModelName, UserType
from detectors.performance.detector import PerformanceDetector, detect_performance
from detectors.performance.schemas import (
    ClaimStatus,
    LatencyBreakdown,
    PerformanceResult,
    ResponseStatus,
)

REFUND_CONTEXT = (
    "Company policy allows customers to request a refund within 30 business "
    "days of purchase, provided the item is unused and in its original "
    "packaging. Refunds are processed within 7 business days."
)


@pytest.fixture(scope="module")
def detector() -> PerformanceDetector:
    return PerformanceDetector()


CLEAN_REFUND_RESPONSE = (
    "You are eligible for a refund within 30 business days of your purchase, "
    "as long as the item is unused and in its original packaging."
)


def _interaction(response: str, context: str) -> Interaction:
    return Interaction(
        interaction_id="INT-TEST-1",
        timestamp="2026-08-21T12:00:00",
        application=Application.CUSTOMER_SUPPORT,
        user_type=UserType.EXTERNAL_CUSTOMER,
        model=ModelName.GPT_4O_MINI,
        session_id="SESSION-1",
        prompt="What is the refund policy?",
        context=context,
        response=response,
        tokens_in=40,
        tokens_out=30,
        latency_ms=300.0,
        tool_calls=0,
        retry_count=0,
        action_type=ActionType.INFORMATION,
        action_amount_inr=0.0,
        affected_entities=1,
    )


# 1. supported response
def test_supported_response(detector):
    result = detector.detect(
        CLEAN_REFUND_RESPONSE,
        REFUND_CONTEXT,
    )
    assert result.status == ResponseStatus.SUPPORTED
    assert result.grounding_score == 1.0
    assert result.performance_risk < 0.2
    assert result.evidence_available is True


# 2. contradiction
def test_contradicted_response(detector):
    result = detector.detect(
        "You are eligible for a refund within 90 business days of your "
        "purchase, and the item condition does not matter.",
        REFUND_CONTEXT,
    )
    assert result.status in (ResponseStatus.CONTRADICTED, ResponseStatus.PARTIALLY_SUPPORTED)
    assert any(c.status == ClaimStatus.CONTRADICTED for c in result.claim_results)
    assert result.performance_risk >= 0.6


# 3. numeric contradiction
def test_numeric_contradiction(detector):
    result = detector.detect(
        "You can request a refund within 90 days of purchase.",
        REFUND_CONTEXT,
    )
    assert any(c.status == ClaimStatus.CONTRADICTED for c in result.claim_results)
    assert result.performance_risk >= 0.6


# 4. empty context -> unverified, NOT contradicted
def test_empty_context_is_unverified_not_hallucinated(detector):
    result = detector.detect("You can request a refund within 30 days.", "")
    assert result.status == ResponseStatus.UNVERIFIED
    assert result.evidence_available is False
    assert all(c.status != ClaimStatus.CONTRADICTED for c in result.claim_results)
    assert result.grounding_score is None
    # Moderate risk, never the risk of a confirmed contradiction.
    assert 0.3 <= result.performance_risk <= 0.7


# 5. low similarity -> unverified
def test_low_similarity_is_unverified(detector):
    result = detector.detect(
        "The quarterly financial results exceeded analyst expectations.",
        REFUND_CONTEXT,
    )
    assert result.status == ResponseStatus.UNVERIFIED
    assert all(c.status == ClaimStatus.NO_EVIDENCE for c in result.claim_results)
    assert all(c.nli_label is None for c in result.claim_results)


# 6. mixed claims (one supported, one contradicted)
def test_mixed_claims(detector):
    result = detector.detect(
        "Refunds are processed within 7 business days. "
        "You are also eligible for a refund within 90 business days of purchase.",
        REFUND_CONTEXT,
    )
    statuses = {c.status for c in result.claim_results}
    assert ClaimStatus.SUPPORTED in statuses
    assert ClaimStatus.CONTRADICTED in statuses
    assert result.status == ResponseStatus.PARTIALLY_SUPPORTED
    assert result.grounding_score is not None and result.grounding_score < 1.0


# 7. no claims
def test_no_claims(detector):
    result = detector.detect("Okay.", REFUND_CONTEXT)
    assert result.claim_results == []
    assert result.status == ResponseStatus.UNVERIFIED
    assert result.performance_risk < 0.3
    assert result.verification_confidence == 0.0


# 8. top-k behaviour
def test_top_k_evidence_respected():
    long_context = ". ".join(f"Sentence number {i} about refunds and returns" for i in range(20))
    detector = PerformanceDetector()
    result = detector.detect("Refunds and returns are handled here.", long_context)
    for claim_result in result.claim_results:
        assert len(claim_result.top_evidence) <= detector.top_k
        ranks = [e.rank for e in claim_result.top_evidence]
        assert ranks == sorted(ranks)


# 9. latency contract
def test_latency_contract(detector):
    result = detector.detect("You can request a refund within 30 days.", REFUND_CONTEXT)
    assert isinstance(result.latency, LatencyBreakdown)
    assert result.latency.total_ms >= 0.0
    for stage in ("claim_extraction_ms", "chunking_ms", "embedding_ms", "retrieval_ms", "nli_ms"):
        assert getattr(result.latency, stage) >= 0.0


# 10. method reporting
def test_method_reporting(detector):
    result = detector.detect("You can request a refund within 30 days.", REFUND_CONTEXT)
    assert result.method == "tfidf+lexical_nli"


# 11. deterministic output
def test_deterministic_output():
    d1 = PerformanceDetector()
    d2 = PerformanceDetector()
    response = "You can request a refund within 90 days of purchase."
    r1 = d1.detect(response, REFUND_CONTEXT)
    r2 = d2.detect(response, REFUND_CONTEXT)

    def _stable(result: PerformanceResult) -> dict:
        dumped = result.model_dump()
        dumped.pop("latency")
        return dumped

    assert _stable(r1) == _stable(r2)


# 12. explanations
def test_explanations_are_human_readable(detector):
    result = detector.detect(
        "You can request a refund within 90 days of purchase.", REFUND_CONTEXT
    )
    assert len(result.explanation) > 20
    assert "threshold" not in result.explanation.lower() or "similarity" in result.explanation.lower()
    for claim_result in result.claim_results:
        assert len(claim_result.explanation) > 10


def test_accepts_interaction_object(detector):
    interaction = _interaction(CLEAN_REFUND_RESPONSE, REFUND_CONTEXT)
    result = detector.detect(interaction)
    assert result.status == ResponseStatus.SUPPORTED


def test_convenience_wrapper():
    interaction = _interaction(CLEAN_REFUND_RESPONSE, REFUND_CONTEXT)
    result = detect_performance(interaction)
    assert isinstance(result, PerformanceResult)


def test_no_ground_truth_attributes_consumed():
    """The detector must work with only production-visible fields."""
    interaction = _interaction(CLEAN_REFUND_RESPONSE, REFUND_CONTEXT)
    # Interaction has no ground-truth attributes at all; assert that holds.
    for attr in ("ground_truth_hallucination", "ground_truth_performance_risk", "expected_decision"):
        assert not hasattr(interaction, attr)
    result = detect_performance(interaction)
    assert isinstance(result, PerformanceResult)
