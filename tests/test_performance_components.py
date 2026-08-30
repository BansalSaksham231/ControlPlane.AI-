"""
Component-level tests for the Performance Detector's three
independent building blocks: chunker, embeddings, and NLI.

These validate each component in isolation (plus one manual
integration smoke test wiring them together) before the integrated
detector is built. No production code is modified here.
"""

from __future__ import annotations

import pytest

from detectors.performance.chunker import chunk_context, extract_claims
from detectors.performance.embeddings import TfidfEmbeddingBackend, rank_documents
from detectors.performance.nli import LexicalNLIBackend
from detectors.performance.schemas import NLILabel


# ==================================================
# CHUNKER TESTS
# ==================================================


def test_extract_claims_splits_two_sentences():
    claims = extract_claims(
        "Refunds are available within 30 days. Final-sale items cannot be returned."
    )
    assert len(claims) == 2


def test_extract_claims_filters_by_min_tokens():
    claims = extract_claims("Okay. Refunds are available.", min_tokens=2)
    assert "Okay." not in claims


def test_extract_claims_empty_response():
    assert extract_claims("") == []


def test_extract_claims_whitespace_response():
    assert extract_claims("   ") == []


def test_chunk_context_splits_two_chunks():
    chunks = chunk_context(
        "Refunds are available within 30 days. "
        "Final-sale items cannot be returned."
    )
    assert len(chunks) == 2


def test_chunk_context_long_sentence_respects_max_chars():
    long_text = "word " * 200  # a single long "sentence" with no terminator
    chunks = chunk_context(long_text, max_chars=50)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= 50


# ==================================================
# EMBEDDING TESTS
# ==================================================


def test_similarity_empty_documents_returns_empty_list():
    backend = TfidfEmbeddingBackend()
    assert backend.similarity("refund", []) == []


def test_similarity_empty_query_returns_zeros():
    backend = TfidfEmbeddingBackend()
    scores = backend.similarity("", ["refund", "account"])
    assert scores == [0.0, 0.0]


def test_similarity_ranks_relevant_document_higher():
    backend = TfidfEmbeddingBackend()
    scores = backend.similarity(
        "refund within 30 days",
        [
            "Customers may request a refund within 30 days.",
            "Employees receive annual leave.",
        ],
    )
    assert len(scores) == 2
    for score in scores:
        assert 0.0 <= score <= 1.0
    assert scores[0] > scores[1]


def test_similarity_is_deterministic():
    backend = TfidfEmbeddingBackend()
    documents = [
        "Customers may request a refund within 30 days.",
        "Employees receive annual leave.",
    ]
    first = backend.similarity("refund within 30 days", documents)
    second = backend.similarity("refund within 30 days", documents)
    assert first == second


# ==================================================
# RANKING TESTS
# ==================================================


def test_rank_documents_sorts_by_score_descending():
    result = rank_documents("refund", ["A", "B", "C"], [0.2, 0.9, 0.5], top_k=2)
    assert result == [(1, 0.9), (2, 0.5)]


def test_rank_documents_breaks_ties_by_index():
    result = rank_documents("refund", ["A", "B", "C"], [0.5, 0.5, 0.2], top_k=2)
    assert result[:2] == [(0, 0.5), (1, 0.5)]


def test_rank_documents_top_k_non_positive_returns_empty():
    result = rank_documents("refund", ["A", "B", "C"], [0.2, 0.9, 0.5], top_k=0)
    assert result == []


# ==================================================
# NLI TESTS
# ==================================================


def test_nli_exact_entailment():
    backend = LexicalNLIBackend()
    label, confidence = backend.predict(
        "Refunds are available within 30 days.",
        "Refunds are available within 30 days.",
    )
    assert label == NLILabel.ENTAILMENT
    assert 0.0 <= confidence <= 1.0


def test_nli_paraphrase_entailment():
    backend = LexicalNLIBackend()
    label, confidence = backend.predict(
        "Customers may request refunds within 30 days.",
        "Customers can request refunds within 30 days.",
    )
    assert label == NLILabel.ENTAILMENT
    assert 0.0 <= confidence <= 1.0


def test_nli_negation_contradiction():
    backend = LexicalNLIBackend()
    label, confidence = backend.predict(
        "Refunds are available within 30 days.",
        "Refunds are not available within 30 days.",
    )
    assert label == NLILabel.CONTRADICTION
    assert 0.0 <= confidence <= 1.0


def test_nli_numeric_contradiction_30_vs_90_days():
    backend = LexicalNLIBackend()
    label, confidence = backend.predict(
        "Refunds are available within 30 days.",
        "Refunds are available within 90 days.",
    )
    assert label == NLILabel.CONTRADICTION
    assert 0.0 <= confidence <= 1.0


def test_nli_neutral_unrelated_topics():
    backend = LexicalNLIBackend()
    label, confidence = backend.predict(
        "Refunds are available within 30 days.",
        "Employees receive 20 days of annual leave.",
    )
    assert label == NLILabel.NEUTRAL
    assert 0.0 <= confidence <= 1.0


def test_nli_numeric_contradiction_7_vs_10_days():
    backend = LexicalNLIBackend()
    label, confidence = backend.predict(
        "Refunds are processed within 7 business days.",
        "Refunds are processed within 10 business days.",
    )
    assert label == NLILabel.CONTRADICTION
    assert 0.0 <= confidence <= 1.0


def test_nli_predict_batch():
    backend = LexicalNLIBackend()
    premise = "Refunds are available within 30 days."
    hypotheses = [
        "Refunds are available within 30 days.",
        "Refunds are not available within 30 days.",
        "Employees receive 20 days of annual leave.",
    ]
    results = backend.predict_batch(premise, hypotheses)

    assert len(results) == 3
    for label, confidence in results:
        assert isinstance(label, NLILabel)
        assert 0.0 <= confidence <= 1.0

    # Order matches the input hypotheses.
    assert results[0][0] == NLILabel.ENTAILMENT
    assert results[1][0] == NLILabel.CONTRADICTION
    assert results[2][0] == NLILabel.NEUTRAL


# ==================================================
# INTEGRATION SMOKE TEST
# ==================================================


def test_chunker_embeddings_nli_integration_smoke():
    context = (
        "Customers may request refunds within 30 days. "
        "Refunds are processed within 7 business days."
    )
    response = "Customers can request refunds within 90 days."

    claims = extract_claims(response)
    assert len(claims) >= 1
    claim = claims[0]

    chunks = chunk_context(context)
    assert len(chunks) > 0

    backend = TfidfEmbeddingBackend()
    scores = backend.similarity(claim, chunks)
    assert len(scores) == len(chunks)
    for score in scores:
        assert 0.0 <= score <= 1.0

    ranked = rank_documents(claim, chunks, scores, top_k=1)
    assert len(ranked) == 1
    best_index, _best_score = ranked[0]
    best_evidence = chunks[best_index]

    nli_backend = LexicalNLIBackend()
    label, confidence = nli_backend.predict(best_evidence, claim)

    assert isinstance(label, NLILabel)
    assert 0.0 <= confidence <= 1.0