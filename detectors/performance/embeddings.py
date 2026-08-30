"""
Lightweight similarity backend for the Performance Detector.

Given a claim and a set of candidate evidence chunks, this module's
only job is to produce similarity scores between them. It does not
extract claims, chunk context, run NLI, classify hallucinations,
calculate risk, or touch ground truth — those are separate layers.

The default backend uses TF-IDF + cosine similarity (via scikit-learn)
so the system runs on a laptop without downloading a large model.
The abstraction (``EmbeddingBackend``) is deliberately backend-agnostic
so a transformer-based backend can be added later without changing
callers.

Deterministic: no randomness, no wall-clock time, no network calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingBackend(ABC):
    """
    Abstraction the rest of ControlPlane depends on, instead of
    depending directly on TfidfVectorizer (or any future backend).
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> Any:
        """
        Embed ``texts`` and return an opaque, backend-specific
        representation (e.g. a sparse TF-IDF matrix). Callers should
        not assume anything about the returned object's shape or type.
        """
        raise NotImplementedError

    @abstractmethod
    def similarity(self, query: str, documents: list[str]) -> list[float]:
        """
        Return one similarity score per document in ``documents``, in
        the same order, each in [0, 1].

        Must return ``[]`` if ``documents`` is empty, and a list of
        zeros (one per document) if ``query`` is empty or a shared
        vocabulary cannot be built.
        """
        raise NotImplementedError


class TfidfEmbeddingBackend(EmbeddingBackend):
    """
    Default system backend: TF-IDF vectors + cosine similarity.

    This is a lexical similarity measure, not semantic truth
    verification — two claims can be lexically similar while being
    factually opposite (e.g. "returns allowed in 30 days" vs. "returns
    NOT allowed in 30 days"). Determining agreement/contradiction is
    the job of the later NLI layer, not this one; this module only
    finds text that is *potentially* relevant.

    No model is loaded in the constructor. Each call fits a fresh,
    small TF-IDF vectorizer over just the texts involved, which keeps
    the backend simple, deterministic, and dependency-light.
    """

    def __init__(self, max_features: int | None = None) -> None:
        self.max_features = max_features

    def _make_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(max_features=self.max_features)

    def embed(self, texts: list[str]) -> Any:
        """
        Fit a TF-IDF vectorizer over ``texts`` and return the resulting
        sparse matrix (one row per text). Returns ``None`` if ``texts``
        is empty or no usable vocabulary can be built (e.g. every text
        is empty or contains only stop words).
        """
        if not texts:
            return None

        try:
            matrix = self._make_vectorizer().fit_transform(texts)
        except ValueError:
            # Empty vocabulary — e.g. all texts are blank/whitespace.
            return None

        return matrix

    def similarity(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        corpus = [query] + list(documents)

        try:
            matrix = self._make_vectorizer().fit_transform(corpus)
        except ValueError:
            # No usable vocabulary at all (e.g. every text is blank) —
            # fail gracefully rather than crashing the caller.
            return [0.0] * len(documents)

        query_vector = matrix[0:1]
        document_vectors = matrix[1:]

        scores = cosine_similarity(query_vector, document_vectors)[0]

        # Cosine similarity on non-negative TF-IDF vectors is already
        # in [0, 1]; clip defensively against floating-point drift.
        return [min(1.0, max(0.0, float(score))) for score in scores]


def rank_documents(
    query: str,
    documents: list[str],
    scores: list[float],
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """
    Rank documents by similarity score, preserving original indexes.

    Sorts by ``scores`` descending, breaking ties deterministically by
    ascending original index. Returns at most ``top_k`` results as
    ``(original_index, score)`` pairs.

    Any index beyond the shorter of ``documents``/``scores`` is
    ignored rather than raising. Returns ``[]`` if there are no
    documents. ``query`` is accepted for interface symmetry with
    ``EmbeddingBackend.similarity`` and possible future use, but the
    ranking itself depends only on ``scores``.
    """
    del query  # Not used for ranking; kept for interface symmetry.

    if not documents or not scores:
        return []

    usable_count = min(len(documents), len(scores))
    indexed_scores = [(index, scores[index]) for index in range(usable_count)]
    indexed_scores.sort(key=lambda pair: (-pair[1], pair[0]))

    if top_k <= 0:
        return []

    return indexed_scores[:top_k]