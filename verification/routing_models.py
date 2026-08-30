"""
Routing models for the tiered verification cascade.
===================================================

This module owns everything about *scoring how much a DEEP verification pass
matters* for a given interaction, kept separate from the routing orchestration
in :mod:`verification.cascade_router` so the model can be swapped without
touching the cascade.

    extract_features(...)              cheap, deterministic feature bag
    ComplexityClassifier               the interface the cascade depends on
    HeuristicComplexityClassifier      zero-artifact linear-logistic fallback
    MatrixFactorizationRouter          RouteLLM-style low-rank router
    fit_matrix_factorization(...)      pure-Python offline trainer
    select_cost_optimal_threshold(...) safety-first threshold calibration

Nothing here downloads a model, opens a socket, or uses randomness. The MF
router's parameters are a small fitted artifact (a few hundred floats), exactly
as RouteLLM ships its matrix-factorisation router — training happens offline and
the factors are reloaded here.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol, runtime_checkable

__all__ = [
    "RoutingInput",
    "FEATURE_NAMES",
    "extract_features",
    "ComplexityPrediction",
    "ComplexityClassifier",
    "HeuristicComplexityClassifier",
    "EmbeddingBackend",
    "HashingEmbeddingBackend",
    "MFParameters",
    "MatrixFactorizationRouter",
    "EmbeddingRoutingSample",
    "fit_matrix_factorization",
    "fit_matrix_factorization_embeddings",
    "build_default_mf_router",
    "build_default_embedding_mf_router",
    "RoutingSample",
    "ThresholdCalibration",
    "select_cost_optimal_threshold",
]


# --------------------------------------------------------------------------- #
# Structural input contract (avoids a circular import with cascade_router)
# --------------------------------------------------------------------------- #
@runtime_checkable
class RoutingInput(Protocol):
    """The fields the model layer reads. ``Interaction`` satisfies this."""

    prompt: str
    response: str
    context: str

    @property
    def has_evidence(self) -> bool: ...
    @property
    def is_action(self) -> bool: ...
    @property
    def estimated_response_tokens(self) -> int: ...


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
FEATURE_NAMES: tuple[str, ...] = (
    "response_length_ratio",
    "digit_density",
    "entity_density",
    "hedge_density",
    "low_context_overlap",
    "claim_count",
    "question_form",
    "action_flag",
    "no_evidence",
)

_WORD = re.compile(r"[A-Za-z']+")
_CAP_RUN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_SENTENCE_SPLIT = re.compile(r"[.!?;]\s+|\n+")
_HEDGE_TOKENS = frozenset({
    "might", "maybe", "perhaps", "possibly", "probably", "likely", "unsure",
    "think", "believe", "approximately", "around", "about", "roughly", "seems",
    "appears", "could", "should", "generally", "typically", "assume", "guess",
})
_QUESTION_WORDS = frozenset(
    {"who", "what", "when", "where", "why", "how", "which", "is", "are", "does", "can"}
)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _tokenset(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def estimate_tokens(text: str) -> int:
    """Whitespace token estimate — deterministic, dependency-free."""
    return len(text.split())


def extract_features(item: RoutingInput) -> dict[str, float]:
    """
    Turn an interaction (or a single incremental turn) into a bounded [0, 1]
    feature bag. Every feature is monotonic in "a DEEP pass is more likely to
    change the decision": more numbers, more named entities, more hedging, less
    overlap with the retrieved evidence, an action in play, no evidence at all.
    """
    response, context, prompt = item.response, item.context, item.prompt
    response_tokens = max(1, estimate_tokens(response))
    context_tokens = max(1, estimate_tokens(context))
    resp_words = _WORD.findall(response)

    digit_chars = sum(c.isdigit() for c in response)
    entities = len(_CAP_RUN.findall(response))
    hedges = sum(1 for w in resp_words if w.lower() in _HEDGE_TOKENS)
    claims = max(1, len(_SENTENCE_SPLIT.split(response.strip())))

    resp_set, ctx_set = _tokenset(response), _tokenset(context)
    overlap = (
        len(resp_set & ctx_set) / len(resp_set | ctx_set)
        if resp_set and ctx_set
        else 0.0
    )
    first_word = (prompt.strip().split(" ") or [""])[0].lower()

    return {
        "response_length_ratio": _clamp01(response_tokens / (context_tokens + 8)),
        "digit_density": _clamp01(digit_chars / max(len(response), 1) * 12.0),
        "entity_density": _clamp01(entities / (response_tokens / 20 + 1)),
        "hedge_density": _clamp01(hedges / (len(resp_words) / 15 + 1)),
        "low_context_overlap": _clamp01(1.0 - overlap) if item.has_evidence else 1.0,
        "claim_count": _clamp01(claims / 8.0),
        "question_form": 1.0
        if first_word in _QUESTION_WORDS or prompt.rstrip().endswith("?")
        else 0.0,
        "action_flag": 1.0 if item.is_action else 0.0,
        "no_evidence": 0.0 if item.has_evidence else 1.0,
    }


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# Embedding backend — the seam where a real sentence encoder plugs in
# --------------------------------------------------------------------------- #
class EmbeddingBackend(ABC):
    """
    Maps a prompt string to a fixed-dimension dense vector.

    In an enterprise deployment this wraps a real sentence encoder — a
    quantised ``sentence-transformers`` MiniLM, an in-house distilled encoder,
    or a hosted embeddings endpoint — loaded once at process start. The MF
    router depends only on this interface, so swapping the encoder never
    touches the router or the cascade.
    """

    dim: int = 0
    name: str = "abstract-embedding-backend"

    @abstractmethod
    def embed(self, text: str) -> tuple[float, ...]:
        raise NotImplementedError


class HashingEmbeddingBackend(EmbeddingBackend):
    """
    Deterministic, offline stand-in for a learned sentence encoder using the
    hashing trick over word + character-trigram tokens, then L2-normalised.

    It is not semantic — it captures lexical surface only — but it is stable,
    dependency-free and fast (O(len(text))), which is all the router needs to
    exercise the full matrix-factorisation code path in tests and demos. Point
    ``MatrixFactorizationRouter`` at a real :class:`EmbeddingBackend` for
    production.
    """

    name = "hashing-embedding (offline stand-in)"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self.dim
        lowered = text.lower()
        tokens = _WORD.findall(lowered)
        grams = [lowered[i : i + 3] for i in range(max(0, len(lowered) - 2))]
        for token in (*tokens, *grams):
            h = _stable_hash(token)
            idx = h % self.dim
            sign = 1.0 if (h >> 32) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return tuple(v / norm for v in vec)


def _stable_hash(token: str) -> int:
    """FNV-1a — deterministic across processes (unlike ``hash``)."""
    h = 0xCBF29CE484222325
    for byte in token.encode("utf-8"):
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


# --------------------------------------------------------------------------- #
# Classifier interface + predictions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ComplexityPrediction:
    """Output of a :class:`ComplexityClassifier`."""

    score: float                       # predicted_complexity_score in [0, 1]
    model_name: str
    features: Mapping[str, float]


class ComplexityClassifier(ABC):
    """
    Tier-1 routing model interface.

    Its single job: predict cheaply how likely a full DEEP verification pass is
    to *change the downstream decision* for this interaction. Implementations
    must be deterministic and stay well inside the classifier latency budget
    (target ~5 ms on CPU).
    """

    model_name: str = "abstract-complexity-classifier"

    @abstractmethod
    def predict_complexity(self, item: RoutingInput) -> ComplexityPrediction:
        raise NotImplementedError

    def predict_from_features(self, features: Mapping[str, float]) -> float:
        """Score a pre-extracted feature bag (used by contextual-snapshot routing)."""
        raise NotImplementedError


# Illustrative linear weights for the zero-artifact fallback. Not hand-tuned
# rules — a monotonic surface good enough for tests, demos and threshold
# calibration when no fitted MF artifact is available.
_HEURISTIC_WEIGHTS: Mapping[str, float] = {
    "response_length_ratio": 1.30,
    "digit_density": 1.90,
    "entity_density": 1.55,
    "hedge_density": 2.10,
    "low_context_overlap": 2.35,
    "claim_count": 0.28,
    "question_form": 0.45,
    "action_flag": 1.20,
    "no_evidence": 1.40,
}
_HEURISTIC_BIAS = -1.15


class HeuristicComplexityClassifier(ComplexityClassifier):
    """Deterministic linear-logistic fallback. No fitted artifact required."""

    model_name = "heuristic-complexity-v1"

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        bias: float = _HEURISTIC_BIAS,
    ) -> None:
        self._w = {**_HEURISTIC_WEIGHTS, **(weights or {})}
        self._b = bias

    def predict_complexity(self, item: RoutingInput) -> ComplexityPrediction:
        features = extract_features(item)
        return ComplexityPrediction(
            score=round(self.predict_from_features(features), 6),
            model_name=self.model_name,
            features={k: round(v, 4) for k, v in features.items()},
        )

    def predict_from_features(self, features: Mapping[str, float]) -> float:
        z = self._b + sum(
            self._w.get(name, 0.0) * value for name, value in features.items()
        )
        return _sigmoid(z)


# --------------------------------------------------------------------------- #
# Matrix-factorisation router (RouteLLM-style)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MFParameters:
    """
    Fitted low-rank router parameters. Supports two scoring paths:

    * **feature-bag** (``projection`` ``d x k`` + ``verifier_latent`` ``k``) —
      logit = ``(features @ projection) . verifier_latent + bias``. Zero external
      dependencies; used when no embedding backend is available.
    * **embedding** (``query_projection`` ``e x k`` + ``model_preference`` with a
      latent vector per routing arm) — the RouteLLM formulation. The prompt is
      embedded (``e``-dim), projected to the shared ``k``-dim latent space, then
      scored as a *preference* between the two arms:
      logit = ``q . (model_preference["DEEP"] - model_preference["FAST"]) + bias``.
      ``model_preference`` is the "user/model preference matrix" — here the two
      users are the strong verifier (DEEP) and the weak one (FAST).
    """

    feature_names: tuple[str, ...] = ()
    projection: tuple[tuple[float, ...], ...] = ()
    verifier_latent: tuple[float, ...] = ()
    bias: float = 0.0
    latent_dim: int = 0
    training_samples: int = 0
    # embedding path (optional)
    embedding_dim: int = 0
    query_projection: tuple[tuple[float, ...], ...] = ()
    model_preference: Mapping[str, tuple[float, ...]] | None = None

    @property
    def uses_embeddings(self) -> bool:
        return bool(self.model_preference) and bool(self.query_projection)

    def logit(self, features: Mapping[str, float]) -> float:
        x = [features.get(name, 0.0) for name in self.feature_names]
        k = self.latent_dim
        q = [_dot(x, [self.projection[i][j] for i in range(len(x))]) for j in range(k)]
        return _dot(q, self.verifier_latent) + self.bias

    def embed_logit(self, embedding: Sequence[float]) -> float:
        k = self.latent_dim
        q = [
            _dot(embedding, [self.query_projection[i][j] for i in range(len(embedding))])
            for j in range(k)
        ]
        pref = self.model_preference or {}
        delta = [
            pref.get("DEEP", (0.0,) * k)[j] - pref.get("FAST", (0.0,) * k)[j]
            for j in range(k)
        ]
        return _dot(q, delta) + self.bias


class MatrixFactorizationRouter(ComplexityClassifier):
    """
    Routing model in the shape of RouteLLM's matrix-factorisation router.

    Algorithm
    ---------
    1. embed the prompt:            ``e = Embedder(prompt)``          (``E``-dim)
    2. project into latent space:   ``q = e @ Wq``                    (``K``-dim)
    3. score as an arm preference:  ``s = sigmoid(q . (p_DEEP - p_FAST) + b)``

    where ``Wq`` (``E x K``) and the per-arm latent vectors ``p_DEEP`` / ``p_FAST``
    (``K``-dim each) are fitted offline on preference data — "for this query, did
    escalating to the strong verifier change the outcome?" — the routing analogue
    of RouteLLM's "did the strong model win?".

    Algorithmic complexity
    ----------------------
    * embedding:   ``O(L)`` in prompt length ``L`` (hashing) or one encoder
      forward pass (a few ms on CPU for a MiniLM-class model);
    * projection:  ``O(E * K)`` — a single dense matmul;
    * scoring:     ``O(K)``.

    Total ``O(L + E*K)``, with ``K`` small (4–16) and ``E`` fixed (64–384). This
    is *independent of the DEEP verifier's* ``O(claims * chunks * top_k)`` NLI
    cost — the entire point of routing. Inference is allocation-light and
    trivially batchable.

    Enterprise weights
    ------------------
    Use :meth:`from_pretrained` to load a fitted checkpoint. Conceptually::

        router = MatrixFactorizationRouter.from_pretrained(
            "s3://controlplane-models/routers/mf-v3.json",
            embedder=SentenceTransformerBackend("all-MiniLM-L6-v2"),
        )

    which mirrors RouteLLM's ``Controller(routers=["mf"], strong_model=...,
    weak_model=...)`` — here ``strong`` = DEEP verification, ``weak`` = FAST.
    """

    model_name = "matrix-factorization-router"

    def __init__(
        self,
        params: MFParameters,
        *,
        embedder: EmbeddingBackend | None = None,
    ) -> None:
        self.params = params
        self.embedder = embedder
        mode = "embedding" if (params.uses_embeddings and embedder) else "feature-bag"
        self.model_name = f"{type(self).model_name} (k={params.latent_dim}, {mode})"

    # -- construction ---------------------------------------------------- #
    @classmethod
    def from_preferences(
        cls, samples: Sequence["RoutingSample"], **fit_kwargs: object
    ) -> "MatrixFactorizationRouter":
        """Fit the feature-bag path from labelled routing samples (offline)."""
        return cls(fit_matrix_factorization(samples, **fit_kwargs))  # type: ignore[arg-type]

    @classmethod
    def from_preferences_with_embeddings(
        cls,
        samples: Sequence["EmbeddingRoutingSample"],
        embedder: EmbeddingBackend,
        **fit_kwargs: object,
    ) -> "MatrixFactorizationRouter":
        """Fit the RouteLLM-style embedding path from ``(prompt, label)`` pairs."""
        params = fit_matrix_factorization_embeddings(samples, embedder, **fit_kwargs)  # type: ignore[arg-type]
        return cls(params, embedder=embedder)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | None = None,
        *,
        embedder: EmbeddingBackend | None = None,
    ) -> "MatrixFactorizationRouter":
        """
        Load a fitted MF checkpoint.

        Expected JSON schema::

            {
              "embedding_dim": 384,
              "latent_dim": 8,
              "query_projection": [[...], ...],          # E x K
              "model_preference": {"DEEP": [...], "FAST": [...]},   # K each
              "bias": -0.31,
              "trained_on": "2026-08-30 / 41k preference pairs"
            }

        **No live model backend is connected in this environment.** When
        ``checkpoint_path`` is ``None`` or unreadable this returns a router with
        deterministically-synthesised parameters (a working simulation) and leaves the
        wiring point obvious: swap in the real loader and a real
        :class:`EmbeddingBackend` and nothing else changes.
        """
        embedder = embedder or HashingEmbeddingBackend()
        if checkpoint_path:
            try:  # pragma: no cover - exercised only with a real artifact
                import json
                import pathlib

                raw = json.loads(pathlib.Path(checkpoint_path).read_text("utf-8"))
                return cls(
                    MFParameters(
                        embedding_dim=int(raw["embedding_dim"]),
                        latent_dim=int(raw["latent_dim"]),
                        query_projection=tuple(map(tuple, raw["query_projection"])),
                        model_preference={
                            k: tuple(v) for k, v in raw["model_preference"].items()
                        },
                        bias=float(raw.get("bias", 0.0)),
                        training_samples=int(raw.get("training_samples", 0)),
                    ),
                    embedder=embedder,
                )
            except (OSError, KeyError, ValueError):
                pass
        return cls(_synthesise_mf_parameters(embedder.dim or 64), embedder=embedder)

    # -- inference ----------------------------------------------------- #
    def predict_complexity(self, item: RoutingInput) -> ComplexityPrediction:
        if self.params.uses_embeddings and self.embedder is not None:
            embedding = self.embedder.embed(item.prompt)
            score = _sigmoid(self.params.embed_logit(embedding))
            feats = {"prompt_tokens": float(estimate_tokens(item.prompt)),
                     "embedding_dim": float(len(embedding))}
        else:
            feats = extract_features(item)
            score = self.predict_from_features(feats)
        return ComplexityPrediction(
            score=round(score, 6),
            model_name=self.model_name,
            features={k: round(v, 4) for k, v in feats.items()},
        )

    def predict_from_features(self, features: Mapping[str, float]) -> float:
        return _sigmoid(self.params.logit(features))


def fit_matrix_factorization(
    samples: Sequence["RoutingSample"],
    *,
    latent_dim: int = 4,
    epochs: int = 500,
    learning_rate: float = 0.06,
    l2: float = 1e-4,
) -> MFParameters:
    """
    Fit :class:`MFParameters` by minimising logistic loss on labelled routing
    outcomes with L2 regularisation. Pure Python, fully deterministic (fixed
    structured initialisation, fixed sample order — no RNG), a few milliseconds
    for the dataset sizes this router is trained on.

    Each sample carries a feature bag and ``deep_would_change_decision`` — the
    label we want to predict.
    """
    if not samples:
        raise ValueError("fit_matrix_factorization requires >= 1 sample")

    names = tuple(sorted({name for s in samples for name in s.features}))
    d, k = len(names), latent_dim
    # deterministic structured init in a small band around zero
    W = [[0.02 * (((i * 7 + j * 13) % 5) - 2) for j in range(k)] for i in range(d)]
    v = [0.05 * ((j % 3) - 1) or 0.03 for j in range(k)]
    b = 0.0

    xs = [[s.features.get(n, 0.0) for n in names] for s in samples]
    ys = [1.0 if s.deep_would_change_decision else 0.0 for s in samples]

    for _ in range(epochs):
        for x, y in zip(xs, ys):
            q = [sum(x[i] * W[i][j] for i in range(d)) for j in range(k)]
            pred = _sigmoid(sum(q[j] * v[j] for j in range(k)) + b)
            err = pred - y
            for j in range(k):
                grad_q_j = err * v[j]
                v[j] -= learning_rate * (err * q[j] + l2 * v[j])
                for i in range(d):
                    W[i][j] -= learning_rate * (grad_q_j * x[i] + l2 * W[i][j])
            b -= learning_rate * err

    return MFParameters(
        feature_names=names,
        projection=tuple(tuple(row) for row in W),
        verifier_latent=tuple(v),
        bias=b,
        latent_dim=k,
        training_samples=len(samples),
    )


# A tiny bundled preference set so ``build_default_mf_router()`` needs no args.
# Rows are (feature bag, deep_changed_decision). Chosen to span the surface:
# grounded echoes -> fast; hedged / ungrounded / numeric / action -> deep.
_BUNDLED_PREFERENCES: tuple[tuple[dict[str, float], bool], ...] = (
    ({"low_context_overlap": 0.05, "digit_density": 0.0, "hedge_density": 0.0,
      "entity_density": 0.0, "no_evidence": 0.0, "claim_count": 0.2}, False),
    ({"low_context_overlap": 0.10, "digit_density": 0.1, "hedge_density": 0.0,
      "entity_density": 0.1, "no_evidence": 0.0, "claim_count": 0.3}, False),
    ({"low_context_overlap": 0.20, "digit_density": 0.0, "hedge_density": 0.1,
      "entity_density": 0.0, "no_evidence": 0.0, "claim_count": 0.4,
      "question_form": 1.0}, False),
    ({"low_context_overlap": 0.35, "digit_density": 0.2, "hedge_density": 0.2,
      "entity_density": 0.3, "no_evidence": 0.0, "claim_count": 0.5}, False),
    ({"low_context_overlap": 0.55, "digit_density": 0.4, "hedge_density": 0.4,
      "entity_density": 0.4, "no_evidence": 0.0, "claim_count": 0.6}, True),
    ({"low_context_overlap": 0.70, "digit_density": 0.5, "hedge_density": 0.6,
      "entity_density": 0.5, "no_evidence": 0.0, "claim_count": 0.7,
      "action_flag": 1.0}, True),
    ({"low_context_overlap": 1.0, "digit_density": 0.3, "hedge_density": 0.5,
      "entity_density": 0.4, "no_evidence": 1.0, "claim_count": 0.6}, True),
    ({"low_context_overlap": 1.0, "digit_density": 0.7, "hedge_density": 0.7,
      "entity_density": 0.6, "no_evidence": 1.0, "claim_count": 0.9,
      "action_flag": 1.0, "response_length_ratio": 0.8}, True),
    ({"low_context_overlap": 0.15, "digit_density": 0.05, "hedge_density": 0.0,
      "entity_density": 0.1, "no_evidence": 0.0, "claim_count": 0.25}, False),
    ({"low_context_overlap": 0.85, "digit_density": 0.6, "hedge_density": 0.55,
      "entity_density": 0.55, "no_evidence": 0.0, "claim_count": 0.8,
      "action_flag": 1.0}, True),
)


@lru_cache(maxsize=1)
def build_default_mf_router() -> MatrixFactorizationRouter:
    """A ready MF router fitted on the bundled preference set. Cached per process."""
    samples = tuple(
        RoutingSample(features=f, deep_would_change_decision=label, complexity_score=0.0)
        for f, label in _BUNDLED_PREFERENCES
    )
    return MatrixFactorizationRouter(fit_matrix_factorization(samples))


# --------------------------------------------------------------------------- #
# Embedding-path MF: RouteLLM-style fit + pretrained loading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class EmbeddingRoutingSample:
    """A ``(prompt, label)`` preference pair for the embedding MF path."""

    prompt: str
    deep_would_change_decision: bool


def _synthesise_mf_parameters(embedding_dim: int, latent_dim: int = 8) -> MFParameters:
    """
    Deterministically build a *working simulation* of a fitted embedding-MF checkpoint
    (used by ``from_pretrained`` when no real artifact is reachable). Structured,
    RNG-free — the projection is a fixed sinusoidal basis and the two arm
    vectors are pushed apart so the preference term is non-degenerate.
    """
    e, k = embedding_dim, latent_dim
    proj = tuple(
        tuple(0.15 * math.cos((i * (j + 1) + 1) * 0.37) for j in range(k))
        for i in range(e)
    )
    deep = tuple(0.6 + 0.05 * (j % 3) for j in range(k))
    fast = tuple(-0.6 - 0.05 * (j % 3) for j in range(k))
    return MFParameters(
        embedding_dim=e,
        latent_dim=k,
        query_projection=proj,
        model_preference={"DEEP": deep, "FAST": fast},
        bias=-0.10,
    )


def fit_matrix_factorization_embeddings(
    samples: Sequence[EmbeddingRoutingSample],
    embedder: EmbeddingBackend,
    *,
    latent_dim: int = 8,
    epochs: int = 300,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
) -> MFParameters:
    """
    Fit the RouteLLM-style embedding path: learn ``query_projection`` (``E x K``)
    and the two arm latent vectors by minimising logistic loss on embedded
    prompts. Deterministic (structured init, fixed order). ``O(epochs * N * E * K)``.
    """
    if not samples:
        raise ValueError("fit_matrix_factorization_embeddings requires >= 1 sample")

    embeds = [embedder.embed(s.prompt) for s in samples]
    ys = [1.0 if s.deep_would_change_decision else 0.0 for s in samples]
    e, k = embedder.dim, latent_dim

    Wq = [[0.02 * (((i * 5 + j * 11) % 7) - 3) for j in range(k)] for i in range(e)]
    p_deep = [0.10 + 0.01 * (j % 4) for j in range(k)]
    p_fast = [-0.10 - 0.01 * (j % 4) for j in range(k)]
    b = 0.0

    for _ in range(epochs):
        for x, y in zip(embeds, ys):
            q = [sum(x[i] * Wq[i][j] for i in range(e)) for j in range(k)]
            delta = [p_deep[j] - p_fast[j] for j in range(k)]
            pred = _sigmoid(sum(q[j] * delta[j] for j in range(k)) + b)
            err = pred - y
            for j in range(k):
                g_qj = err * delta[j]
                p_deep[j] -= learning_rate * (err * q[j] + l2 * p_deep[j])
                p_fast[j] -= learning_rate * (-err * q[j] + l2 * p_fast[j])
                for i in range(e):
                    Wq[i][j] -= learning_rate * (g_qj * x[i] + l2 * Wq[i][j])
            b -= learning_rate * err

    return MFParameters(
        embedding_dim=e,
        latent_dim=k,
        query_projection=tuple(tuple(row) for row in Wq),
        model_preference={"DEEP": tuple(p_deep), "FAST": tuple(p_fast)},
        bias=b,
        training_samples=len(samples),
    )


_BUNDLED_EMBED_PREFERENCES: tuple[tuple[str, bool], ...] = (
    ("what are the support hours", False),
    ("where is the roadmap document", False),
    ("is the office open on monday", False),
    ("confirm my email address on file", False),
    ("should I move my balance into the new fund", True),
    ("what is the guaranteed return on this product", True),
    ("estimate the refund amount for my order", True),
    ("is this medication safe to combine with mine", True),
    ("summarise the contract's liability clause", True),
    ("what is your return policy window", False),
)


@lru_cache(maxsize=1)
def build_default_embedding_mf_router() -> MatrixFactorizationRouter:
    """A ready embedding-path MF router (hashing encoder + bundled prefs)."""
    embedder = HashingEmbeddingBackend(dim=64)
    samples = tuple(
        EmbeddingRoutingSample(prompt=p, deep_would_change_decision=lbl)
        for p, lbl in _BUNDLED_EMBED_PREFERENCES
    )
    return MatrixFactorizationRouter(
        fit_matrix_factorization_embeddings(samples, embedder), embedder=embedder
    )


# --------------------------------------------------------------------------- #
# Offline threshold calibration (safety first, then cost)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RoutingSample:
    """
    One labelled routing example.

    ``features`` feeds the MF trainer; ``complexity_score`` (a model's output on
    this sample) feeds the threshold sweep; ``deep_would_change_decision`` is the
    label for both.
    """

    deep_would_change_decision: bool
    features: Mapping[str, float] = field(default_factory=dict)
    complexity_score: float = 0.0


@dataclass(frozen=True, slots=True)
class ThresholdCalibration:
    threshold: float
    expected_cost_per_call: float
    escalation_rate: float
    missed_risk_rate: float
    samples_evaluated: int
    satisfied_constraint: bool


def select_cost_optimal_threshold(
    samples: Sequence[RoutingSample],
    *,
    deep_verification_cost: float,
    missed_risk_penalty: float,
    max_missed_risk_rate: float = 0.02,
    candidate_thresholds: Sequence[float] | None = None,
) -> ThresholdCalibration:
    """
    Pick the escalation threshold that minimises expected cost per call subject
    to ``missed_risk_rate <= max_missed_risk_rate`` — the same "safety first,
    then efficiency" ordering ControlPlane's ``calibration.select`` uses.

    For candidate ``t``: escalate iff ``complexity_score >= t``;
    ``cost = deep_verification_cost`` when escalating,
    ``missed_risk_penalty`` when not escalating a sample that DEEP would have
    flipped, ``0`` otherwise.

    If no candidate satisfies the constraint, returns the *safest* one (lowest
    missed-risk rate) with ``satisfied_constraint = False``.
    """
    if not samples:
        raise ValueError("select_cost_optimal_threshold requires >= 1 sample")

    grid = (
        list(candidate_thresholds)
        if candidate_thresholds is not None
        else [round(i / 100, 2) for i in range(0, 101, 2)]
    )
    n = len(samples)
    best: ThresholdCalibration | None = None
    safest: ThresholdCalibration | None = None

    for t in grid:
        cost = escalations = missed = 0.0
        for s in samples:
            if s.complexity_score >= t:
                escalations += 1
                cost += deep_verification_cost
            elif s.deep_would_change_decision:
                missed += 1
                cost += missed_risk_penalty
        cand = ThresholdCalibration(
            threshold=t,
            expected_cost_per_call=cost / n,
            escalation_rate=escalations / n,
            missed_risk_rate=missed / n,
            samples_evaluated=n,
            satisfied_constraint=(missed / n) <= max_missed_risk_rate,
        )
        if safest is None or cand.missed_risk_rate < safest.missed_risk_rate:
            safest = cand
        if cand.satisfied_constraint and (
            best is None or cand.expected_cost_per_call < best.expected_cost_per_call
        ):
            best = cand

    assert safest is not None
    return best or safest
