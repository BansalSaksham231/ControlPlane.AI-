"""
Lightweight lexical NLI backend for the Performance Detector.

Given a premise (retrieved evidence) and a hypothesis (an AI-generated
claim), this module estimates whether the hypothesis is entailed by,
contradicted by, or neutral with respect to the premise.

Architecture:

    claim (hypothesis) + retrieved evidence (premise)
        -> NLI
        -> ENTAILMENT / CONTRADICTION / NEUTRAL

IMPORTANT PROTOTYPE LIMITATION
-------------------------------
This is a transparent, deterministic, rule-based lexical prototype —
NOT a trained NLI transformer. It approximates entailment/contradiction
using token overlap, negation polarity, and numeric agreement. It will
miss genuine semantic relationships that don't show up lexically (e.g.
synonyms, paraphrase without word overlap, complex logical inference)
and can be fooled by surface-level tricks. It exists so the system
runs on a normal laptop without a model download. The ``NLIBackend``
abstraction lets a stronger transformer-based backend be substituted
later without any caller changes.

Standard library only. Deterministic: no randomness, no wall-clock
time, no network calls, no file I/O, no configuration loading, no
ground-truth access, no risk/consequence/policy logic.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from detectors.performance.schemas import NLILabel

# ==================================================
# TOKENIZATION
# ==================================================

# Matches words/numbers, keeping simple contractions (e.g. "can't",
# "doesn't") intact as single tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Negation cues. Presence/absence of these drives polarity comparison.
NEGATION_WORDS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "cannot",
        "can't",
        "won't",
        "doesn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "without",
    }
)

# Modal verbs are treated as near-interchangeable paraphrase cues
# ("may" vs "can") rather than content words that must match exactly.
_MODAL_WORDS: frozenset[str] = frozenset(
    {"can", "could", "may", "might", "will", "would", "shall", "should", "must"}
)

# Generic function words that carry little topical content.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "but",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "by",
        "with",
        "from",
        "within",
        "per",
    }
)

# ==================================================
# THRESHOLDS
# ==================================================

# Minimum content-word overlap required before a negation mismatch is
# treated as a genuine contradiction (rather than two unrelated
# sentences that happen to each contain/omit a negation word).
_NEGATION_CONTRADICTION_OVERLAP = 0.4

# Minimum content-word overlap required before disagreeing numbers are
# treated as a genuine contradiction (rather than two unrelated facts
# that happen to both contain numbers).
_NUMERIC_CONTRADICTION_OVERLAP = 0.4

# Minimum content-word overlap required to call something entailment.
_ENTAILMENT_OVERLAP = 0.5


def _normalize(text: str) -> list[str]:
    """
    Tokenize ``text`` into lowercase word/number tokens.

    Strips punctuation except the apostrophe inside simple
    contractions (so negations like "doesn't" stay intact as one
    token), and preserves numeric tokens verbatim.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _content_words(tokens: list[str]) -> set[str]:
    """Content-bearing tokens: not a stopword, modal, negation, or number."""
    return {
        token
        for token in tokens
        if token not in _STOPWORDS
        and token not in _MODAL_WORDS
        and token not in NEGATION_WORDS
        and not token.isdigit()
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard overlap of two token sets; 0.0 when both are empty."""
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


# ==================================================
# BACKEND ABSTRACTION
# ==================================================


class NLIBackend(ABC):
    """
    Abstraction the retrieval layer depends on, instead of depending
    directly on any particular NLI implementation. This makes it
    possible to swap in a transformer-based backend later without
    changing callers.
    """

    @abstractmethod
    def predict(self, premise: str, hypothesis: str) -> tuple[NLILabel, float]:
        """
        Classify ``hypothesis`` (the AI claim) against ``premise`` (the
        evidence/context) as ENTAILMENT, CONTRADICTION, or NEUTRAL.

        Returns ``(label, confidence)`` where ``confidence`` is in
        [0, 1] and reflects the strength of the classification, not a
        calibrated probability.
        """
        raise NotImplementedError

    def predict_batch(
        self, premise: str, hypotheses: list[str]
    ) -> list[tuple[NLILabel, float]]:
        """Apply ``predict`` to each hypothesis against the same premise, in order."""
        return [self.predict(premise, hypothesis) for hypothesis in hypotheses]


class LexicalNLIBackend(NLIBackend):
    """
    Deterministic, rule-based lexical NLI backend.

    Decision process for a given (premise, hypothesis) pair:

    1. Tokenize both texts; identify negation cues and numeric tokens.
    2. Compute Jaccard overlap of content words (ignoring stopwords,
       modal verbs, negation words, and numbers) as a proxy for
       "are these about the same thing".
    3. If the two disagree on negation polarity and are clearly about
       the same topic (overlap above threshold) -> CONTRADICTION.
    4. Else if both mention numbers, the numbers disagree, and the two
       are clearly about the same topic -> CONTRADICTION.
    5. Else if content-word overlap is high enough -> ENTAILMENT.
    6. Otherwise -> NEUTRAL (weak or no useful lexical relationship).

    UNVERIFIED is not a value this backend produces directly — a
    NEUTRAL result (or the absence of retrieved evidence entirely)
    is how "insufficient evidence" gets represented upstream.
    """

    def predict(self, premise: str, hypothesis: str) -> tuple[NLILabel, float]:
        premise_tokens = _normalize(premise)
        hypothesis_tokens = _normalize(hypothesis)

        premise_negated = any(token in NEGATION_WORDS for token in premise_tokens)
        hypothesis_negated = any(token in NEGATION_WORDS for token in hypothesis_tokens)
        negation_mismatch = premise_negated != hypothesis_negated

        premise_numbers = {token for token in premise_tokens if token.isdigit()}
        hypothesis_numbers = {token for token in hypothesis_tokens if token.isdigit()}
        numeric_conflict = (
            bool(premise_numbers)
            and bool(hypothesis_numbers)
            and premise_numbers != hypothesis_numbers
        )

        overlap = _jaccard(
            _content_words(premise_tokens), _content_words(hypothesis_tokens)
        )

        if negation_mismatch and overlap >= _NEGATION_CONTRADICTION_OVERLAP:
            confidence = min(0.97, 0.75 + 0.25 * overlap)
            return NLILabel.CONTRADICTION, round(confidence, 2)

        if numeric_conflict and overlap >= _NUMERIC_CONTRADICTION_OVERLAP:
            confidence = min(0.95, 0.75 + 0.20 * overlap)
            return NLILabel.CONTRADICTION, round(confidence, 2)

        if overlap >= _ENTAILMENT_OVERLAP:
            confidence = min(0.97, 0.70 + 0.30 * overlap)
            return NLILabel.ENTAILMENT, round(confidence, 2)

        # Weak or no useful lexical relationship -> NEUTRAL. Confidence
        # scales gently with whatever small overlap exists: near 0.50
        # for essentially unrelated text, drifting up toward ~0.55-0.60
        # for topics that share a little vocabulary but not enough to
        # call support or contradiction.
        confidence = min(0.60, 0.45 + 0.30 * overlap)
        return NLILabel.NEUTRAL, round(confidence, 2)


# Directional-coverage thresholds for the entailment-oriented backend.
_COVERAGE_ENTAILMENT = 0.48
_COVERAGE_CONFLICT_MIN = 0.30


class CoverageNLIBackend(NLIBackend):
    """
    Directional lexical NLI backend used by the integrated Performance
    Detector.

    Where :class:`LexicalNLIBackend` measures *symmetric* Jaccard overlap
    (which collapses whenever the premise carries many words the
    hypothesis does not), this backend measures **directional coverage**:
    what fraction of the hypothesis's content words are present in the
    premise. That matches the real question — "does the evidence cover
    what the claim asserts?" — and behaves far better on retrieved
    evidence sentences that are longer and wordier than the claim.

    Same system limitations apply: it is a transparent, deterministic,
    rule-based approximation, not a trained NLI transformer. It reasons
    over token coverage, negation polarity and numeric agreement only.
    """

    def predict(self, premise: str, hypothesis: str) -> tuple[NLILabel, float]:
        premise_tokens = _normalize(premise)
        hypothesis_tokens = _normalize(hypothesis)

        premise_content = _content_words(premise_tokens)
        hypothesis_content = _content_words(hypothesis_tokens)

        if not hypothesis_content:
            return NLILabel.NEUTRAL, 0.50

        covered = len(premise_content & hypothesis_content)
        coverage = covered / len(hypothesis_content)

        premise_negated = any(t in NEGATION_WORDS for t in premise_tokens)
        hypothesis_negated = any(t in NEGATION_WORDS for t in hypothesis_tokens)
        negation_mismatch = premise_negated != hypothesis_negated

        # Only compare "small" standalone numbers (<= 3 digits): day
        # counts, percentages, small quantities. Longer digit runs are
        # identifiers (account / order / phone / card numbers) whose
        # digits carry no entailment meaning and would cause spurious
        # numeric "conflicts".
        premise_numbers = {t for t in premise_tokens if t.isdigit() and len(t) <= 3}
        hypothesis_numbers = {t for t in hypothesis_tokens if t.isdigit() and len(t) <= 3}
        numeric_conflict = (
            bool(premise_numbers)
            and bool(hypothesis_numbers)
            and premise_numbers.isdisjoint(hypothesis_numbers)
        )

        on_topic = coverage >= _COVERAGE_CONFLICT_MIN

        if on_topic and negation_mismatch:
            confidence = min(0.97, 0.72 + 0.25 * coverage)
            return NLILabel.CONTRADICTION, round(confidence, 4)

        if on_topic and numeric_conflict:
            confidence = min(0.95, 0.72 + 0.20 * coverage)
            return NLILabel.CONTRADICTION, round(confidence, 4)

        if coverage >= _COVERAGE_ENTAILMENT:
            confidence = min(0.95, 0.55 + 0.40 * coverage)
            return NLILabel.ENTAILMENT, round(confidence, 4)

        confidence = min(0.60, 0.40 + 0.35 * coverage)
        return NLILabel.NEUTRAL, round(confidence, 4)