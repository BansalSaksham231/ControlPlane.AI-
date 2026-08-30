"""
Text segmentation utilities for the Performance Detector.

This module performs ONLY segmentation:

1. Extracting atomic, sentence-like claims from an AI response.
2. Splitting supplied context into evidence chunks.

It never judges whether a claim is true, false, supported, or
contradicted — that is the job of the retrieval/NLI layer. A claim
that flatly contradicts its context is still extracted verbatim and
returned as-is; this module has no opinion about it.

Standard library only. Deterministic. No file I/O, no configuration
loading, no detector logic.
"""

from __future__ import annotations

import re

# Splits a single line into sentence-like fragments after a sentence
# terminator (., !, ?) that is followed by whitespace. The terminator
# itself stays attached to the preceding fragment. Lines with no
# terminator are returned whole.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """
    Split ``text`` into sentence-like fragments, honoring newlines and
    ``.``/``!``/``?`` as boundaries.

    Returns stripped, non-empty fragments in original order. Never
    raises on empty, whitespace-only, or malformed input.
    """
    if not text or not text.strip():
        return []

    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for fragment in _SENTENCE_SPLIT_RE.split(line):
            fragment = fragment.strip()
            if fragment:
                sentences.append(fragment)

    return sentences


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """
    Split a single sentence that exceeds ``max_chars`` into smaller
    pieces, preferring word boundaries.

    A single word longer than ``max_chars`` is hard-split by
    character so this never crashes and always returns a finite list.
    """
    if max_chars <= 0:
        # Degenerate config guard — fall back to a safe minimum so we
        # still terminate rather than looping forever or crashing.
        max_chars = 1

    chunks: list[str] = []
    current = ""

    for word in sentence.split():
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(word), max_chars):
                chunks.append(word[start : start + max_chars])
            continue

        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars:
            if current:
                chunks.append(current)
            current = word
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def extract_claims(response: str, min_tokens: int = 2) -> list[str]:
    """
    Extract atomic, sentence-like claims from an AI ``response``.

    This is pure segmentation: a claim is returned exactly as worded,
    regardless of whether it is accurate, exaggerated, unsupported, or
    contradicts any given context. That judgment happens later, in
    the retrieval/NLI layer — not here.

    Fragments with fewer than ``min_tokens`` whitespace-separated
    tokens are dropped (e.g. filler like "Okay." at ``min_tokens=2``).

    Returns claims in original order. Never raises on empty or
    whitespace-only input.
    """
    sentences = _split_sentences(response)
    return [sentence for sentence in sentences if len(sentence.split()) >= min_tokens]


def chunk_context(context: str, max_chars: int = 800) -> list[str]:
    """
    Split supplied ``context`` into evidence chunks for retrieval.

    Prefers sentence boundaries. Any single sentence longer than
    ``max_chars`` is further split (preferring word boundaries, with a
    hard character split as a last resort) so no returned chunk ever
    exceeds ``max_chars``.

    Returns chunks in original order. Never raises on empty,
    whitespace-only, or very long input.
    """
    sentences = _split_sentences(context)
    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            chunks.extend(_split_long_sentence(sentence, max_chars))
    return chunks