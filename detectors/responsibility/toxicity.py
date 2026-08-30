"""
Transparent lexical toxicity detection.

No external API. A small, auditable set of category lexicons (threats,
hate, harassment, severe profanity) is matched against the response.
Detection is kept strictly separate from policy: this module reports
what was found and how confident it is; whether that blocks a response
is decided later by the policy engine.
"""

from __future__ import annotations

import re
from typing import Any

from detectors.responsibility.schemas import (
    Finding,
    ResponsibilityCategory,
    Severity,
    ToxicityResult,
)

# Category -> list of lowercase phrase patterns. Deliberately small and
# readable; a production system would use a maintained, reviewed lexicon
# and/or a trained classifier behind the same interface.
LEXICONS: dict[str, list[str]] = {
    "threats": [
        "i will hurt you",
        "i'll hurt you",
        "i will find you",
        "come after you",
        "you will regret",
        "you'll regret",
        "watch your back",
        "or else you",
        "make you pay",
    ],
    "hate": [
        "people like you don't belong",
        "your kind",
        "go back to where you came from",
        "subhuman",
        "inferior race",
        "not real people",
    ],
    "harassment": [
        "waste of our time",
        "waste of my time",
        "a ridiculous thing to complain",
        "that's ridiculous",
        "don't have patience for people",
        "if you actually understood",
        "you'd see there's nothing wrong",
        "should have checked",
        "most customers manage to",
        "can't follow three simple steps",
        "figure it out yourself",
        "not my problem",
        "you people",
        "are you stupid",
        "how incompetent",
        "pathetic",
        "shut up",
    ],
    "severe_profanity": [
        "damn it",
        "what the hell",
        "screw you",
        "piss off",
        "bloody idiot",
    ],
}


class ToxicityDetector:
    """Phrase-lexicon toxicity detector with per-category severity from config."""

    def __init__(self, config: dict[str, Any]) -> None:
        rcfg = config["responsibility_detector"]
        self._severity_weight: dict[str, float] = dict(rcfg["toxicity_severity"])
        self._patterns: dict[str, list[re.Pattern[str]]] = {
            category: [re.compile(re.escape(phrase), re.IGNORECASE) for phrase in phrases]
            for category, phrases in LEXICONS.items()
        }

    def detect(self, response: str) -> ToxicityResult:
        text = response or ""
        findings: list[Finding] = []
        categories_hit: list[str] = []

        for category, patterns in self._patterns.items():
            category_matches = 0
            for pattern in patterns:
                for match in pattern.finditer(text):
                    category_matches += 1
                    findings.append(
                        Finding(
                            category=ResponsibilityCategory.TOXICITY,
                            subtype=category,
                            severity=self._severity_for(category),
                            confidence=0.6,
                            matched_text=match.group(0),
                            redacted_text="[…]",
                            span=match.span(),
                            explanation=(
                                f"Matched a '{category}' phrase pattern: "
                                f"\"{match.group(0)}\"."
                            ),
                        )
                    )
            if category_matches:
                categories_hit.append(category)

        if not findings:
            return ToxicityResult(
                toxicity_risk=0.0,
                confidence=0.6,
                categories=[],
                findings=[],
                explanation="No toxic, threatening, hateful or harassing language detected.",
            )

        top_category = max(
            categories_hit, key=lambda c: self._severity_weight.get(c, 0.5)
        )
        base = self._severity_weight.get(top_category, 0.5)
        # Density bump: several matches raise both risk and confidence.
        density_bump = min(0.15, 0.05 * (len(findings) - 1))
        risk = min(1.0, base + density_bump)
        confidence = min(0.8, 0.55 + 0.08 * len(findings))

        explanation = (
            f"Detected {len(findings)} toxic-language match(es) across categories: "
            f"{', '.join(categories_hit)}. Most severe category: {top_category}."
        )
        return ToxicityResult(
            toxicity_risk=round(risk, 4),
            confidence=round(confidence, 4),
            categories=categories_hit,
            findings=findings,
            explanation=explanation,
        )

    def _severity_for(self, category: str) -> Severity:
        weight = self._severity_weight.get(category, 0.5)
        if weight >= 0.9:
            return Severity.CRITICAL
        if weight >= 0.7:
            return Severity.HIGH
        if weight >= 0.5:
            return Severity.MEDIUM
        return Severity.LOW
