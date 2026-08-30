"""
Structured decision reason codes.

Reason codes are the machine-readable, deterministic "why" behind a
ControlPlane decision. They sit alongside ``triggered_rules`` (which name
the policy rules that fired) and the free-text ``explanation``:

* ``triggered_rules``  — internal policy-rule identifiers
* ``reason_codes``     — canonical, stable, product-level reasons (this module)
* ``explanation``      — a human sentence generated from the trace

Every reason code must be traceable to concrete detector / policy /
consequence evidence in the ``DecisionTrace``. They are never generated
speculatively.
"""

from __future__ import annotations

from enum import Enum


class ReasonCode(str, Enum):
    # performance / grounding
    CONTRADICTED_EVIDENCE = "CONTRADICTED_EVIDENCE"
    LOW_VERIFICATION_COVERAGE = "LOW_VERIFICATION_COVERAGE"
    HIGH_PERFORMANCE_RISK = "HIGH_PERFORMANCE_RISK"

    # responsibility
    CRITICAL_PII = "CRITICAL_PII"
    PII_EXPOSURE = "PII_EXPOSURE"
    TOXICITY = "TOXICITY"
    POTENTIAL_BIAS_SIGNAL = "POTENTIAL_BIAS_SIGNAL"
    HIGH_RESPONSIBILITY_RISK = "HIGH_RESPONSIBILITY_RISK"

    # cost / operational
    COST_SPIKE = "COST_SPIKE"
    RETRY_ANOMALY = "RETRY_ANOMALY"
    TOOL_LOOP = "TOOL_LOOP"
    LATENCY_ANOMALY = "LATENCY_ANOMALY"

    # consequence / criticality
    HIGH_CONSEQUENCE = "HIGH_CONSEQUENCE"
    HIGH_FINANCIAL_IMPACT = "HIGH_FINANCIAL_IMPACT"
    IRREVERSIBLE_ACTION = "IRREVERSIBLE_ACTION"
    AUTOMATED_EXTERNAL_ACTION = "AUTOMATED_EXTERNAL_ACTION"
    HIGH_BLAST_RADIUS = "HIGH_BLAST_RADIUS"

    # cross-cutting
    MULTI_RISK = "MULTI_RISK"
    SESSION_ESCALATION = "SESSION_ESCALATION"
    LOW_CONFIDENCE_HIGH_RISK = "LOW_CONFIDENCE_HIGH_RISK"


DESCRIPTIONS: dict[ReasonCode, str] = {
    ReasonCode.CONTRADICTED_EVIDENCE: "A claim in the response contradicts the supplied evidence.",
    ReasonCode.LOW_VERIFICATION_COVERAGE: "Key claims could not be verified against available evidence.",
    ReasonCode.HIGH_PERFORMANCE_RISK: "The response's grounding risk is high.",
    ReasonCode.CRITICAL_PII: "The response exposes critical personal or financial identifiers.",
    ReasonCode.PII_EXPOSURE: "The response contains personally identifiable information.",
    ReasonCode.TOXICITY: "The response contains toxic, threatening or harassing language.",
    ReasonCode.POTENTIAL_BIAS_SIGNAL: "The response shows a heuristic bias signal requiring human interpretation.",
    ReasonCode.HIGH_RESPONSIBILITY_RISK: "The combined responsibility risk (PII/toxicity/bias) is high.",
    ReasonCode.COST_SPIKE: "The interaction's estimated cost or token usage is anomalously high.",
    ReasonCode.RETRY_ANOMALY: "The interaction used an abnormal number of retries.",
    ReasonCode.TOOL_LOOP: "Tool-call behaviour looks like an unproductive loop.",
    ReasonCode.LATENCY_ANOMALY: "The interaction latency is anomalously high.",
    ReasonCode.HIGH_CONSEQUENCE: "The consequence of the action being wrong is high.",
    ReasonCode.HIGH_FINANCIAL_IMPACT: "The action carries a large financial exposure.",
    ReasonCode.IRREVERSIBLE_ACTION: "The action would be hard or impossible to reverse.",
    ReasonCode.AUTOMATED_EXTERNAL_ACTION: "The action executes automatically and/or reaches outside parties.",
    ReasonCode.HIGH_BLAST_RADIUS: "The action affects a large number of entities.",
    ReasonCode.MULTI_RISK: "Two or more independent risk dimensions are elevated at once.",
    ReasonCode.SESSION_ESCALATION: "Repeated high-risk turns in this session raised the effective risk.",
    ReasonCode.LOW_CONFIDENCE_HIGH_RISK: "Risk appears high but detector confidence is low; human review preferred over an automatic block.",
}


def describe(code: ReasonCode | str) -> str:
    try:
        return DESCRIPTIONS[ReasonCode(code)]
    except (ValueError, KeyError):
        return str(code)


def dedupe(codes: list[ReasonCode | str]) -> list[str]:
    """Order-preserving de-duplication, coercing to plain strings."""
    seen: set[str] = set()
    out: list[str] = []
    for code in codes:
        value = code.value if isinstance(code, ReasonCode) else str(code)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
