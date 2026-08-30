"""
Data contracts for the unified Responsibility Detector.

The Responsibility Detector composes three transparent, deterministic
sub-detectors — PII, toxicity and bias — into a single result. None of
them use ground truth. Each is a heuristic system, not a certified
classifier; the bias sub-detector in particular reports a *signal*, not
established discrimination.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ResponsibilityCategory(str, Enum):
    PII = "PII"
    TOXICITY = "TOXICITY"
    BIAS = "BIAS"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Finding(BaseModel):
    """A single responsibility finding within an AI response."""

    category: ResponsibilityCategory
    subtype: str
    severity: Severity
    confidence: float = Field(ge=0, le=1)

    # Raw matched span is retained for the audit trail / human appeal only;
    # ``redacted_text`` is what dashboards and non-privileged views show.
    matched_text: str
    redacted_text: str
    span: tuple[int, int] | None = None

    explanation: str

    # Round 2 upgrade: why this severity (e.g. "financial identifier in an
    # external-communication response -> CRITICAL").
    severity_rationale: str = ""


class PIIResult(BaseModel):
    pii_risk: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    contains_critical_pii: bool = False
    critical_types: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    redacted_response: str
    explanation: str


class ToxicityResult(BaseModel):
    toxicity_risk: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    categories: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    explanation: str


class BiasResult(BaseModel):
    # Named ``bias_signal`` rather than ``bias_risk`` to underline that this
    # is a heuristic indicator, not a determination of discrimination.
    bias_signal: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    findings: list[Finding] = Field(default_factory=list)
    explanation: str


class ResponsibilityResult(BaseModel):
    """Unified output of the Responsibility Detector for one interaction."""

    pii_risk: float = Field(ge=0, le=1)
    toxicity_risk: float = Field(ge=0, le=1)
    bias_risk: float = Field(ge=0, le=1)

    overall_responsibility_risk: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)

    contains_critical_pii: bool = False
    critical_pii_types: list[str] = Field(default_factory=list)

    findings: list[Finding] = Field(default_factory=list)
    redacted_response: str

    # Individual sub-detector outputs, kept inspectable.
    pii: PIIResult
    toxicity: ToxicityResult
    bias: BiasResult

    explanation: str
    latency_ms: float = Field(ge=0)

    # Round 2 upgrade: canonical reason codes contributed by this layer.
    reason_codes: list[str] = Field(default_factory=list)
