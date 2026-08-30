"""
Deterministic PII / sensitive-entity detection.

Regex + light entity heuristics only — no models, no network. Every
finding carries both the raw matched span (for the audit trail) and a
redacted form (for everything else), and the detector can produce a
fully redacted copy of the response without losing the original record.
"""

from __future__ import annotations

import re
from typing import Any

from detectors.responsibility.schemas import (
    Finding,
    PIIResult,
    ResponsibilityCategory,
    Severity,
)

# ------------------------------------------------------------------
# patterns  (order matters: more specific / higher-severity first)
# ------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\-\s]{7,}\d)(?!\w)")
_CREDIT_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")
_AADHAAR_RE = re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PAN_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])")
_ACCOUNT_RE = re.compile(r"(?<![A-Za-z0-9])ACC-\d{5,}(?![A-Za-z0-9])")
_EMPLOYEE_RE = re.compile(r"(?<![A-Za-z0-9])EMP-\d{4,}(?![A-Za-z0-9])")
_ADDRESS_RE = re.compile(
    r"\d{1,4}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Nagar|Sector|Marg|Block)\b"
)
# Two/three capitalised words — a possible person name. Only escalated to a
# finding when it co-occurs with another sensitive marker (see detector).
_NAME_RE = re.compile(r"(?<![A-Za-z])[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}(?![A-Za-z])")

_NAME_STOP = {
    "Dear Customer",
    "Best Regards",
    "Kind Regards",
    "Thank You",
    "Purchase Order",
    "Team Alpha",
    "Team Beta",
    "Region North",
    "Region South",
    "Warehouse A",
    "Warehouse B",
}

# Leading words that mean a capitalised bigram is almost certainly not a
# person's name (usually sentence-initial).
_NAME_LEADING_STOP = {
    "Your", "The", "This", "That", "These", "Those", "Please", "Our", "We",
    "It", "As", "For", "With", "And", "But", "Order", "Account", "Employee",
    "Customer", "Purchase", "Region", "Team", "Warehouse", "Company", "Refunds",
    "Payment", "Here", "There", "If", "When", "After", "Before",
}


def _luhn_ok(digits: str) -> bool:
    nums = [int(ch) for ch in digits if ch.isdigit()]
    if len(nums) < 13:
        return False
    checksum = 0
    parity = len(nums) % 2
    for index, digit in enumerate(nums):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _mask(text: str, keep_start: int = 0, keep_end: int = 0, token: str = "*") -> str:
    if keep_start + keep_end >= len(text):
        return token * len(text)
    return text[:keep_start] + token * (len(text) - keep_start - keep_end) + (
        text[-keep_end:] if keep_end else ""
    )


def _redact_email(value: str) -> str:
    local, _, domain = value.partition("@")
    return f"{local[:1]}***@{domain.split('.')[0][:1]}***"


_SEVERITY_ORDER = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


class PIIDetector:
    """Regex-driven PII detector with per-entity severity from config."""

    # Actions whose responses reach outside parties (or are irreversible),
    # so any sensitive identifier in them is escalated to CRITICAL.
    _EXTERNAL_ACTIONS = {"external_communication", "account_cancellation"}
    _EXTERNALLY_IDENTIFYING = {
        "email", "phone", "person_name", "postal_address",
        "credit_card", "government_id", "financial_account",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        rcfg = config["responsibility_detector"]
        self._severity_weight: dict[str, float] = dict(rcfg["pii_severity"])
        self._critical_entities: set[str] = set(rcfg["pii_critical_entities"])

    # ------------------------------------------------------------------

    def detect(self, response: str, action_type: str | None = None) -> PIIResult:
        text = response or ""
        is_external = action_type in self._EXTERNAL_ACTIONS
        findings: list[Finding] = []
        claimed_spans: list[tuple[int, int]] = []

        def _overlaps(start: int, end: int) -> bool:
            return any(not (end <= s or start >= e) for s, e in claimed_spans)

        def _add(match: re.Match[str], entity: str, redacted: str, note: str) -> None:
            start, end = match.span()
            if _overlaps(start, end):
                return
            claimed_spans.append((start, end))
            severity = self._severity_for(entity)
            rationale = f"{entity} -> {severity.value} by base severity weight."
            if (
                is_external
                and entity in self._EXTERNALLY_IDENTIFYING
                and severity != Severity.CRITICAL
            ):
                severity = Severity.CRITICAL
                rationale = (
                    f"{entity} exposed in an outward-facing '{action_type}' "
                    "response -> escalated to CRITICAL."
                )
            findings.append(
                Finding(
                    category=ResponsibilityCategory.PII,
                    subtype=entity,
                    severity=severity,
                    confidence=self._confidence_for(entity),
                    matched_text=match.group(0),
                    redacted_text=redacted,
                    span=(start, end),
                    explanation=note,
                    severity_rationale=rationale,
                )
            )

        # Card numbers first: a Luhn-valid 13-16 digit run is almost
        # certainly a card and should claim its span before the shorter
        # government-ID patterns can partially match inside it.
        for match in _CREDIT_CARD_RE.finditer(text):
            raw = match.group(0)
            if _luhn_ok(raw):
                _add(match, "credit_card", "[REDACTED_CARD]", "Card-like number passing a Luhn check.")
        for match in _AADHAAR_RE.finditer(text):
            _add(match, "government_id", "[REDACTED_GOV_ID]", "Aadhaar-style 12-digit identifier.")
        for match in _SSN_RE.finditer(text):
            _add(match, "government_id", "[REDACTED_GOV_ID]", "US SSN-style identifier.")
        for match in _PAN_RE.finditer(text):
            _add(match, "government_id", "[REDACTED_GOV_ID]", "PAN-style identifier.")
        for match in _ACCOUNT_RE.finditer(text):
            _add(match, "account_id", "[REDACTED_ACCOUNT]", "Internal account identifier (ACC-######).")
        for match in _EMPLOYEE_RE.finditer(text):
            _add(match, "employee_id", "[REDACTED_EMP_ID]", "Internal employee identifier (EMP-#####).")
        for match in _EMAIL_RE.finditer(text):
            _add(match, "email", _redact_email(match.group(0)), "Email address.")
        for match in _PHONE_RE.finditer(text):
            digits = re.sub(r"\D", "", match.group(0))
            if len(digits) >= 8:
                _add(match, "phone", _mask(match.group(0), 0, 2), "Phone-number-like sequence.")
        for match in _ADDRESS_RE.finditer(text):
            _add(match, "postal_address", "[REDACTED_ADDRESS]", "Street-address-like phrase.")

        # Person names — only if the response already contains another
        # sensitive marker (name alone is not treated as PII).
        if findings:
            for match in _NAME_RE.finditer(text):
                phrase = match.group(0)
                if phrase in _NAME_STOP or phrase.split()[0] in _NAME_LEADING_STOP:
                    continue
                _add(match, "person_name", "[REDACTED_NAME]", "Personal name alongside other sensitive data.")

        return self._assemble(text, findings, is_external)

    # ------------------------------------------------------------------

    def _assemble(
        self, text: str, findings: list[Finding], is_external: bool = False
    ) -> PIIResult:
        findings.sort(key=lambda f: f.span[0] if f.span else 0)
        redacted = self.redact(text, findings)

        if not findings:
            return PIIResult(
                pii_risk=0.0,
                confidence=0.6,
                contains_critical_pii=False,
                findings=[],
                redacted_response=text,
                explanation="No PII or sensitive identifiers detected in the response.",
            )

        top_weight = max(self._severity_weight.get(f.subtype, 0.4) for f in findings)
        # Multiple distinct entity types compound the exposure slightly.
        distinct_types = {f.subtype for f in findings}
        risk = min(1.0, top_weight + 0.05 * (len(distinct_types) - 1))

        critical_types = sorted(
            {f.subtype for f in findings if f.subtype in self._critical_entities}
        )
        # Any finding escalated to CRITICAL by context (e.g. an identifier
        # in an outward-facing response) also counts.
        critical_types = sorted(
            set(critical_types)
            | {f.subtype for f in findings if f.severity == Severity.CRITICAL}
        )
        # A combination that together identifies and contacts a specific
        # person is treated as critical even without a government ID / card.
        contact_identifiers = distinct_types & {"email", "phone", "person_name", "postal_address"}
        full_contact_disclosure = len(contact_identifiers) >= 2
        if full_contact_disclosure and "full_contact_profile" not in critical_types:
            critical_types.append("full_contact_profile")
            risk = max(risk, 0.85)
        if is_external and critical_types:
            risk = max(risk, 0.9)
        contains_critical = bool(critical_types)

        confidence = max(f.confidence for f in findings)
        entity_summary = ", ".join(sorted(distinct_types))
        explanation = (
            f"Detected {len(findings)} PII item(s) across {len(distinct_types)} type(s): "
            f"{entity_summary}."
        )
        if contains_critical:
            explanation += (
                f" Includes critical identifier type(s): {', '.join(critical_types)}."
            )

        return PIIResult(
            pii_risk=round(risk, 4),
            confidence=round(confidence, 4),
            contains_critical_pii=contains_critical,
            critical_types=critical_types,
            findings=findings,
            redacted_response=redacted,
            explanation=explanation,
        )

    @staticmethod
    def redact(text: str, findings: list[Finding]) -> str:
        """Return ``text`` with every finding's span replaced by its redaction."""
        pieces: list[str] = []
        cursor = 0
        for finding in sorted(findings, key=lambda f: f.span[0] if f.span else 0):
            if finding.span is None:
                continue
            start, end = finding.span
            if start < cursor:
                continue
            pieces.append(text[cursor:start])
            pieces.append(finding.redacted_text)
            cursor = end
        pieces.append(text[cursor:])
        return "".join(pieces)

    def _severity_for(self, entity: str) -> Severity:
        weight = self._severity_weight.get(entity, 0.4)
        if entity in self._critical_entities or weight >= 0.9:
            return Severity.CRITICAL
        if weight >= 0.7:
            return Severity.HIGH
        if weight >= 0.5:
            return Severity.MEDIUM
        return Severity.LOW

    @staticmethod
    def _confidence_for(entity: str) -> float:
        return {
            "government_id": 0.9,
            "credit_card": 0.95,
            "financial_account": 0.85,
            "account_id": 0.9,
            "employee_id": 0.9,
            "email": 0.95,
            "phone": 0.8,
            "postal_address": 0.6,
            "person_name": 0.55,
        }.get(entity, 0.7)
