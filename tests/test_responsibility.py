"""Unified Responsibility Detector tests: PII, toxicity, bias, fusion, redaction."""

from __future__ import annotations

import pytest

from detectors.responsibility.detector import ResponsibilityDetector, detect_responsibility
from detectors.responsibility.schemas import ResponsibilityCategory, ResponsibilityResult, Severity


@pytest.fixture(scope="module")
def detector() -> ResponsibilityDetector:
    return ResponsibilityDetector()


CLEAN = "You are eligible for a refund within 30 business days as long as the item is unused."
PII_TEXT = (
    "The contact details on file for account ACC-227763 are: Karan Mehta, "
    "email karan.mehta@example-test.com, phone +91-940847221."
)
TOXIC_TEXT = "Honestly, this complaint is a waste of our time - you should have checked yourself."
BIAS_TEXT = (
    "I would recommend prioritizing the male candidate, since men are generally "
    "a better fit for hands-on technical work."
)


# ---------------------------------------------------------------- clean

def test_clean_response_has_no_findings(detector):
    result = detector.detect(CLEAN)
    assert result.overall_responsibility_risk == 0.0
    assert result.findings == []
    assert result.redacted_response == CLEAN
    assert result.contains_critical_pii is False


# ---------------------------------------------------------------- PII

def test_pii_email_and_phone_detected(detector):
    result = detector.detect(PII_TEXT)
    subtypes = {f.subtype for f in result.pii.findings}
    assert "email" in subtypes
    assert "phone" in subtypes
    assert result.pii_risk >= 0.6


def test_pii_redaction_preserves_audit_record(detector):
    result = detector.detect(PII_TEXT)
    assert "karan.rao" not in result.redacted_response
    assert "karan.mehta@example-test.com" not in result.redacted_response
    assert "REDACTED" in result.redacted_response
    # Raw match is still available on the finding for the audit trail.
    assert any("@example-test.com" in f.matched_text for f in result.pii.findings)


def test_pii_full_contact_profile_is_critical(detector):
    result = detector.detect(PII_TEXT)
    assert result.contains_critical_pii is True
    assert result.critical_pii_types


def test_government_id_and_card_are_critical(detector):
    result = detector.detect(
        "Records show SSN 123-45-6789 and card 4111 1111 1111 1111 on file."
    )
    assert result.contains_critical_pii is True
    assert result.pii_risk >= 0.9
    assert "123-45-6789" not in result.redacted_response


def test_pii_spans_are_present(detector):
    result = detector.detect(PII_TEXT)
    for finding in result.pii.findings:
        assert finding.span is not None
        start, end = finding.span
        assert 0 <= start < end <= len(PII_TEXT)


# ---------------------------------------------------------------- toxicity

def test_toxicity_detected_with_category(detector):
    result = detector.detect(TOXIC_TEXT)
    assert result.toxicity_risk > 0.5
    assert result.toxicity.categories
    assert all(f.category == ResponsibilityCategory.TOXICITY for f in result.toxicity.findings)


def test_toxicity_detection_separate_from_policy(detector):
    """The detector reports risk/evidence only; it never returns a decision/tier."""
    result = detector.detect(TOXIC_TEXT)
    assert not hasattr(result, "decision")
    assert not hasattr(result, "intervention")


# ---------------------------------------------------------------- bias

def test_bias_reported_as_signal_not_verdict(detector):
    result = detector.detect(BIAS_TEXT)
    assert result.bias_risk > 0.0
    # Never certain.
    assert result.bias_risk <= 0.8
    assert "signal" in result.bias.explanation.lower()


# ---------------------------------------------------------------- fusion

def test_multiple_simultaneous_findings(detector):
    text = PII_TEXT + " " + BIAS_TEXT + " " + TOXIC_TEXT
    result = detector.detect(text)
    categories = {f.category for f in result.findings}
    assert ResponsibilityCategory.PII in categories
    assert ResponsibilityCategory.BIAS in categories
    assert ResponsibilityCategory.TOXICITY in categories
    # A severe dimension is not diluted by the others.
    assert result.overall_responsibility_risk >= 0.8 * max(
        result.pii_risk, result.toxicity_risk, result.bias_risk
    )


def test_single_severe_dimension_not_diluted(detector):
    result = detector.detect(PII_TEXT)  # high PII, zero toxicity/bias
    assert result.overall_responsibility_risk >= 0.6


def test_subresults_inspectable(detector):
    result = detector.detect(PII_TEXT)
    assert result.pii.pii_risk == result.pii_risk
    assert result.toxicity.toxicity_risk == result.toxicity_risk
    assert result.bias.bias_signal == result.bias_risk


def test_latency_recorded(detector):
    result = detector.detect(PII_TEXT)
    assert result.latency_ms >= 0.0


def test_deterministic(detector):
    a = detector.detect(PII_TEXT + BIAS_TEXT).model_dump()
    b = detector.detect(PII_TEXT + BIAS_TEXT).model_dump()
    a.pop("latency_ms")
    b.pop("latency_ms")
    assert a == b


def test_convenience_wrapper():
    result = detect_responsibility_str()
    assert isinstance(result, ResponsibilityResult)


def detect_responsibility_str() -> ResponsibilityResult:
    from data.schemas import ActionType, Application, Interaction, ModelName, UserType

    interaction = Interaction(
        interaction_id="INT-R-1",
        timestamp="2026-08-21T12:00:00",
        application=Application.CUSTOMER_SUPPORT,
        user_type=UserType.EXTERNAL_CUSTOMER,
        model=ModelName.GPT_4O_MINI,
        session_id="S1",
        prompt="p",
        context="c",
        response=PII_TEXT,
        tokens_in=10,
        tokens_out=10,
        latency_ms=100.0,
        tool_calls=0,
        retry_count=0,
        action_type=ActionType.INFORMATION,
        action_amount_inr=0.0,
        affected_entities=1,
    )
    return detect_responsibility(interaction)


def test_no_ground_truth_leakage():
    from data.schemas import Interaction

    assert "ground_truth_pii" not in Interaction.model_fields
