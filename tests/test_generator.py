"""
Tests for the ControlPlane.ai foundation / data-generation layer.

These tests exercise ``data/generator.py`` against the frozen
``data/schemas.py`` contracts and ``config/settings.yaml`` configuration.
None of those three files are modified by this test module.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Any

import pytest

import data.generator as generator_module
from data.generator import (
    CONFIG_PATH,
    _SYNTHETIC_NAMES,
    compute_consequence_factors,
    generate_evaluation_cases,
    generate_interactions,
    load_config,
)
from data.schemas import (
    Application,
    ActionType,
    Interaction,
    InterventionTier,
    ModelName,
    UserType,
)

# Ground-truth / evaluation-only fields that must never appear on a
# production Interaction instance.
GROUND_TRUTH_FIELD_NAMES = {
    "ground_truth_hallucination",
    "ground_truth_pii",
    "ground_truth_toxicity",
    "ground_truth_bias",
    "ground_truth_cost_anomaly",
    "ground_truth_performance_risk",
    "ground_truth_responsibility_risk",
    "ground_truth_cost_risk",
    "human_review_expected",
    "expected_decision",
    "final_outcome",
    "financial_impact",
    "reversibility",
    "sensitivity",
    "blast_radius",
    "action_automation",
    "consequence_score",
    "grounding_score",
    "confidence",
}

CONSEQUENCE_FACTOR_KEYS = [
    "financial_impact",
    "reversibility",
    "sensitivity",
    "blast_radius",
    "action_automation",
]

GT_FLAG_FIELDS = [
    "ground_truth_hallucination",
    "ground_truth_pii",
    "ground_truth_toxicity",
    "ground_truth_bias",
    "ground_truth_cost_anomaly",
]

EMAIL_RE = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")
# Broad "looks like a phone number" detector: a run of digits/dashes at
# least 8 characters long, starting and ending on a digit. Deliberately
# generic so it would also catch anything that *isn't* the generator's
# own synthetic format.
GENERIC_PHONE_RE = re.compile(r"\+?\d[\d\-]{6,}\d")
SYNTHETIC_PHONE_RE = re.compile(r"^\+91-9\d{9}$")
ALLOWED_EMAIL_DOMAIN = "example-test.com"


# ==================================================
# FIXTURES
# ==================================================


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def generated_data(config):
    """Generate interactions + evaluation cases from a single seeded rng,
    mirroring the exact sequencing ``generator.run()`` uses."""
    rng = random.Random(config["seed"])
    interactions = generate_interactions(config, rng)
    evaluation_cases = generate_evaluation_cases(config, rng)
    return interactions, evaluation_cases


@pytest.fixture(scope="module")
def interactions(generated_data):
    return generated_data[0]


@pytest.fixture(scope="module")
def evaluation_cases(generated_data):
    return generated_data[1]


# ==================================================
# 1-2: RECORD COUNTS
# ==================================================


def test_generated_interaction_count(interactions, config):
    assert len(interactions) == 6000
    assert len(interactions) == config["data_generation"]["num_synthetic_records"]


def test_generated_evaluation_count(evaluation_cases, config):
    assert len(evaluation_cases) == 150
    assert len(evaluation_cases) == config["data_generation"]["num_evaluation_cases"]


# ==================================================
# 3-4: ID UNIQUENESS
# ==================================================


def test_interaction_ids_unique(interactions):
    ids = [interaction.interaction_id for interaction in interactions]
    assert len(ids) == len(set(ids))


def test_evaluation_ids_unique(evaluation_cases):
    ids = [row["interaction_id"] for row in evaluation_cases]
    assert len(ids) == len(set(ids))


# ==================================================
# 5-6: SCHEMA / ENUM VALIDATION
# ==================================================


def test_every_interaction_passes_pydantic_validation(interactions):
    for interaction in interactions:
        assert isinstance(interaction, Interaction)
        # Round-trip through validation again as an extra guarantee.
        revalidated = Interaction.model_validate(interaction.model_dump(mode="json"))
        assert revalidated == interaction


def test_every_enum_value_valid(interactions, evaluation_cases):
    for interaction in interactions:
        assert interaction.application in Application
        assert interaction.user_type in UserType
        assert interaction.model in ModelName
        assert interaction.action_type in ActionType

    valid_applications = {a.value for a in Application}
    valid_user_types = {u.value for u in UserType}
    valid_models = {m.value for m in ModelName}
    valid_action_types = {a.value for a in ActionType}
    valid_decisions = {tier.value for tier in InterventionTier}

    for row in evaluation_cases:
        assert row["application"] in valid_applications
        assert row["user_type"] in valid_user_types
        assert row["model"] in valid_models
        assert row["action_type"] in valid_action_types
        assert row["expected_decision"] in valid_decisions


# ==================================================
# 7-13: FIELD-LEVEL VALUE CONSTRAINTS
# (checked on both production interactions and evaluation rows)
# ==================================================


def test_tokens_in_non_negative(interactions, evaluation_cases):
    for interaction in interactions:
        assert interaction.tokens_in >= 0
    for row in evaluation_cases:
        assert row["tokens_in"] >= 0


def test_tokens_out_non_negative(interactions, evaluation_cases):
    for interaction in interactions:
        assert interaction.tokens_out >= 0
    for row in evaluation_cases:
        assert row["tokens_out"] >= 0


def test_latency_ms_positive(interactions, evaluation_cases):
    for interaction in interactions:
        assert interaction.latency_ms > 0
    for row in evaluation_cases:
        assert row["latency_ms"] > 0


def test_tool_calls_non_negative(interactions, evaluation_cases):
    for interaction in interactions:
        assert interaction.tool_calls >= 0
    for row in evaluation_cases:
        assert row["tool_calls"] >= 0


def test_retry_count_non_negative(interactions, evaluation_cases):
    for interaction in interactions:
        assert interaction.retry_count >= 0
    for row in evaluation_cases:
        assert row["retry_count"] >= 0


def test_action_amount_inr_non_negative(interactions, evaluation_cases):
    for interaction in interactions:
        assert interaction.action_amount_inr >= 0
    for row in evaluation_cases:
        assert row["action_amount_inr"] >= 0


def test_affected_entities_at_least_one(interactions, evaluation_cases):
    for interaction in interactions:
        assert interaction.affected_entities >= 1
    for row in evaluation_cases:
        assert row["affected_entities"] >= 1


# ==================================================
# 14-15: CONSEQUENCE FACTOR RANGES
# ==================================================


def test_consequence_factors_in_unit_range(evaluation_cases):
    for row in evaluation_cases:
        for key in CONSEQUENCE_FACTOR_KEYS:
            assert 0.0 <= row[key] <= 1.0


def test_consequence_score_in_unit_range(evaluation_cases):
    for row in evaluation_cases:
        assert 0.0 <= row["consequence_score"] <= 1.0


# ==================================================
# 16-17: PRODUCTION/EVALUATION BOUNDARY
# ==================================================


def test_grounding_and_confidence_absent_or_none(interactions, evaluation_cases):
    # Interaction (production schema) has no such fields at all.
    for interaction in interactions:
        dumped = interaction.model_dump()
        assert "grounding_score" not in dumped
        assert "confidence" not in dumped
        assert not hasattr(interaction, "grounding_score")
        assert not hasattr(interaction, "confidence")

    # Evaluation rows carry them as explicit, still-unset placeholders.
    for row in evaluation_cases:
        assert row["grounding_score"] is None
        assert row["confidence"] is None


def test_production_interactions_have_no_ground_truth_fields(interactions):
    for interaction in interactions:
        dumped = interaction.model_dump()
        assert GROUND_TRUTH_FIELD_NAMES.isdisjoint(dumped.keys())


# ==================================================
# 18-19: CATEGORY COVERAGE
# ==================================================


def test_all_configured_traffic_categories_represented(config, monkeypatch):
    seen_categories: list[str] = []
    original_build_case = generator_module.build_case

    def _tracking_build_case(rng, category, application, overlay=None):
        seen_categories.append(category)
        return original_build_case(rng, category, application, overlay=overlay)

    monkeypatch.setattr(generator_module, "build_case", _tracking_build_case)

    rng = random.Random(config["seed"])
    generator_module.generate_interactions(config, rng)

    configured_categories = set(config["data_generation"]["traffic_distribution"].keys())
    assert configured_categories == set(seen_categories)
    for category in configured_categories:
        assert seen_categories.count(category) > 0


def test_evaluation_category_counts_match_config_exactly(config, monkeypatch):
    seen_categories: list[str] = []
    original_build_case = generator_module.build_case

    def _tracking_build_case(rng, category, application, overlay=None):
        seen_categories.append(category)
        return original_build_case(rng, category, application, overlay=overlay)

    monkeypatch.setattr(generator_module, "build_case", _tracking_build_case)

    rng = random.Random(config["seed"])
    # Advance the rng through interaction generation exactly as run() does,
    # then reset the tracker so only evaluation-case categories are counted.
    generator_module.generate_interactions(config, rng)
    seen_categories.clear()
    generator_module.generate_evaluation_cases(config, rng)

    evaluation_distribution = config["data_generation"]["evaluation_distribution"]
    counts = Counter(seen_categories)
    for category, expected_count in evaluation_distribution.items():
        assert counts[category] == expected_count
    assert sum(counts.values()) == sum(evaluation_distribution.values())


# ==================================================
# 20: OVERLAPPING-RISK EVALUATION EXAMPLES
# ==================================================


def test_evaluation_contains_overlapping_risk_examples(evaluation_cases):
    overlapping = [
        row
        for row in evaluation_cases
        if sum(bool(row[field]) for field in GT_FLAG_FIELDS) >= 2
    ]
    assert len(overlapping) > 0


# ==================================================
# 21-22: DETERMINISM
# ==================================================


def test_generation_is_reproducible_with_same_seed(config):
    rng1 = random.Random(config["seed"])
    interactions1 = generate_interactions(config, rng1)
    evaluation1 = generate_evaluation_cases(config, rng1)

    rng2 = random.Random(config["seed"])
    interactions2 = generate_interactions(config, rng2)
    evaluation2 = generate_evaluation_cases(config, rng2)

    dumped1 = [i.model_dump(mode="json") for i in interactions1]
    dumped2 = [i.model_dump(mode="json") for i in interactions2]
    assert dumped1 == dumped2
    assert evaluation1 == evaluation2


def test_consequence_calculation_is_deterministic(config):
    weights = config["consequence_weights"]
    result1 = compute_consequence_factors(ActionType.REFUND, 12345.0, 42, weights)
    result2 = compute_consequence_factors(ActionType.REFUND, 12345.0, 42, weights)
    assert result1 == result2
    assert result1.model_dump() == result2.model_dump()

    # And across a fresh set of inputs, for good measure.
    result3 = compute_consequence_factors(
        ActionType.EXTERNAL_COMMUNICATION, 0.0, 350, weights
    )
    result4 = compute_consequence_factors(
        ActionType.EXTERNAL_COMMUNICATION, 0.0, 350, weights
    )
    assert result3 == result4


# ==================================================
# 23: NO REAL-LOOKING PII
# ==================================================


def test_no_real_looking_pii_only_synthetic_examples(interactions, evaluation_cases):
    all_text: list[str] = []
    for interaction in interactions:
        all_text.append(interaction.prompt)
        all_text.append(interaction.context)
        all_text.append(interaction.response)
    for row in evaluation_cases:
        all_text.append(row["prompt"])
        all_text.append(row["context"])
        all_text.append(row["response"])

    synthetic_names_lower = {name.lower() for name in _SYNTHETIC_NAMES}

    found_email = False
    found_phone = False

    for text in all_text:
        for email in EMAIL_RE.findall(text):
            found_email = True
            assert email.lower().endswith("@" + ALLOWED_EMAIL_DOMAIN), (
                f"Unexpected email domain in generated data: {email}"
            )
            local_part = email.split("@")[0]
            candidate_name = local_part.replace(".", " ").lower()
            assert candidate_name in synthetic_names_lower, (
                f"Email local-part does not map to a known synthetic name: {email}"
            )

        for phone in GENERIC_PHONE_RE.findall(text):
            found_phone = True
            assert SYNTHETIC_PHONE_RE.match(phone), (
                f"Unexpected phone-like value in generated data: {phone}"
            )

    # Confirm the PII category actually produced examples to validate.
    assert found_email
    assert found_phone