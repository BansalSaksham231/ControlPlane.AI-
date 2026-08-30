"""Feedback store tests — create, persist, retrieve, aggregate."""

from __future__ import annotations

from datetime import datetime

import pytest

from data.schemas import InterventionTier
from feedback.schemas import FeedbackOutcome
from feedback.store import FeedbackStore

TS = datetime(2026, 8, 21, 12, 0, 0)


@pytest.fixture()
def store() -> FeedbackStore:
    return FeedbackStore()


def test_submit_approved(store):
    record = store.submit(
        interaction_id="INT-1",
        system_decision=InterventionTier.VERIFY,
        reviewer_decision=InterventionTier.VERIFY,
        reason="looks right",
        reviewer="alice",
        timestamp=TS,
    )
    assert record.outcome == FeedbackOutcome.APPROVED
    assert record.human_override is False
    assert record.feedback_id.startswith("FB-")


def test_submit_override_is_modified(store):
    record = store.submit(
        interaction_id="INT-2",
        system_decision=InterventionTier.ANNOTATE,
        reviewer_decision=InterventionTier.HUMAN_REVIEW,
        reason="should have escalated",
        timestamp=TS,
    )
    assert record.outcome == FeedbackOutcome.MODIFIED
    assert record.human_override is True


def test_submit_explicit_rejected(store):
    record = store.submit(
        interaction_id="INT-3",
        system_decision=InterventionTier.BLOCK,
        outcome="rejected",
        reason="false positive block",
        timestamp=TS,
    )
    assert record.outcome == FeedbackOutcome.REJECTED
    assert record.human_override is True


def test_retrieve_by_id_and_interaction(store):
    r = store.submit(interaction_id="INT-4", system_decision="ALLOW", reviewer_decision="ALLOW", timestamp=TS)
    assert store.get(r.feedback_id) == r
    assert store.for_interaction("INT-4") == [r]
    assert store.get("FB-999999") is None


def test_aggregate(store):
    store.submit(interaction_id="A", system_decision="ALLOW", reviewer_decision="ALLOW", timestamp=TS)
    store.submit(interaction_id="B", system_decision="VERIFY", reviewer_decision="HUMAN_REVIEW", timestamp=TS)
    store.submit(interaction_id="C", system_decision="BLOCK", reviewer_decision="VERIFY", timestamp=TS)
    agg = store.aggregate()
    assert agg.total == 3
    assert agg.by_outcome["approved"] == 1
    assert agg.by_outcome["modified"] == 2
    assert agg.override_count == 2
    assert agg.override_rate == pytest.approx(2 / 3)
    assert agg.escalations == 1      # VERIFY -> HUMAN_REVIEW
    assert agg.de_escalations == 1   # BLOCK -> VERIFY
    assert any(c.count == 1 for c in agg.decision_confusion)


def test_empty_aggregate(store):
    agg = store.aggregate()
    assert agg.total == 0
    assert agg.override_rate == 0.0
    assert agg.approval_rate == 0.0


def test_jsonl_persistence(tmp_path):
    path = tmp_path / "feedback.jsonl"
    store1 = FeedbackStore(path=str(path))
    store1.submit(interaction_id="INT-9", system_decision="VERIFY", reviewer_decision="BLOCK", reason="x", timestamp=TS)
    store1.submit(interaction_id="INT-10", system_decision="ALLOW", reviewer_decision="ALLOW", timestamp=TS)

    store2 = FeedbackStore(path=str(path))
    assert len(store2) == 2
    assert store2.for_interaction("INT-9")[0].reviewer_decision == InterventionTier.BLOCK
    # new ids continue, do not collide
    new = store2.submit(interaction_id="INT-11", system_decision="ALLOW", timestamp=TS)
    assert new.feedback_id == "FB-000003"


def test_deterministic_ids(store):
    a = store.submit(interaction_id="x", system_decision="ALLOW", timestamp=TS)
    b = store.submit(interaction_id="y", system_decision="ALLOW", timestamp=TS)
    assert a.feedback_id == "FB-000001"
    assert b.feedback_id == "FB-000002"
