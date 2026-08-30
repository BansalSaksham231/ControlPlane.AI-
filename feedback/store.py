"""
Feedback store.

A clean data contract plus simple aggregation for human review outcomes.
It does NOT retrain anything — it captures the signal a future
calibration / evaluation pass would consume.

Storage is an in-memory list, optionally mirrored to a JSONL file so
feedback survives a restart and can be analysed offline.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Iterable

from data.schemas import InterventionTier
from feedback.schemas import (
    DecisionConfusionCell,
    FeedbackAggregate,
    FeedbackOutcome,
    FeedbackRecord,
)
from policy.schemas import TIER_RANK

STORE_NAME = "feedback"


class FeedbackStore:
    def __init__(self, path: str | None = None) -> None:
        self._path = path
        self._records: list[FeedbackRecord] = []
        self._lock = threading.Lock()
        self._counter = 0
        if path and os.path.exists(path):
            self._load()

    # ------------------------------------------------------------------

    def submit(
        self,
        *,
        interaction_id: str,
        system_decision: InterventionTier | str,
        reviewer_decision: InterventionTier | str | None = None,
        outcome: FeedbackOutcome | str | None = None,
        actual_outcome: str | None = None,
        reason: str = "",
        reviewer: str | None = None,
        timestamp: datetime | None = None,
    ) -> FeedbackRecord:
        system_tier = InterventionTier(system_decision)
        reviewer_tier = (
            InterventionTier(reviewer_decision) if reviewer_decision is not None else None
        )

        resolved_outcome = self._resolve_outcome(outcome, system_tier, reviewer_tier)
        human_override = (
            reviewer_tier is not None and reviewer_tier != system_tier
        ) or resolved_outcome in (FeedbackOutcome.MODIFIED, FeedbackOutcome.REJECTED)

        with self._lock:
            self._counter += 1
            record = FeedbackRecord(
                feedback_id=f"FB-{self._counter:06d}",
                interaction_id=interaction_id,
                system_decision=system_tier,
                reviewer_decision=reviewer_tier,
                human_override=human_override,
                outcome=resolved_outcome,
                actual_outcome=actual_outcome,
                reason=reason,
                reviewer=reviewer,
                timestamp=timestamp or datetime.now(timezone.utc),
            )
            self._records.append(record)
            self._append(record)
        return record

    # ------------------------------------------------------------------

    def get(self, feedback_id: str) -> FeedbackRecord | None:
        with self._lock:
            return next((r for r in self._records if r.feedback_id == feedback_id), None)

    def for_interaction(self, interaction_id: str) -> list[FeedbackRecord]:
        with self._lock:
            return [r for r in self._records if r.interaction_id == interaction_id]

    def all(self) -> list[FeedbackRecord]:
        with self._lock:
            return list(self._records)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    # ------------------------------------------------------------------

    def aggregate(self) -> FeedbackAggregate:
        with self._lock:
            records = list(self._records)

        total = len(records)
        by_outcome: dict[str, int] = {o.value: 0 for o in FeedbackOutcome}
        by_system: dict[str, int] = {}
        confusion: dict[tuple[str, str], int] = {}
        override_count = escalations = de_escalations = 0

        for record in records:
            by_outcome[record.outcome.value] += 1
            by_system[record.system_decision.value] = (
                by_system.get(record.system_decision.value, 0) + 1
            )
            if record.human_override:
                override_count += 1
            if record.reviewer_decision is not None:
                key = (record.system_decision.value, record.reviewer_decision.value)
                confusion[key] = confusion.get(key, 0) + 1
                delta = (
                    TIER_RANK[record.reviewer_decision]
                    - TIER_RANK[record.system_decision]
                )
                if delta > 0:
                    escalations += 1
                elif delta < 0:
                    de_escalations += 1

        return FeedbackAggregate(
            total=total,
            by_outcome=by_outcome,
            by_system_decision=by_system,
            override_count=override_count,
            override_rate=(override_count / total) if total else 0.0,
            approval_rate=(by_outcome["approved"] / total) if total else 0.0,
            decision_confusion=[
                DecisionConfusionCell(
                    system_decision=s, reviewer_decision=r, count=c
                )
                for (s, r), c in sorted(confusion.items())
            ],
            escalations=escalations,
            de_escalations=de_escalations,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_outcome(
        outcome: FeedbackOutcome | str | None,
        system_tier: InterventionTier,
        reviewer_tier: InterventionTier | None,
    ) -> FeedbackOutcome:
        if outcome is not None:
            return FeedbackOutcome(outcome)
        if reviewer_tier is None or reviewer_tier == system_tier:
            return FeedbackOutcome.APPROVED
        return FeedbackOutcome.MODIFIED

    def _append(self, record: FeedbackRecord) -> None:
        if not self._path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def _load(self) -> None:
        with open(self._path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = FeedbackRecord.model_validate_json(line)
                self._records.append(record)
                self._counter = max(
                    self._counter, int(record.feedback_id.split("-")[1])
                )

    def load_records(self, records: Iterable[FeedbackRecord]) -> None:
        with self._lock:
            for record in records:
                self._records.append(record)
                self._counter += 1
