"""
Human approval gate — a system in-memory store.

There is NO authentication and NO production-deployment path. Approving a
recommendation records ``APPROVED_FOR_EVALUATION``; it never writes
``config/settings.yaml`` and never applies the candidate to production.
"""

from __future__ import annotations

import threading

from adaptive.schemas import ApprovalRecord

__all__ = ["ApprovalStore"]

_APPROVED = "APPROVED_FOR_EVALUATION"
_REJECTED = "REJECTED"


class ApprovalStore:
    def __init__(self) -> None:
        self._by_id: dict[str, ApprovalRecord] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def _record(self, recommendation_id: str, decision: str, actor: str, comment: str) -> ApprovalRecord:
        with self._lock:
            self._seq += 1
            rec = ApprovalRecord(
                recommendation_id=recommendation_id,
                decision=decision,
                actor=actor or "reviewer",
                comment=(comment or "").strip()[:400],
                sequence=self._seq,
            )
            self._by_id[recommendation_id] = rec
        return rec

    def approve(self, recommendation_id: str, *, actor: str = "reviewer", comment: str = "") -> ApprovalRecord:
        return self._record(recommendation_id, _APPROVED, actor, comment)

    def reject(self, recommendation_id: str, *, actor: str = "reviewer", comment: str = "") -> ApprovalRecord:
        return self._record(recommendation_id, _REJECTED, actor, comment)

    def get(self, recommendation_id: str) -> ApprovalRecord | None:
        return self._by_id.get(recommendation_id)

    def all(self) -> list[ApprovalRecord]:
        return sorted(self._by_id.values(), key=lambda r: r.sequence)
