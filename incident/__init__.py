"""
ControlPlane.ai — Incident Intelligence (Phase 10, Component 1-5).

Transforms individual stored ``DecisionTrace`` records into meaningful
operational **incidents**, groups them into deterministic **clusters /
patterns**, detects operational **drift**, and produces a deterministic
**attribution** ("what appears to be driving this pattern?").

Boundaries (enforced by tests)
------------------------------
* Read-only. It NEVER re-runs a detector, ``DecisionEngine.evaluate``,
  fusion, policy or verification.
* It NEVER imports the evaluation package and NEVER reads any
  ground-truth / evaluation-only label.
* It exposes only structured, already-safe fields — no ``matched_text``,
  no raw response, no raw PII.
* Deterministic: signature-based grouping, stable sorting, no randomness,
  no LLM, no network. Timestamps affect *reporting windows* only, never
  classification.
* Reviewer feedback is a GOVERNANCE SIGNAL, not ground truth, and not
  evidence that any automated decision was incorrect.
* "confidence" in a pattern means confidence in the PATTERN DETECTION,
  never correctness of the underlying AI responses.
"""

from incident.schemas import (
    AttributionResult,
    DriftReport,
    DriftSignal,
    IncidentCluster,
    IncidentIntelligenceReport,
    IncidentPattern,
    IncidentRecord,
    Phase10IncidentConfig,
    ReviewerOverridePattern,
)
from incident.report import build_incident_intelligence
from incident.store import IncidentStore

__all__ = [
    "IncidentRecord",
    "IncidentCluster",
    "IncidentPattern",
    "AttributionResult",
    "DriftReport",
    "DriftSignal",
    "ReviewerOverridePattern",
    "IncidentIntelligenceReport",
    "Phase10IncidentConfig",
    "IncidentStore",
    "build_incident_intelligence",
]
