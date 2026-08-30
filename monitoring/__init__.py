"""
ControlPlane.ai — Enterprise Observability / Risk Monitoring (Phase 5).

An operational monitoring layer for an enterprise operator to answer
questions like:

    "What is happening across our AI traffic?"
    "Are risks increasing?  Which applications / models produce the most risk?"
    "How often are we using FAST vs DEEP verification?"
    "How many interactions reach a human?  Which policy rules fire?"

Design boundaries
-----------------
* The ONLY data source is real :class:`decision.schemas.DecisionTrace`
  records produced by the production pipeline.
* Monitoring is a pure aggregation pass — O(number_of_traces). It NEVER
  re-runs a detector, the decision engine, or verification.
* Monitoring NEVER reads ground truth or the evaluation-only labels that
  live on ``EvaluationCase``. It observes production decisions; it does
  not judge whether they were correct — that is the evaluation
  subsystem's job.
* A metric whose denominator is zero is ``None``, never a misleading ``0``.
* Malformed records are reported (data-quality section), not silently
  dropped.
"""

from monitoring.schemas import (
    MonitoringReport,
    MonitoringWindow,
    OperationalMonitoringReport,
)

__all__ = [
    "MonitoringReport",
    "MonitoringWindow",
    "OperationalMonitoringReport",
]

# Phase 8: OperationalMonitor lives in monitoring.engine (imported lazily by
# callers) so that ``python -m monitoring`` and ``import monitoring`` stay light.
