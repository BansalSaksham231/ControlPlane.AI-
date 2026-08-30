"""
Calibration Advisor (Phase 4) — OFFLINE / EVALUATION ONLY.

This subsystem simulates alternative threshold configurations against the
synthetic evaluation dataset and reports the safety-vs-cost tradeoff. It
MAY read ground-truth evaluation labels (that is its job). It is NEVER
imported or called by the production decision path
(``decision/`` / ``verification/`` / ``detectors/`` / ``fusion/`` /
``policy/`` / ``consequence/`` / ``criticality/``) and it NEVER mutates
``config/settings.yaml``.
"""
