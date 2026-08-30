"""
Generate the deterministic synthetic datasets.

    python run_generator.py

Writes ``data/generated/interactions.csv`` (production-shaped) and
``data/generated/evaluation_cases.csv`` (interactions + evaluation-only
ground truth). Reproducible under ``seed`` in ``config/settings.yaml``.
"""

from __future__ import annotations

from data.generator import run

if __name__ == "__main__":
    run()
