"""
Lightweight wall-clock timing helpers for detector latency breakdowns.

Latency naturally varies run to run, so timing values are never used in
determinism assertions — they are reported for observability only.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Stopwatch:
    """Accumulates named stage durations (milliseconds) for one detector run."""

    def __init__(self) -> None:
        self._stages: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block of work and record it under ``name`` (ms)."""
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._stages[name] = self._stages.get(name, 0.0) + elapsed_ms

    def record(self, name: str, milliseconds: float) -> None:
        self._stages[name] = self._stages.get(name, 0.0) + max(0.0, milliseconds)

    def get(self, name: str) -> float:
        return round(self._stages.get(name, 0.0), 3)

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000.0, 3)

    def as_dict(self) -> dict[str, float]:
        return {name: round(value, 3) for name, value in self._stages.items()}


def clamp01(value: float) -> float:
    """Clamp ``value`` into the closed unit interval [0, 1]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)
