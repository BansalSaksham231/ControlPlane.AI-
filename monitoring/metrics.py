"""
Pure metric helpers for the observability layer.

Every function here is deterministic and side-effect free. They operate
on plain numbers / lists extracted from ``DecisionTrace`` records — no
detector calls, no engine calls, no I/O, no ground truth.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Iterable, Sequence


def mean_or_none(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` when there is nothing to average."""
    values = list(values)
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def rate_or_none(numerator: int, denominator: int) -> float | None:
    """``numerator / denominator`` rounded, or ``None`` when the denominator is zero."""
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def percentile_or_none(values: Sequence[float], pct: float) -> float | None:
    """
    Nearest-rank percentile (``pct`` in 0..100), or ``None`` when empty.

    Deterministic: sorts a copy and picks a fixed index. ``pct=50`` on an
    even-length list returns the lower-middle element (no interpolation),
    which keeps the result exactly representable and reproducible.
    """
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 6)
    frac = max(0.0, min(1.0, pct / 100.0))
    idx = int(round(frac * (len(values) - 1)))
    return round(values[idx], 6)


def max_or_none(values: Sequence[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return round(max(values), 6)


def count_by(items: Iterable[str]) -> dict[str, int]:
    """Frequency map, insertion-ordered by descending count then name."""
    counts = Counter(items)
    return {
        key: counts[key]
        for key in sorted(counts, key=lambda k: (-counts[k], k))
    }


def truncate_timestamp(ts: datetime, granularity: str) -> datetime:
    """
    Floor ``ts`` to the start of its hour or day. Used only for trend
    bucketing; never reads the system clock.
    """
    if granularity == "daily":
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    # default: hourly
    return ts.replace(minute=0, second=0, microsecond=0)


def bucket_index(value: float, bucket_bounds: Sequence[tuple[float, float]]) -> int:
    """
    Index of the first bucket whose ``max`` ``value`` is below. If it is
    below none (e.g. a risk of exactly 1.0 with a final max of 1.0), it is
    placed in the last bucket. ``bucket_bounds`` must be contiguous and
    ascending.
    """
    for i, (_lo, hi) in enumerate(bucket_bounds):
        if value < hi:
            return i
    return len(bucket_bounds) - 1
