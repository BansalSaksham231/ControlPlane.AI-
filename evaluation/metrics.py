"""
Pure evaluation metric helpers.

No ground-truth policy here — just arithmetic over label/prediction
lists. All functions are deterministic and dependency-free.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, float]:
    """Precision / recall / F1 / FPR / FNR / accuracy for a boolean classifier."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "support": len(y_true),
        "positives": tp + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "accuracy": round(accuracy, 4),
    }


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None
) -> dict[str, object]:
    """Return a labelled confusion matrix (rows = true, cols = predicted)."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        matrix[index[t]][index[p]] += 1
    exact = sum(matrix[i][i] for i in range(len(labels)))
    return {
        "labels": list(labels),
        "matrix": matrix,
        "exact_match": exact,
        "total": len(y_true),
        "accuracy": round(exact / len(y_true), 4) if y_true else 0.0,
    }


def threshold_sweep(
    y_true: Sequence[bool],
    scores: Sequence[float],
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, float]]:
    """Binary metrics at each threshold (score >= threshold => positive)."""
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(1, 20)]
    out: list[dict[str, float]] = []
    for threshold in thresholds:
        preds = [score >= threshold for score in scores]
        row = binary_metrics(y_true, preds)
        row["threshold"] = threshold
        out.append(row)
    return out


def best_threshold(
    y_true: Sequence[bool],
    scores: Sequence[float],
    thresholds: Sequence[float] | None = None,
    objective: str = "f1",
) -> dict[str, float]:
    sweep = threshold_sweep(y_true, scores, thresholds)
    return max(sweep, key=lambda row: row[objective]) if sweep else {}


def distribution(values: Sequence[str]) -> dict[str, int]:
    return dict(Counter(values))


def rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0
