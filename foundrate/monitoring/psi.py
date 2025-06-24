"""Population Stability Index, reference distributions and rolling AUC."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-4


def make_reference(values: Sequence[float], bins: int = 10) -> dict[str, list[float]]:
    """Quantile bin edges + expected proportions from reference data."""
    v = np.asarray(values, dtype="float64")
    edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
    return {"edges": [float(e) for e in edges], "props": [float(p) for p in _proportions(v, edges)]}


def _proportions(values: np.ndarray, edges: np.ndarray | Sequence[float]) -> np.ndarray:
    idx = np.searchsorted(np.asarray(edges), values, side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype("float64")
    return counts / max(counts.sum(), 1.0)


def psi(reference: dict[str, Any], values: Sequence[float]) -> float:
    """PSI = sum((actual - expected) * ln(actual / expected)) over the reference bins."""
    v = np.asarray(values, dtype="float64")
    if v.size == 0:
        return 0.0
    expected = np.clip(np.asarray(reference["props"]), EPS, None)
    actual = np.clip(_proportions(v, reference["edges"]), EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def safe_auc(y: Sequence[int], score: Sequence[float]) -> float | None:
    y = np.asarray(y)
    if y.size == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def rolling_auc(y: Sequence[int], score: Sequence[float], window: int, step: int | None = None) -> list[float]:
    """AUC over sliding windows of the (time-ordered) labelled events; empty if too few events."""
    y, score = np.asarray(y), np.asarray(score)
    step = step or max(window // 2, 1)
    out = []
    for end in range(window, len(y) + 1, step):
        a = safe_auc(y[end - window : end], score[end - window : end])
        if a is not None:
            out.append(a)
    return out
