"""Feature definitions and the single code path that turns raw features into model input.

Training (`assemble_frame`) and serving (`assemble_vector`) both apply the same default-filling
rules and the same column order; tests/test_parity.py guarantees they agree.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

ENTITY_FEATURES = [
    "si_rate_30d",
    "si_n_30d",
    "si_rate_7d",
    "si_n_7d",
    "days_since_miss",
    "store_rate_7d",
    "store_cat_rate_30d",
    "item_rate_30d",
    "is_perishable",
]
REQUEST_FEATURES = ["order_hour", "quantity"]
FEATURE_COLUMNS = ENTITY_FEATURES + REQUEST_FEATURES


def compute_defaults(frame: pd.DataFrame) -> dict[str, float]:
    """Fallback value per entity feature (mean), used for unseen store/item pairs and nulls."""
    out = {}
    for c in ENTITY_FEATURES:
        m = frame[c].mean()
        out[c] = 0.0 if m is None or math.isnan(m) else float(m)
    return out


def assemble_frame(frame: pd.DataFrame, defaults: Mapping[str, float]) -> pd.DataFrame:
    """Offline path: model-ready feature frame (column order = FEATURE_COLUMNS)."""
    x = frame[FEATURE_COLUMNS].astype("float64").copy()
    for c in ENTITY_FEATURES:
        x[c] = x[c].fillna(defaults[c])
    return x


def assemble_vector(
    entity_row: Mapping[str, float | None] | None,
    order_hour: int,
    quantity: int,
    defaults: Mapping[str, float],
) -> list[float]:
    """Online path: one model-ready feature vector from an online-store row (or None)."""
    vec = []
    for c in ENTITY_FEATURES:
        v = None if entity_row is None else entity_row.get(c)
        vec.append(float(defaults[c]) if v is None or (isinstance(v, float) and math.isnan(v)) else float(v))
    vec.extend([float(order_hour), float(quantity)])
    return vec


def to_matrix(x: pd.DataFrame) -> np.ndarray:
    return x[FEATURE_COLUMNS].to_numpy(dtype="float64")
