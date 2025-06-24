"""Model training: time-based validation, metrics, drift reference, registration."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from foundrate.feature_store import OfflineStore
from foundrate.features import FEATURE_COLUMNS, assemble_frame, compute_defaults, to_matrix
from foundrate.monitoring.psi import make_reference
from foundrate.registry import ModelRegistry

SEGMENT_FEATURES = ["si_rate_7d"]  # monitored per store, next to the score
MIN_SEGMENT_ROWS = 100


def training_as_ofs(ds: date, horizon_days: int, n: int) -> list[date]:
    """Weekly snapshot dates whose full label window ends on or before `ds` (oldest first)."""
    return [ds - timedelta(days=horizon_days * k) for k in range(n, 0, -1)]


COARSE_FEATURES = {"store_rate_7d", "store_cat_rate_30d", "item_rate_30d"}  # few distinct values per snapshot


def build_reference(x_all: pd.DataFrame, score_valid: np.ndarray, stores_all: np.ndarray) -> dict[str, Any]:
    """Reference distributions over *all* snapshots, so normal week-to-week variation is part of the baseline."""
    ref: dict[str, Any] = {
        "features": {c: make_reference(x_all[c].to_numpy(), bins=5 if c in COARSE_FEATURES else 10)
                     for c in FEATURE_COLUMNS},
        "score": make_reference(score_valid),
        "segments": {f: {} for f in SEGMENT_FEATURES},
    }
    for s in np.unique(stores_all):
        m = stores_all == s
        if m.sum() < MIN_SEGMENT_ROWS:
            continue
        for f in SEGMENT_FEATURES:
            ref["segments"][f][str(int(s))] = make_reference(x_all[f].to_numpy()[m], bins=5)
    return ref


def fit_model(frame: pd.DataFrame, seed: int = 0, shuffle_labels: bool = False) -> tuple[Any, dict[str, Any]]:
    """Fit on all but the newest snapshot, validate on the newest. Returns (model, metadata)."""
    last = frame["as_of"].max()
    train, valid = frame[frame["as_of"] < last], frame[frame["as_of"] == last]
    defaults = compute_defaults(train)
    x_tr, x_va = assemble_frame(train, defaults), assemble_frame(valid, defaults)
    y_tr, y_va = train["found"].to_numpy(), valid["found"].to_numpy()
    if shuffle_labels:  # fault injection used by the demo to create a regressed candidate
        y_tr = np.random.default_rng(seed).permutation(y_tr)
    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=24, l2_regularization=1.0, random_state=seed
    )
    model.fit(to_matrix(x_tr), y_tr)
    p_va = model.predict_proba(to_matrix(x_va))[:, 1]
    metrics = {
        "auc": float(roc_auc_score(y_va, p_va)),
        "logloss": float(log_loss(y_va, p_va)),
        "brier": float(brier_score_loss(y_va, p_va)),
        "base_rate": float(y_va.mean()),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
    }
    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "defaults": defaults,
        "metrics": metrics,
        "reference": build_reference(pd.concat([x_tr, x_va]), p_va, np.concatenate(
            [train["store_id"].to_numpy(), valid["store_id"].to_numpy()])),
        "train_as_ofs": sorted(str(d.date()) if hasattr(d, "date") else str(d) for d in train["as_of"].unique()),
        "valid_as_of": str(last.date()) if hasattr(last, "date") else str(last),
        "algorithm": "HistGradientBoostingClassifier",
    }
    return model, meta


def train_and_register(
    offline: OfflineStore,
    registry: ModelRegistry,
    ds: date,
    horizon_days: int = 7,
    n_snapshots: int = 8,
    extra_meta: dict[str, Any] | None = None,
    **fit_kwargs: Any,
) -> tuple[str, dict[str, Any]]:
    as_ofs = training_as_ofs(ds, horizon_days, n_snapshots)
    frame = offline.training_frame(as_ofs, horizon_days)
    model, meta = fit_model(frame, **fit_kwargs)
    meta.update(ds=ds.isoformat(), **(extra_meta or {}))
    version = registry.register(model, meta)
    return version, meta
