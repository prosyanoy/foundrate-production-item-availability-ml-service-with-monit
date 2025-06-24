"""Daily pipeline steps (shared by the Airflow DAG, the CLI and the tests)."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Sequence

from foundrate.config import Settings
from foundrate.feature_store import OfflineStore, OnlineStore
from foundrate.registry import ModelRegistry
from foundrate.synthetic import Shift, World, WorldConfig, generate_order_lines
from foundrate.training import train_and_register, training_as_ofs


def world_for(settings: Settings) -> World:
    return World(WorldConfig(settings.n_stores, settings.n_items, settings.lines_per_day, settings.seed))


def ingest(settings: Settings, ds: date, shifts: Sequence[Shift] = (), start: date | None = None) -> int:
    """Load synthetic order lines for [start, ds) (default: the full lookback window). Idempotent."""
    settings.ensure_dirs()
    start = start or ds - timedelta(days=settings.lookback_days)
    world = world_for(settings)
    offline = OfflineStore(settings.offline_db)
    offline.load_items(world)
    lines = generate_order_lines(world, start, (ds - start).days, tuple(shifts))
    offline.ingest_order_lines(lines, start, ds)
    return len(lines)


def build_features(settings: Settings, ds: date) -> dict[str, Any]:
    """Training snapshots (weekly, labels fully observed before ds) plus the serving snapshot at ds."""
    offline = OfflineStore(settings.offline_db)
    train_dates = training_as_ofs(ds, settings.horizon_days, settings.train_snapshots)
    rows = {d.isoformat(): offline.build_snapshot(d, settings.feature_window_days) for d in [*train_dates, ds]}
    return {"snapshots": rows}


def train(settings: Settings, ds: date, **fit_kwargs: Any) -> dict[str, Any]:
    """Train on the snapshots for `ds` and register a new version. Auto-promotes only the first model."""
    offline = OfflineStore(settings.offline_db)
    registry = ModelRegistry(settings.registry_dir, settings.model_name)
    version, meta = train_and_register(
        offline, registry, ds, settings.horizon_days, settings.train_snapshots, **fit_kwargs
    )
    promoted = False
    if registry.aliases()["production"] is None:
        registry.promote(version, reason="first model")
        promoted = True
    return {"version": version, "metrics": meta["metrics"], "promoted": promoted}


def publish_online(settings: Settings, ds: date) -> int:
    """Push the serving snapshot for `ds` to the online store."""
    offline = OfflineStore(settings.offline_db)
    online = OnlineStore(settings.online_db)
    try:
        return online.publish(offline.snapshot(ds), ds)
    finally:
        online.close()


def run_daily(settings: Settings, ds: date, shifts: Sequence[Shift] = ()) -> dict[str, Any]:
    """The whole DAG, sequentially (used by `foundrate bootstrap` and tests)."""
    n = ingest(settings, ds, shifts)
    feats = build_features(settings, ds)
    result = train(settings, ds)
    rows = publish_online(settings, ds)
    return {"ds": ds.isoformat(), "order_lines": n, **feats, **result, "online_rows": rows}
