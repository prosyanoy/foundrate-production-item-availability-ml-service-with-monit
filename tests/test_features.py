from datetime import date, timedelta

import pandas as pd

from foundrate.feature_store import OfflineStore
from foundrate.features import ENTITY_FEATURES
from foundrate.pipeline import world_for

from .conftest import DS


def test_snapshot_uses_only_past_rows(settings):
    """Changing labels on/after as_of must not change that snapshot (no label leakage)."""
    offline = OfflineStore(settings.offline_db)
    as_of = DS - timedelta(days=7)
    before = offline.snapshot(as_of)
    assert len(before) > 0
    with offline._con() as con:
        con.execute("UPDATE order_lines SET found = 1 - found WHERE order_date >= ?", [as_of])
    offline.build_snapshot(as_of, settings.feature_window_days)
    pd.testing.assert_frame_equal(before, offline.snapshot(as_of))


def test_snapshot_values_match_pandas(settings):
    offline = OfflineStore(settings.offline_db)
    snap = offline.snapshot(DS)
    with offline._con() as con:
        lines = con.execute("SELECT * FROM order_lines").df()
    window = lines[(lines.order_date >= pd.Timestamp(DS - timedelta(days=30))) & (lines.order_date < pd.Timestamp(DS))]
    row = snap.iloc[0]
    sub = window[(window.store_id == row.store_id) & (window.item_id == row.item_id)]
    assert row.si_n_30d == len(sub)
    assert abs(row.si_rate_30d - sub.found.mean()) < 1e-9
    recent = sub[sub.order_date >= pd.Timestamp(DS - timedelta(days=7))]
    assert row.si_n_7d == len(recent)
    assert abs(row.store_rate_7d - window[(window.store_id == row.store_id) & (
        window.order_date >= pd.Timestamp(DS - timedelta(days=7)))].found.mean()) < 1e-9


def test_snapshot_columns_and_item_dim(settings):
    snap = OfflineStore(settings.offline_db).snapshot(DS)
    assert set(ENTITY_FEATURES) <= set(snap.columns)
    assert snap[["store_id", "item_id"]].duplicated().sum() == 0
    assert set(snap.is_perishable.unique()) <= {0.0, 1.0}
    assert len(world_for(settings).items) == settings.n_items


def test_ingest_is_idempotent(settings):
    offline = OfflineStore(settings.offline_db)
    with offline._con() as con:
        n = con.execute("SELECT COUNT(*) FROM order_lines").fetchone()[0]
    from foundrate import pipeline

    pipeline.ingest(settings, date(2026, 1, 15))
    with offline._con() as con:
        assert con.execute("SELECT COUNT(*) FROM order_lines").fetchone()[0] == n
