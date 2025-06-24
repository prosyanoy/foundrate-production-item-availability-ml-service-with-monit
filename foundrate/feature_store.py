"""Simple feature store: an offline DuckDB table of point-in-time snapshots and an online SQLite KV."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from foundrate.features import ENTITY_FEATURES
from foundrate.synthetic import World

_SQL_DIR = Path(__file__).parent / "sql"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS items (item_id BIGINT PRIMARY KEY, category VARCHAR, is_perishable BIGINT);
CREATE TABLE IF NOT EXISTS order_lines (
    order_id BIGINT, order_date DATE, order_hour BIGINT, store_id BIGINT, item_id BIGINT,
    quantity BIGINT, found BIGINT);
CREATE TABLE IF NOT EXISTS feature_snapshots (
    as_of DATE, store_id BIGINT, item_id BIGINT,
    {', '.join(f'{c} DOUBLE' for c in ENTITY_FEATURES)});
"""


class OfflineStore:
    """Offline table: order lines + versioned feature snapshots, queried with SQL."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            con.execute(_SCHEMA)

    @contextmanager
    def _con(self) -> Iterator[duckdb.DuckDBPyConnection]:
        con = duckdb.connect(str(self.path))
        try:
            yield con
        finally:
            con.close()

    def load_items(self, world: World) -> None:
        items = world.items
        with self._con() as con:
            con.execute("DELETE FROM items")
            con.register("new_items", items)
            con.execute("INSERT INTO items SELECT item_id, category, is_perishable FROM new_items")

    def ingest_order_lines(self, lines: pd.DataFrame, start: date, end: date) -> None:
        """Idempotently replace order lines for [start, end)."""
        with self._con() as con:
            con.execute("DELETE FROM order_lines WHERE order_date >= ? AND order_date < ?", [start, end])
            con.register("new_lines", lines)
            con.execute(
                "INSERT INTO order_lines SELECT order_id, CAST(order_date AS DATE), order_hour, store_id, "
                "item_id, quantity, found FROM new_lines"
            )

    def max_order_date(self) -> date | None:
        with self._con() as con:
            return con.execute("SELECT MAX(order_date) FROM order_lines").fetchone()[0]

    def build_snapshot(self, as_of: date, window_days: int = 30) -> int:
        """Materialise features as of `as_of` (idempotent). Returns the number of rows."""
        sql = (_SQL_DIR / "features.sql").read_text()
        cols = ", ".join(["as_of", "store_id", "item_id", *ENTITY_FEATURES])
        with self._con() as con:
            con.execute("DELETE FROM feature_snapshots WHERE as_of = ?", [as_of])
            con.execute(
                f"INSERT INTO feature_snapshots ({cols}) {sql}",  # noqa: S608
                {"as_of": as_of, "window_days": window_days},
            )
            return con.execute("SELECT COUNT(*) FROM feature_snapshots WHERE as_of = ?", [as_of]).fetchone()[0]

    def snapshot(self, as_of: date) -> pd.DataFrame:
        with self._con() as con:
            return con.execute(
                "SELECT * FROM feature_snapshots WHERE as_of = ? ORDER BY store_id, item_id", [as_of]
            ).df()

    def training_frame(self, as_ofs: list[date], horizon_days: int) -> pd.DataFrame:
        """Labelled rows: order lines in [as_of, as_of+horizon) joined to the snapshot taken at as_of."""
        cols = ", ".join(f"s.{c}" for c in ENTITY_FEATURES)
        parts = []
        with self._con() as con:
            for a in as_ofs:
                parts.append(
                    con.execute(
                        f"""SELECT CAST(? AS DATE) AS as_of, l.order_date, l.store_id, l.item_id,
                                   l.order_hour, l.quantity, l.found, {cols}
                            FROM order_lines l
                            LEFT JOIN feature_snapshots s
                              ON s.as_of = CAST(? AS DATE) AND s.store_id = l.store_id AND s.item_id = l.item_id
                            WHERE l.order_date >= CAST(? AS DATE) AND l.order_date < CAST(? AS DATE)
                            ORDER BY l.order_date, l.order_id, l.item_id""",  # noqa: S608
                        [a, a, a, a + timedelta(days=horizon_days)],
                    ).df()
                )
        return pd.concat(parts, ignore_index=True)


class OnlineStore:
    """Online key-value lookup (SQLite). Key = (store_id, item_id); value = JSON of entity features."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS online_features ("
            "store_id INTEGER, item_id INTEGER, payload TEXT NOT NULL, PRIMARY KEY (store_id, item_id))"
        )
        self._con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self._con.commit()

    def publish(self, snapshot: pd.DataFrame, as_of: date) -> int:
        """Atomically replace the online table with a snapshot."""
        rows = [
            (int(r.store_id), int(r.item_id), json.dumps({c: _clean(getattr(r, c)) for c in ENTITY_FEATURES}))
            for r in snapshot.itertuples(index=False)
        ]
        with self._lock, self._con:
            self._con.execute("DELETE FROM online_features")
            self._con.executemany("INSERT INTO online_features VALUES (?, ?, ?)", rows)
            self._con.execute("INSERT OR REPLACE INTO meta VALUES ('as_of', ?)", (as_of.isoformat(),))
        return len(rows)

    def get(self, store_id: int, item_id: int) -> dict[str, float | None] | None:
        with self._lock:
            row = self._con.execute(
                "SELECT payload FROM online_features WHERE store_id = ? AND item_id = ?", (store_id, item_id)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def as_of(self) -> date | None:
        with self._lock:
            row = self._con.execute("SELECT value FROM meta WHERE key = 'as_of'").fetchone()
        return None if row is None else date.fromisoformat(row[0])

    def close(self) -> None:
        self._con.close()


def _clean(v: object) -> float | None:
    if v is None or pd.isna(v):
        return None
    return float(v)
