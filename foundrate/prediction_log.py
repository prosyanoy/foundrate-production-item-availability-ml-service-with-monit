"""Append-only log of served predictions (+ later feedback labels) that the monitor reads."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd


class PredictionLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS predictions (request_id TEXT PRIMARY KEY, ts REAL, store_id INTEGER, "
            "item_id INTEGER, model_version TEXT, score REAL, features TEXT, features_found INTEGER, label INTEGER)"
        )
        self._con.commit()

    def log(self, request_id: str, store_id: int, item_id: int, version: str, score: float,
            features: dict[str, float], features_found: bool) -> None:
        with self._lock, self._con:
            self._con.execute(
                "INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (request_id, time.time(), store_id, item_id, version, score, json.dumps(features), int(features_found)),
            )

    def set_label(self, request_id: str, found: bool) -> bool:
        with self._lock, self._con:
            cur = self._con.execute("UPDATE predictions SET label = ? WHERE request_id = ?", (int(found), request_id))
            return cur.rowcount > 0

    def recent(self, n: int, model_version: str | None = None) -> pd.DataFrame:
        """Most recent n predictions (oldest first), with feature columns expanded."""
        q = "SELECT rowid, request_id, ts, store_id, item_id, model_version, score, features, label FROM predictions"
        args: list = []
        if model_version:
            q += " WHERE model_version = ?"
            args.append(model_version)
        q += " ORDER BY rowid DESC LIMIT ?"
        with self._lock:
            rows = self._con.execute(q, [*args, n]).fetchall()
        cols = ["rowid", "request_id", "ts", "store_id", "item_id", "model_version", "score", "features", "label"]
        df = pd.DataFrame(rows[::-1], columns=cols)
        if df.empty:
            return df
        feats = pd.DataFrame([json.loads(f) for f in df.pop("features")], index=df.index)
        return pd.concat([df, feats], axis=1)

    def close(self) -> None:
        self._con.close()
