"""Synthetic live traffic: replays generator rows as /predict calls and reports outcomes as feedback."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterator, Sequence

import httpx

from foundrate.synthetic import Shift, World, generate_order_lines


@dataclass
class Sample:
    store_id: int
    item_id: int
    order_hour: int
    quantity: int
    found: int


class TrafficSource:
    """Endless stream of labelled requests starting at `start`, one synthetic day at a time."""

    def __init__(self, world: World, start: date, shifts: Sequence[Shift] = ()):
        self.world, self.day, self.shifts = world, start, tuple(shifts)
        self._buf: list[Sample] = []

    def _refill(self) -> None:
        df = generate_order_lines(self.world, self.day, 1, self.shifts)
        self.day += timedelta(days=1)
        self._buf = [Sample(*r) for r in df[["store_id", "item_id", "order_hour", "quantity", "found"]]
                     .itertuples(index=False, name=None)][::-1]

    def take(self, n: int) -> Iterator[Sample]:
        for _ in range(n):
            if not self._buf:
                self._refill()
            yield self._buf.pop()


@dataclass
class Outcome:
    version: str | None
    status: int
    latency_s: float
    score: float | None
    found: int


@dataclass
class TrafficResult:
    outcomes: list[Outcome] = field(default_factory=list)


def send_traffic(client: httpx.Client, samples: Iterator[Sample], send_feedback: bool = True) -> list[Outcome]:
    """Send each sample to /predict (and its true outcome to /feedback). Works with TestClient too."""
    out = []
    for s in samples:
        rid = uuid.uuid4().hex
        t0 = time.perf_counter()
        r = client.post("/predict", json={"store_id": s.store_id, "item_id": s.item_id, "order_hour": s.order_hour,
                                          "quantity": s.quantity, "request_id": rid})
        dt = time.perf_counter() - t0
        ok = r.status_code == 200
        out.append(Outcome(r.headers.get("X-Model-Version"), r.status_code, dt,
                           r.json()["found_probability"] if ok else None, s.found))
        if ok and send_feedback:
            client.post("/feedback", json={"request_id": rid, "found": bool(s.found)})
    return out
