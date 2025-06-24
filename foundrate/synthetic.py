"""Deterministic synthetic grocery order-line generator.

Each (store, item) pair has a latent stock level; whether the shopper finds the item depends on
that stock level, the store, the item, the hour of day, the quantity and day-level noise.
`Shift` lets tests/demos change a store's stock pattern from a given date onward.
Rows for a given day depend only on (seed, day), so any date range regenerates identically.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

CATEGORIES = ["produce", "dairy", "bakery", "meat", "frozen", "pantry", "snacks", "household"]
PERISHABLE = {"produce", "dairy", "bakery", "meat"}


@dataclass(frozen=True)
class Shift:
    """From `start` on, `item_fraction` of the items at `store_id` become harder to find."""

    store_id: int
    start: date
    logit_delta: float = 3.0
    item_fraction: float = 0.7


@dataclass(frozen=True)
class WorldConfig:
    n_stores: int = 12
    n_items: int = 80
    lines_per_day: int = 800
    seed: int = 7


class World:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        rng = np.random.default_rng([cfg.seed, 0])
        self.store_effect = rng.normal(0.0, 0.5, cfg.n_stores)
        self.item_effect = rng.normal(0.0, 0.6, cfg.n_items)
        self.stock = rng.normal(0.0, 0.9, (cfg.n_stores, cfg.n_items))
        cats = [CATEGORIES[i % len(CATEGORIES)] for i in range(cfg.n_items)]
        self.items = pd.DataFrame(
            {
                "item_id": np.arange(cfg.n_items),
                "category": cats,
                "is_perishable": [int(c in PERISHABLE) for c in cats],
            }
        )
        pop = 1.0 / (np.arange(cfg.n_items) + 8.0)
        self.item_weights = pop / pop.sum()
        sw = rng.uniform(0.7, 1.3, cfg.n_stores)
        self.store_weights = sw / sw.sum()
        hours = np.exp(-0.5 * ((np.arange(24) - 14.0) / 4.5) ** 2)
        hours[:7] = 0.0
        hours[22:] = 0.0
        self.hour_weights = hours / hours.sum()
        self.perishable = self.items["is_perishable"].to_numpy()

    def shifted_items(self, shift: Shift) -> np.ndarray:
        rng = np.random.default_rng([self.cfg.seed, 1, shift.store_id])
        k = int(round(shift.item_fraction * self.cfg.n_items))
        mask = np.zeros(self.cfg.n_items, dtype=bool)
        mask[rng.choice(self.cfg.n_items, k, replace=False)] = True
        return mask


def _generate_day(world: World, day: date, shifts: tuple[Shift, ...]) -> pd.DataFrame:
    cfg = world.cfg
    rng = np.random.default_rng([cfg.seed, 2, day.toordinal()])
    n = int(rng.poisson(cfg.lines_per_day))
    store = rng.choice(cfg.n_stores, n, p=world.store_weights)
    item = rng.choice(cfg.n_items, n, p=world.item_weights)
    hour = rng.choice(24, n, p=world.hour_weights)
    qty = np.clip(1 + rng.poisson(0.6, n), 1, 6)
    store_noise = rng.normal(0.0, 0.15, cfg.n_stores)
    weekend = -0.2 if day.weekday() >= 5 else 0.0
    logit = (
        1.4
        + world.store_effect[store]
        + world.item_effect[item]
        + world.stock[store, item]
        - 0.35 * world.perishable[item]
        - 0.06 * (hour - 12)
        - 0.15 * (qty - 1)
        + store_noise[store]
        + weekend
    )
    for s in shifts:
        if day >= s.start:
            logit = logit - s.logit_delta * ((store == s.store_id) & world.shifted_items(s)[item])
    found = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))
    return pd.DataFrame(
        {
            "order_id": day.toordinal() * 10_000 + np.arange(n) // 5,
            "order_date": pd.Timestamp(day),
            "order_hour": hour.astype("int64"),
            "store_id": store.astype("int64"),
            "item_id": item.astype("int64"),
            "quantity": qty.astype("int64"),
            "found": found.astype("int64"),
        }
    )


def generate_order_lines(world: World, start: date, days: int, shifts: tuple[Shift, ...] = ()) -> pd.DataFrame:
    frames = [_generate_day(world, start + timedelta(days=k), tuple(shifts)) for k in range(days)]
    return pd.concat(frames, ignore_index=True)
