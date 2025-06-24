"""Runtime configuration, read from environment variables with safe local defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # storage
    home: Path = field(default_factory=lambda: Path(_env("FOUNDRATE_HOME", "var")))
    model_name: str = field(default_factory=lambda: _env("FOUNDRATE_MODEL_NAME", "foundrate"))
    # synthetic world
    n_stores: int = field(default_factory=lambda: int(_env("FOUNDRATE_N_STORES", "12")))
    n_items: int = field(default_factory=lambda: int(_env("FOUNDRATE_N_ITEMS", "80")))
    lines_per_day: int = field(default_factory=lambda: int(_env("FOUNDRATE_LINES_PER_DAY", "800")))
    seed: int = field(default_factory=lambda: int(_env("FOUNDRATE_SEED", "7")))
    # pipeline
    horizon_days: int = 7  # label window after a feature snapshot
    train_snapshots: int = field(default_factory=lambda: int(_env("FOUNDRATE_TRAIN_SNAPSHOTS", "8")))
    feature_window_days: int = 30
    # monitoring thresholds (mirrored in deploy/prometheus/alert_rules.yml)
    psi_threshold: float = field(default_factory=lambda: float(_env("FOUNDRATE_PSI_THRESHOLD", "0.2")))
    store_psi_threshold: float = field(default_factory=lambda: float(_env("FOUNDRATE_STORE_PSI_THRESHOLD", "0.5")))
    auc_drop_threshold: float = field(default_factory=lambda: float(_env("FOUNDRATE_AUC_DROP_THRESHOLD", "0.05")))
    monitor_window: int = field(default_factory=lambda: int(_env("FOUNDRATE_MONITOR_WINDOW", "2000")))
    monitor_min_samples: int = field(default_factory=lambda: int(_env("FOUNDRATE_MONITOR_MIN_SAMPLES", "500")))

    @property
    def offline_db(self) -> Path:
        return self.home / "offline.duckdb"

    @property
    def online_db(self) -> Path:
        return self.home / "online.sqlite"

    @property
    def registry_dir(self) -> Path:
        return self.home / "registry"

    @property
    def prediction_log_db(self) -> Path:
        return self.home / "predictions.sqlite"

    @property
    def alerts_file(self) -> Path:
        return self.home / "alerts.jsonl"

    @property
    def reports_dir(self) -> Path:
        return self.home / "reports"

    @property
    def lookback_days(self) -> int:
        """History needed so the oldest training snapshot still has a full feature window."""
        return self.feature_window_days + self.horizon_days * (self.train_snapshots + 1)

    def ensure_dirs(self) -> None:
        for p in (self.home, self.registry_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)
