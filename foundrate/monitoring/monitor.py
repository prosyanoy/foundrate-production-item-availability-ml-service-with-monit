"""Drift + performance monitor: PSI (global and per store) and rolling AUC, with alert evaluation.

Reads the prediction log, compares the live window to the reference distributions stored in the
production model's metadata, evaluates alert rules (mirroring deploy/prometheus/alert_rules.yml),
and exposes everything as Prometheus gauges.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, generate_latest, start_http_server

from foundrate.config import Settings
from foundrate.features import FEATURE_COLUMNS
from foundrate.monitoring.psi import psi, rolling_auc, safe_auc
from foundrate.prediction_log import PredictionLog
from foundrate.registry import ModelRegistry
from foundrate.training import MIN_SEGMENT_ROWS, SEGMENT_FEATURES

log = logging.getLogger("foundrate.monitor")
RUNBOOK = "docs/runbooks/model-drift.md"
MIN_STORE_ROWS = MIN_SEGMENT_ROWS
# store_rate_7d takes one value per store (12 per snapshot): its PSI is dominated by sampling noise, so it is
# reported but never alerts. Store-level drift is caught by the per-store PSI of si_rate_7d instead.
NON_ALERTING_FEATURES = {"store_rate_7d"}


@dataclass
class Alert:
    alert: str
    severity: str
    summary: str
    value: float
    threshold: float
    labels: dict[str, str] = field(default_factory=dict)
    runbook: str = RUNBOOK
    fired_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


@dataclass
class Report:
    model_version: str | None
    status: str  # ok | insufficient_data | no_model
    samples: int = 0
    labelled: int = 0
    feature_psi: dict[str, float] = field(default_factory=dict)
    score_psi: float | None = None
    store_psi: dict[str, dict[str, float]] = field(default_factory=dict)  # "feature|store" -> psi
    baseline_auc: float | None = None
    rolling_auc: float | None = None
    auc_series: list[float] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_report(settings: Settings, registry: ModelRegistry, pred_log: PredictionLog) -> Report:
    prod = registry.aliases()["production"]
    if prod is None:
        return Report(None, "no_model")
    meta = registry.metadata(prod)
    ref = meta["reference"]
    df = pred_log.recent(settings.monitor_window, prod)
    rep = Report(prod, "ok", samples=len(df), baseline_auc=meta["metrics"]["auc"])
    if len(df) < settings.monitor_min_samples:
        rep.status = "insufficient_data"
        return rep

    for f in FEATURE_COLUMNS:
        rep.feature_psi[f] = psi(ref["features"][f], df[f])
    rep.score_psi = psi(ref["score"], df["score"])

    for name in SEGMENT_FEATURES:
        refs = ref["segments"].get(name, {})
        for store, g in df.groupby("store_id"):
            if len(g) >= MIN_STORE_ROWS and str(int(store)) in refs:
                rep.store_psi.setdefault(name, {})[str(int(store))] = psi(refs[str(int(store))], g[name])

    lab = df.dropna(subset=["label"])
    rep.labelled = len(lab)
    window = settings.monitor_min_samples
    if len(lab) >= window:
        rep.rolling_auc = safe_auc(lab["label"].astype(int).tail(window), lab["score"].tail(window))
        rep.auc_series = rolling_auc(lab["label"].astype(int), lab["score"], window)
    rep.alerts = evaluate_alerts(settings, rep)
    return rep


def evaluate_alerts(settings: Settings, rep: Report) -> list[Alert]:
    alerts: list[Alert] = []
    alertable = {f: v for f, v in rep.feature_psi.items() if f not in NON_ALERTING_FEATURES}
    if alertable:
        feat, val = max(alertable.items(), key=lambda kv: kv[1])
        if val > settings.psi_threshold:
            alerts.append(Alert("FoundRateFeatureDrift", "warning",
                                f"Feature {feat} drifted (PSI {val:.3f} > {settings.psi_threshold})",
                                val, settings.psi_threshold, {"feature": feat}))
    worst = max(((n, s, v) for n, d in rep.store_psi.items() for s, v in d.items()), key=lambda t: t[2], default=None)
    if worst and worst[2] > settings.store_psi_threshold:
        alerts.append(Alert("FoundRateStoreDrift", "page",
                            f"Store {worst[1]} {worst[0]} drifted (PSI {worst[2]:.3f} > {settings.store_psi_threshold})",
                            worst[2], settings.store_psi_threshold, {"store": worst[1], "feature": worst[0]}))
    if rep.rolling_auc is not None and rep.baseline_auc is not None:
        drop = rep.baseline_auc - rep.rolling_auc
        if drop > settings.auc_drop_threshold:
            alerts.append(Alert("FoundRateModelAUCDegraded", "page",
                                f"Rolling AUC {rep.rolling_auc:.3f} is {drop:.3f} below baseline {rep.baseline_auc:.3f}",
                                drop, settings.auc_drop_threshold))
    return alerts


class MonitorMetrics:
    """Prometheus gauges for a Report."""

    ALERTS = ("FoundRateFeatureDrift", "FoundRateStoreDrift", "FoundRateModelAUCDegraded")

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.feature_psi = Gauge("foundrate_feature_psi", "PSI of live vs training feature", ["feature"], registry=self.registry)
        self.store_psi = Gauge("foundrate_store_feature_psi", "Per-store PSI", ["store", "feature"], registry=self.registry)
        self.score_psi = Gauge("foundrate_score_psi", "PSI of live vs reference scores", registry=self.registry)
        self.auc = Gauge("foundrate_rolling_auc", "Rolling AUC over labelled live traffic", registry=self.registry)
        self.baseline = Gauge("foundrate_baseline_auc", "Validation AUC of the production model", registry=self.registry)
        self.samples = Gauge("foundrate_monitor_samples", "Predictions in the monitor window", registry=self.registry)
        self.firing = Gauge("foundrate_monitor_alert_firing", "1 while the monitor's alert condition holds",
                            ["alert"], registry=self.registry)

    def update(self, rep: Report) -> None:
        self.samples.set(rep.samples)
        for f, v in rep.feature_psi.items():
            self.feature_psi.labels(f).set(v)
        for name, d in rep.store_psi.items():
            for store, v in d.items():
                self.store_psi.labels(store, name).set(v)
        if rep.score_psi is not None:
            self.score_psi.set(rep.score_psi)
        if rep.baseline_auc is not None:
            self.baseline.set(rep.baseline_auc)
        if rep.rolling_auc is not None:
            self.auc.set(rep.rolling_auc)
        active = {a.alert for a in rep.alerts}
        for name in self.ALERTS:
            self.firing.labels(name).set(1 if name in active else 0)

    def render(self) -> bytes:
        return generate_latest(self.registry)


def record_alerts(settings: Settings, alerts: list[Alert]) -> None:
    if not alerts:
        return
    settings.home.mkdir(parents=True, exist_ok=True)
    with settings.alerts_file.open("a") as f:
        for a in alerts:
            f.write(json.dumps(asdict(a)) + "\n")


def run_once(settings: Settings, metrics: MonitorMetrics | None = None) -> Report:
    registry = ModelRegistry(settings.registry_dir, settings.model_name)
    pred_log = PredictionLog(settings.prediction_log_db)
    try:
        rep = compute_report(settings, registry, pred_log)
    finally:
        pred_log.close()
    if metrics:
        metrics.update(rep)
    record_alerts(settings, rep.alerts)
    return rep


def serve(settings: Settings, port: int, interval_s: float, auto_remediate: bool = False) -> None:  # pragma: no cover
    """Long-running exporter: evaluate every `interval_s`, expose /metrics, optionally run the runbook."""
    from foundrate.runbook import remediate

    metrics = MonitorMetrics()
    start_http_server(port, registry=metrics.registry)
    firing: set[str] = set()
    while True:
        try:
            rep = run_once(settings, metrics)
            now = {a.alert for a in rep.alerts}
            for a in rep.alerts:
                if a.alert not in firing:
                    log.warning("ALERT %s: %s (runbook: %s)", a.alert, a.summary, a.runbook)
            if auto_remediate and now - firing:
                remediate(settings, rep)
            firing = now
        except Exception:
            log.exception("monitor iteration failed")
        time.sleep(interval_s)

