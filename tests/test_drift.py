import json
from datetime import timedelta

import numpy as np
import pytest

from foundrate import pipeline
from foundrate.monitoring import monitor
from foundrate.monitoring.psi import make_reference, psi, rolling_auc
from foundrate.runbook import remediate
from foundrate.simulate import TrafficSource, send_traffic
from foundrate.synthetic import Shift

from .conftest import DS


def test_psi_identical_is_zero_and_shift_is_large():
    rng = np.random.default_rng(0)
    ref = make_reference(rng.normal(0, 1, 20_000))
    assert psi(ref, rng.normal(0, 1, 5_000)) < 0.02
    assert psi(ref, rng.normal(0.5, 1, 5_000)) > 0.1
    assert psi(ref, rng.normal(2, 1, 5_000)) > 0.5


def test_psi_known_value():
    ref = {"edges": [0.5], "props": [0.5, 0.5]}
    # actual 80/20 vs expected 50/50: (0.3 * ln 1.6) + (-0.3 * ln 0.4)
    expected = 0.3 * np.log(1.6) + (-0.3) * np.log(0.4)
    assert psi(ref, [0.0] * 80 + [1.0] * 20) == pytest.approx(expected)


def test_rolling_auc_tracks_degradation():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    good = y + rng.normal(0, 0.6, 2000)
    score = np.concatenate([good[:1000], rng.normal(0, 1, 1000)])  # second half: random scores
    series = rolling_auc(y, score, window=500, step=250)
    assert series[0] > 0.8 and series[-1] < 0.6


def _traffic(settings, client, n, start, shifts=()):
    send_traffic(client, TrafficSource(pipeline.world_for(settings), start, shifts).take(n))


def test_healthy_traffic_raises_no_alert(settings, client):
    _traffic(settings, client, 1200, DS)
    rep = monitor.run_once(settings)
    assert rep.status == "ok" and rep.alerts == []
    assert rep.rolling_auc > rep.baseline_auc - 0.05
    assert not settings.alerts_file.exists()


def test_insufficient_data_is_reported(settings, client):
    _traffic(settings, client, 50, DS)
    assert monitor.run_once(settings).status == "insufficient_data"


def test_store_stock_shift_fires_alert_and_runbook_trains_candidate(settings, client):
    shift = Shift(store_id=2, start=DS + timedelta(days=7), logit_delta=3.0)
    ds2 = DS + timedelta(days=14)
    pipeline.ingest(settings, ds2, [shift], start=DS)
    pipeline.build_features(settings, ds2)
    pipeline.publish_online(settings, ds2)
    _traffic(settings, client, 2500, ds2, [shift])

    metrics = monitor.MonitorMetrics()
    rep = monitor.run_once(settings, metrics)
    names = {a.alert for a in rep.alerts}
    assert "FoundRateStoreDrift" in names
    alert = next(a for a in rep.alerts if a.alert == "FoundRateStoreDrift")
    assert alert.labels["store"] == "2" and alert.runbook == "docs/runbooks/model-drift.md"
    assert json.loads(settings.alerts_file.read_text().splitlines()[0])["alert"] in names
    prom = metrics.render().decode()
    assert 'foundrate_monitor_alert_firing{alert="FoundRateStoreDrift"} 1.0' in prom
    assert 'foundrate_store_feature_psi{feature="si_rate_7d",store="2"}' in prom

    out = remediate(settings, rep)  # the runbook's automated steps
    assert out["candidate"] == "v0002" and out["ds"] == ds2.isoformat()
    assert json.loads(open(out["incident_report"]).read())["status"] == "ok"
    from foundrate.registry import ModelRegistry

    assert ModelRegistry(settings.registry_dir, settings.model_name).aliases()["production"] == "v0001"


def test_auc_alert_when_scores_decorrelate(settings, client):
    _traffic(settings, client, 1000, DS)
    log = client.app.state.prediction_log
    with log._con:  # corrupt the stored scores: simulates a model that no longer ranks anything
        log._con.execute("UPDATE predictions SET score = abs(random() % 1000) / 1000.0")
    rep = monitor.run_once(settings)
    assert any(a.alert == "FoundRateModelAUCDegraded" for a in rep.alerts)


def test_thresholds_match_prometheus_rules():
    import yaml
    from pathlib import Path

    from foundrate.config import Settings

    rules = yaml.safe_load(Path("deploy/prometheus/alert_rules.yml").read_text())
    exprs = {r["alert"]: r["expr"] for g in rules["groups"] for r in g["rules"]}
    s = Settings()
    assert f"> {s.psi_threshold}" in exprs["FoundRateFeatureDrift"]
    assert f"> {s.store_psi_threshold}" in exprs["FoundRateStoreDrift"]
    assert f"> {s.auc_drop_threshold}" in exprs["FoundRateModelAUCDegraded"]
    for g in rules["groups"]:
        for r in g["rules"]:
            assert "runbook" in r["annotations"]
