"""Scripted end-to-end scenario used by `make demo`.

healthy traffic -> store stock pattern shifts -> drift alert -> runbook trains a candidate ->
canary promotes it -> a regressed candidate is auto-rolled back.
"""
from __future__ import annotations

import shutil
from datetime import date, timedelta

from fastapi.testclient import TestClient

from foundrate import pipeline
from foundrate.config import Settings
from foundrate.monitoring import monitor as mon
from foundrate.registry import ModelRegistry
from foundrate.rollout.canary import CanaryController, Guardrails
from foundrate.runbook import remediate
from foundrate.service.app import create_app
from foundrate.simulate import TrafficSource, send_traffic
from foundrate.synthetic import Shift

T = date(2026, 1, 15)
SHIFTED_STORE = 3


def _say(msg: str) -> None:
    print(f"\n== {msg}", flush=True)


def run_demo(settings: Settings, fast: bool = False) -> int:
    n_traffic, n_canary = (1500, 800) if fast else (2500, 1500)
    if settings.home.exists():
        shutil.rmtree(settings.home)
    registry = ModelRegistry(settings.registry_dir, settings.model_name)
    world = pipeline.world_for(settings)
    shift = Shift(SHIFTED_STORE, T + timedelta(days=7), logit_delta=3.0, item_fraction=0.7)

    _say(f"1. daily pipeline for {T}: ingest, SQL features, train, register, publish online features")
    out = pipeline.run_daily(settings, T)
    print(f"   registered {out['version']} (promoted={out['promoted']}) AUC={out['metrics']['auc']:.3f}")

    app = create_app(settings)
    with TestClient(app) as client:
        _say("2. healthy live traffic -> monitor")
        send_traffic(client, TrafficSource(world, T).take(n_traffic))
        rep = mon.run_once(settings)
        print(f"   status={rep.status} max feature PSI={max(rep.feature_psi.values()):.3f} "
              f"rolling AUC={rep.rolling_auc and round(rep.rolling_auc, 3)} alerts={[a.alert for a in rep.alerts]}")
        if rep.alerts:
            print("   unexpected alert on healthy traffic")
            return 1

        _say(f"3. store {SHIFTED_STORE}'s stock pattern changes on {shift.start}; features refresh daily")
        ds2 = T + timedelta(days=14)
        pipeline.ingest(settings, ds2, [shift], start=T)
        pipeline.build_features(settings, ds2)
        pipeline.publish_online(settings, ds2)
        send_traffic(client, TrafficSource(world, ds2, [shift]).take(n_traffic * 2))
        rep = mon.run_once(settings)
        for a in rep.alerts:
            print(f"   ALERT {a.alert} [{a.severity}]: {a.summary}\n         runbook: {a.runbook}")
        if not rep.alerts:
            print("   expected a drift alert but none fired")
            return 1

        _say("4. runbook automation: incident report + candidate trained on the newest data")
        rem = remediate(settings, rep)
        print(f"   candidate {rem['candidate']} AUC={rem['metrics']['auc']:.3f}; incident report {rem['incident_report']}")

        _say(f"5. canary {rem['candidate']} (10% -> 50%) with guardrails")
        ds3 = ds2 + timedelta(days=1)
        ctl = CanaryController(registry, client, TrafficSource(world, ds3, [shift]).take,
                               Guardrails(min_samples=200), (10, 50), n_canary, lambda m: print("   " + m))
        res = ctl.run(rem["candidate"])
        print(f"   -> {res.status.upper()}: {res.reason}")

        _say("6. regressed candidate (label-shuffled training bug) -> canary must roll back")
        bad = pipeline.train(settings, ds3, shuffle_labels=True)["version"]
        ctl = CanaryController(registry, client, TrafficSource(world, ds3 + timedelta(days=1), [shift]).take,
                               Guardrails(min_samples=200), (10, 50), n_canary, lambda m: print("   " + m))
        res_bad = ctl.run(bad)
        print(f"   -> {res_bad.status.upper()}: {res_bad.reason}")

    a = registry.aliases()
    _say(f"final state: production={a['production']} candidate={a['candidate']} canary={a['canary_percent']}%")
    for h in a["history"]:
        print(f"   {h['at']} {h['event']} {h.get('version')} {h.get('reason', '')}")
    return 0 if res.status == "promoted" and res_bad.status == "rolled_back" else 1
