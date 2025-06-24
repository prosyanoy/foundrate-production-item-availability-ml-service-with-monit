"""Command line entry point: python -m foundrate.cli <command>."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta

import httpx

from foundrate import pipeline
from foundrate.config import Settings
from foundrate.feature_store import OfflineStore
from foundrate.monitoring import monitor as mon
from foundrate.registry import ModelRegistry
from foundrate.rollout.canary import CanaryController, Guardrails, write_report
from foundrate.runbook import remediate
from foundrate.simulate import TrafficSource, send_traffic
from foundrate.synthetic import Shift

DEFAULT_DS = "2026-01-15"


def _date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _shifts(args: argparse.Namespace) -> list[Shift]:
    if args.shift_store is None:
        return []
    return [Shift(args.shift_store, _date(args.shift_start), args.shift_delta)]


def _client(args: argparse.Namespace, settings: Settings) -> httpx.Client:
    if args.url:
        return httpx.Client(base_url=args.url, timeout=10)
    from fastapi.testclient import TestClient

    from foundrate.service.app import create_app

    return TestClient(create_app(settings))  # in-process: no server needed


def _next_day(settings: Settings) -> date:
    last = OfflineStore(settings.offline_db).max_order_date()
    return (last + timedelta(days=1)) if last else _date(DEFAULT_DS)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="foundrate")
    p.add_argument("--home", help="state directory (default: $FOUNDRATE_HOME or ./var)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def shift_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--shift-store", type=int, help="store whose stock pattern changes")
        sp.add_argument("--shift-start", default=DEFAULT_DS)
        sp.add_argument("--shift-delta", type=float, default=3.0)

    b = sub.add_parser("bootstrap", help="ingest, build features, train v1, promote, publish online features")
    b.add_argument("--ds", default=DEFAULT_DS)
    shift_args(b)

    f = sub.add_parser("refresh", help="advance data to --ds (ingest, features, publish online); no training")
    f.add_argument("--ds", required=True)
    f.add_argument("--from", dest="start", help="first day to (re)ingest, default: day after current data")
    shift_args(f)

    t = sub.add_parser("train", help="train + register a new version (candidate) as of --ds")
    t.add_argument("--ds")
    t.add_argument("--shuffle-labels", action="store_true", help="fault injection: regressed model for demos")

    tr = sub.add_parser("traffic", help="send synthetic labelled traffic to the service")
    tr.add_argument("--url")
    tr.add_argument("-n", type=int, default=2000)
    tr.add_argument("--start", help="first traffic day (default: day after ingested data)")
    shift_args(tr)

    m = sub.add_parser("monitor", help="evaluate drift/AUC once, or run as an exporter")
    m.add_argument("--serve", action="store_true")
    m.add_argument("--port", type=int, default=9100)
    m.add_argument("--interval", type=float, default=30)
    m.add_argument("--auto-remediate", action="store_true", help="run the runbook's automated steps on alert")

    sub.add_parser("remediate", help="runbook automation: retrain a candidate on the newest data")

    c = sub.add_parser("canary", help="ramp a candidate with auto-rollback")
    c.add_argument("--candidate", required=True)
    c.add_argument("--url")
    c.add_argument("--steps", default="5,25,50")
    c.add_argument("--requests-per-step", type=int, default=1500)
    c.add_argument("--max-auc-drop", type=float, default=Guardrails.max_auc_drop)
    c.add_argument("--min-samples", type=int, default=Guardrails.min_samples)
    shift_args(c)

    sub.add_parser("registry", help="show versions and rollout state")
    d = sub.add_parser("demo", help="scripted end-to-end scenario (healthy -> drift -> runbook -> canary)")
    d.add_argument("--fast", action="store_true", help="smaller traffic volumes")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings(home=__import__("pathlib").Path(args.home)) if args.home else Settings()
    registry = ModelRegistry(settings.registry_dir, settings.model_name)

    if args.cmd == "bootstrap":
        print(json.dumps(pipeline.run_daily(settings, _date(args.ds), _shifts(args)), indent=2))
    elif args.cmd == "refresh":
        ds = _date(args.ds)
        start = _date(args.start) if args.start else _next_day(settings)
        pipeline.ingest(settings, ds, _shifts(args), start=start)
        pipeline.build_features(settings, ds)
        print(json.dumps({"ds": ds.isoformat(), "online_rows": pipeline.publish_online(settings, ds)}))
    elif args.cmd == "train":
        ds = _date(args.ds) if args.ds else _next_day(settings)
        print(json.dumps(pipeline.train(settings, ds, shuffle_labels=args.shuffle_labels), indent=2))
    elif args.cmd == "traffic":
        start = _date(args.start) if args.start else _next_day(settings)
        with _client(args, settings) as client:
            outs = send_traffic(client, TrafficSource(pipeline.world_for(settings), start, _shifts(args)).take(args.n))
        print(json.dumps({"sent": len(outs), "errors": sum(o.status != 200 for o in outs)}))
    elif args.cmd == "monitor":
        if args.serve:
            mon.serve(settings, args.port, args.interval, args.auto_remediate)
        else:
            rep = mon.run_once(settings)
            print(json.dumps(rep.to_dict(), indent=2))
            if args.auto_remediate and rep.alerts:
                print(json.dumps(remediate(settings, rep), indent=2))
            return 2 if rep.alerts else 0
    elif args.cmd == "remediate":
        print(json.dumps(remediate(settings), indent=2))
    elif args.cmd == "canary":
        start = _next_day(settings)
        source = TrafficSource(pipeline.world_for(settings), start, _shifts(args))
        with _client(args, settings) as client:
            ctl = CanaryController(registry, client, source.take, Guardrails(max_auc_drop=args.max_auc_drop, min_samples=args.min_samples),
                                   tuple(int(s) for s in args.steps.split(",")), args.requests_per_step, print)
            result = ctl.run(args.candidate)
        write_report(settings.reports_dir / f"canary-{args.candidate}.json", result)
        print(f"{result.status.upper()}: {result.reason}")
        return 0 if result.status == "promoted" else 1
    elif args.cmd == "registry":
        print(json.dumps({"versions": {v: registry.metadata(v)["metrics"] for v in registry.versions()},
                          "aliases": {k: v for k, v in registry.aliases().items() if k != "history"}}, indent=2))
    elif args.cmd == "demo":
        from foundrate.demo import run_demo

        return run_demo(settings, fast=args.fast)
    return 0


if __name__ == "__main__":
    sys.exit(main())
