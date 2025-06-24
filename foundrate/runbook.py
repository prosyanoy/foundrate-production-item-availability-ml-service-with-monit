"""Automated first response of docs/runbooks/model-drift.md (steps 1-3).

Steps 4+ (canary, promote/rollback) stay explicit: `foundrate canary --candidate <version>`.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

from foundrate import pipeline
from foundrate.config import Settings
from foundrate.feature_store import OfflineStore
from foundrate.monitoring.monitor import Report


def remediate(settings: Settings, report: Report | None = None) -> dict[str, Any]:
    """1) save an incident report, 2) refresh features to the newest data, 3) train a *candidate*.

    Nothing is promoted: the candidate must pass the canary guardrails first.
    """
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    incident = settings.reports_dir / f"incident-{stamp}.json"
    incident.write_text(json.dumps(report.to_dict() if report else {}, indent=2))

    last = OfflineStore(settings.offline_db).max_order_date()
    if last is None:
        raise RuntimeError("no order lines ingested; run `foundrate bootstrap` first")
    ds: date = last + timedelta(days=1)
    feats = pipeline.build_features(settings, ds)
    trained = pipeline.train(settings, ds)
    pipeline.publish_online(settings, ds)
    return {"incident_report": str(incident), "ds": ds.isoformat(), "candidate": trained["version"],
            "metrics": trained["metrics"], "snapshots": len(feats["snapshots"]),
            "next": f"foundrate canary --candidate {trained['version']}"}
