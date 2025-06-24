# Runbook: model drift / degraded availability predictions

**Alerts that land here:** `FoundRateFeatureDrift`, `FoundRateStoreDrift`, `FoundRateModelAUCDegraded`
(Prometheus rules in `deploy/prometheus/alert_rules.yml`; the monitor also appends every firing alert,
with this runbook's path, to `$FOUNDRATE_HOME/alerts.jsonl`).

| Alert | Meaning | Typical cause |
|---|---|---|
| `FoundRateFeatureDrift` | PSI > 0.2 for a live input feature vs. the production model's training distribution | upstream pipeline bug, catalogue change, seasonality |
| `FoundRateStoreDrift` | per-store PSI of `si_rate_7d` > 0.5 | a store's stock pattern changed (delivery issue, remodel, supplier switch) |
| `FoundRateModelAUCDegraded` | rolling AUC over labelled traffic is > 0.05 below the validation AUC | concept drift, stale online features, broken labels |

PSI guide: < 0.1 stable, 0.1-0.2 moderate, > 0.2 significant. `store_rate_7d` is exported but never alerts (one value per store, so its PSI is noise).

## 1. Triage (5 min)
1. Look at the Grafana dashboard "FoundRate" (http://localhost:3000): which feature / store moved, since when?
2. `python -m foundrate.cli monitor` prints the full report (per-feature PSI, per-store PSI, rolling AUC series). Exit code 2 = alert firing.
3. Rule out a data problem first:
   - `curl localhost:8000/health` -> `online_features_as_of` should be today or yesterday. If stale, the daily DAG failed: fix that (Airflow UI), re-run it, and re-check; the drift may disappear.
   - `foundrate_feature_store_misses_total` rising means requests for store/item pairs without features (new store or item).
4. If the drift is confined to one store, confirm with the store operations team whether its stock really changed.

## 2. Automated first response
```bash
python -m foundrate.cli remediate        # or: monitor --auto-remediate / monitor --serve --auto-remediate
```
This (a) saves an incident report (the monitor output) to `$FOUNDRATE_HOME/reports/`, (b) rebuilds feature snapshots through the newest ingested day, (c) trains a **candidate** model on them and registers it, and (d) republishes the online features. It never promotes anything.

## 3. Roll out the candidate safely
```bash
python -m foundrate.cli canary --candidate v0002                      # in-process traffic
python -m foundrate.cli canary --candidate v0002 --url http://localhost:8000
```
Traffic ramps 5% -> 25% -> 50% through the live service. Guardrails (candidate vs. production on the same traffic): 5xx rate <= 1%, p95 latency <= 250 ms (and <= 3x production), mean score shift <= 0.10, and AUC not significantly worse than production (tolerance 0.02, 2 standard errors). Any violation sets `candidate=null, canary_percent=0` (automatic rollback, production untouched) and exits 1; passing all steps promotes the candidate. The report is in `reports/canary-<version>.json`.

## 4. If the canary rolls back
The retrained model is not better; the drift is real but retraining did not fix it. Escalate to the model owner; consider (a) more recent data / sample weights, (b) a store-specific feature, (c) accepting the drift if business impact is low. Silence nothing: the alert stays until PSI/AUC recover or the production model is replaced.

## 5. Manual rollback of a promoted model
```bash
python - <<'PY'
from foundrate.config import Settings; from foundrate.registry import ModelRegistry
s = Settings(); r = ModelRegistry(s.registry_dir, s.model_name)
r.promote(r.aliases()["previous_production"], reason="manual rollback")
PY
```
The service picks up `aliases.json` changes on the next request; no restart is needed.

## Service health
For `FoundRateHighLatency` / `FoundRateErrorRate`: check `foundrate_requests_total{status="500"}` by `model_version`. If only the candidate version fails during a canary, the controller rolls back by itself; otherwise follow step 5 and inspect the service logs (`docker compose logs api`).
