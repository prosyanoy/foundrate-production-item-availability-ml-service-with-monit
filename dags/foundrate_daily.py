"""Daily FoundRate pipeline: ingest -> SQL features -> train/register -> publish online features.

All logic lives in `foundrate.pipeline`; this file only wires it into Airflow tasks.
Configuration comes from the same FOUNDRATE_* environment variables as the service.
"""
from __future__ import annotations

from datetime import date, datetime

from airflow.sdk import dag, task

from foundrate import pipeline
from foundrate.config import Settings


def _ds(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


@dag(
    dag_id="foundrate_daily",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["foundrate", "ml"],
    doc_md=__doc__,
)
def foundrate_daily():
    @task
    def ingest_order_lines(ds: str | None = None) -> dict:
        n = pipeline.ingest(Settings(), _ds(ds))
        return {"order_lines": n}

    @task
    def build_features(_upstream: dict, ds: str | None = None) -> dict:
        return pipeline.build_features(Settings(), _ds(ds))

    @task
    def train_and_register(_upstream: dict, ds: str | None = None) -> dict:
        return pipeline.train(Settings(), _ds(ds))

    @task
    def publish_online_features(_upstream: dict, ds: str | None = None) -> int:
        return pipeline.publish_online(Settings(), _ds(ds))

    features = build_features(ingest_order_lines())
    # The online store is refreshed after the model is registered so a new model never
    # reads features that predate the snapshot it was validated against.
    publish_online_features(train_and_register(features))


foundrate_daily()
