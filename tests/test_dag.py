import os
from pathlib import Path

import pytest

pytest.importorskip("airflow", reason="install requirements-airflow.txt to run the DAG import test")

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def dagbag(tmp_path_factory):
    home = tmp_path_factory.mktemp("airflow")
    os.environ.update(AIRFLOW_HOME=str(home), AIRFLOW__CORE__LOAD_EXAMPLES="False",
                      AIRFLOW__CORE__UNIT_TEST_MODE="True")
    from airflow.models import DagBag

    return DagBag(dag_folder=str(DAGS), include_examples=False)


def test_dag_imports_cleanly(dagbag):
    assert dagbag.import_errors == {}
    assert "foundrate_daily" in dagbag.dags


def test_dag_structure(dagbag):
    dag = dagbag.dags["foundrate_daily"]
    assert str(dag.schedule) in ("@daily", "0 0 * * *") or "daily" in str(dag.timetable).lower() or dag.schedule == "@daily"
    order = ["ingest_order_lines", "build_features", "train_and_register", "publish_online_features"]
    assert set(dag.task_ids) == set(order)
    for up, down in zip(order, order[1:]):
        assert down in dag.get_task(up).downstream_task_ids
