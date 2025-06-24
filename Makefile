PY ?= .venv/bin/python
PYTHON ?= python3

.PHONY: install install-airflow test bootstrap serve traffic monitor demo up down

install:
	$(PYTHON) -m venv .venv && .venv/bin/pip install -q -r requirements-dev.txt

install-airflow:  # optional; Python 3.9-3.12
	.venv/bin/pip install -q -r requirements-airflow.txt --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-$$(.venv/bin/python -c 'import sys;print("%d.%d"%sys.version_info[:2])').txt

test:
	$(PY) -m pytest -q

bootstrap:  ## ingest -> features -> train v1 -> promote -> publish online features
	$(PY) -m foundrate.cli bootstrap

serve:
	$(PY) -m uvicorn foundrate.service.app:create_app --factory --port 8000

traffic:
	$(PY) -m foundrate.cli traffic --url http://localhost:8000 -n 2000

monitor:
	$(PY) -m foundrate.cli monitor

demo:  ## scripted scenario: drift -> alert -> runbook -> canary promote + rollback (~2 min)
	$(PY) -m foundrate.cli demo

up:
	docker compose up --build -d && docker compose run --rm tools bootstrap

down:
	docker compose down
