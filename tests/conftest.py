import shutil
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from foundrate import pipeline
from foundrate.config import Settings
from foundrate.service.app import create_app

DS = date(2026, 1, 15)


def small_settings(home: Path) -> Settings:
    return Settings(home=home, n_stores=6, n_items=40, lines_per_day=350, seed=3, train_snapshots=6)


@pytest.fixture(scope="session")
def template_home(tmp_path_factory) -> Path:
    """One bootstrapped state directory (data, features, model v0001, online store) shared by tests."""
    home = tmp_path_factory.mktemp("template")
    pipeline.run_daily(small_settings(home), DS)
    return home


@pytest.fixture
def settings(template_home, tmp_path) -> Settings:
    """Private copy of the template, so tests may mutate the registry/logs freely."""
    home = tmp_path / "home"
    shutil.copytree(template_home, home)
    return small_settings(home)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c
