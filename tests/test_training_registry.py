import json

import pytest

from foundrate import pipeline
from foundrate.registry import ModelRegistry
from foundrate.training import training_as_ofs

from .conftest import DS


def test_training_as_ofs_have_fully_observed_labels():
    ofs = training_as_ofs(DS, 7, 4)
    assert ofs == sorted(ofs) and ofs[-1] == DS.replace(day=8)
    assert all((DS - o).days >= 7 for o in ofs)


def test_registered_artifact_has_metrics_and_reference(settings):
    reg = ModelRegistry(settings.registry_dir, settings.model_name)
    meta = reg.metadata("v0001")
    assert meta["metrics"]["auc"] > 0.68
    assert 0 < meta["metrics"]["logloss"] < 0.7
    assert set(meta["defaults"]) and meta["reference"]["features"]["si_rate_30d"]["edges"]
    assert json.loads((reg.dir / "v0001" / "metadata.json").read_text())["version"] == "v0001"


def test_train_registers_new_version_without_promoting(settings):
    out = pipeline.train(settings, DS)
    assert out["version"] == "v0002" and out["promoted"] is False
    a = ModelRegistry(settings.registry_dir, settings.model_name).aliases()
    assert a["production"] == "v0001" and a["candidate"] is None


def test_registry_promote_canary_rollback(settings):
    reg = ModelRegistry(settings.registry_dir, settings.model_name)
    pipeline.train(settings, DS)
    reg.start_canary("v0002", 10)
    assert reg.aliases()["candidate"] == "v0002" and reg.aliases()["canary_percent"] == 10
    reg.rollback("test")
    assert reg.aliases()["candidate"] is None and reg.aliases()["production"] == "v0001"
    reg.promote("v0002")
    assert reg.aliases()["production"] == "v0002" and reg.aliases()["previous_production"] == "v0001"
    with pytest.raises(KeyError):
        reg.promote("v0099")
    with pytest.raises(ValueError):
        reg.start_canary("v0001", 101)
