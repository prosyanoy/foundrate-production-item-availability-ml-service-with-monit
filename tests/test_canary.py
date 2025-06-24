from datetime import timedelta

import pytest

from foundrate import pipeline
from foundrate.registry import ModelRegistry
from foundrate.rollout.canary import ArmStats, CanaryController, Guardrails, check_guardrails
from foundrate.simulate import TrafficSource

from .conftest import DS
from .helpers import FaultyModel

GUARD = Guardrails(min_samples=150)


def controller(settings, client, **kw):
    src = TrafficSource(pipeline.world_for(settings), DS + timedelta(days=1))
    reg = ModelRegistry(settings.registry_dir, settings.model_name)
    return reg, CanaryController(reg, client, src.take, GUARD, kw.get("steps", (10, 50)), 600)


def test_good_candidate_is_promoted(settings, client):
    reg, ctl = controller(settings, client)
    cand = pipeline.train(settings, DS)["version"]
    res = ctl.run(cand)
    assert res.status == "promoted", res.reason
    a = reg.aliases()
    assert a["production"] == cand and a["candidate"] is None and a["canary_percent"] == 0
    assert [s.percent for s in res.steps] == [10, 50]


def test_regressed_candidate_is_rolled_back(settings, client):
    reg, ctl = controller(settings, client)
    bad = pipeline.train(settings, DS, shuffle_labels=True)["version"]
    res = ctl.run(bad)
    assert res.status == "rolled_back" and "AUC" in res.reason
    a = reg.aliases()
    assert a["production"] == "v0001" and a["candidate"] is None and a["canary_percent"] == 0
    assert a["history"][-1]["event"] == "rollback"


def test_erroring_candidate_is_rolled_back_at_first_step(settings, client):
    reg, ctl = controller(settings, client, steps=(20, 50))
    v = reg.register(FaultyModel(), {"defaults": reg.metadata("v0001")["defaults"], "metrics": {"auc": 0.5}})
    res = ctl.run(v)
    assert res.status == "rolled_back" and "error rate" in res.reason
    assert len(res.steps) == 1  # stopped at the first step, never ramped further
    assert reg.aliases()["canary_percent"] == 0


def test_controller_refuses_current_production(settings, client):
    _, ctl = controller(settings, client)
    with pytest.raises(ValueError):
        ctl.run("v0001")


def test_guardrail_rules():
    prod = ArmStats(n=1000, p95_latency_ms=10, auc=0.75, auc_se=0.01, mean_score=0.6)
    ok = ArmStats(n=400, p95_latency_ms=12, auc=0.745, auc_se=0.02, mean_score=0.61)
    assert check_guardrails(prod, ok, GUARD) == []
    assert any("error rate" in v for v in check_guardrails(prod, ArmStats(n=100, errors=5), GUARD))
    slow = ArmStats(n=400, p95_latency_ms=900, auc=0.75, auc_se=0.02, mean_score=0.6)
    assert any("latency" in v for v in check_guardrails(prod, slow, GUARD))
    shifted = ArmStats(n=400, p95_latency_ms=10, auc=0.75, auc_se=0.02, mean_score=0.8)
    assert any("mean score" in v for v in check_guardrails(prod, shifted, GUARD))
    # a drop inside sampling noise is NOT a violation, a large one is
    noisy = ArmStats(n=400, p95_latency_ms=10, auc=0.70, auc_se=0.03, mean_score=0.6)
    assert check_guardrails(prod, noisy, GUARD) == []
    bad = ArmStats(n=400, p95_latency_ms=10, auc=0.55, auc_se=0.03, mean_score=0.6)
    assert any("AUC" in v for v in check_guardrails(prod, bad, GUARD))
    # too few samples: quality guardrails are skipped, reliability ones still apply
    assert check_guardrails(prod, ArmStats(n=20, auc=0.5, auc_se=0.1, mean_score=0.6, p95_latency_ms=5), GUARD) == []
