from prometheus_client.parser import text_string_to_metric_families

from foundrate.registry import ModelRegistry
from foundrate.service.router import bucket

from .helpers import ConstantModel, FaultyModel

PAYLOAD = {"store_id": 1, "item_id": 3, "order_hour": 14, "quantity": 1}


def metric(client, name, **labels):
    total = 0.0
    for fam in text_string_to_metric_families(client.get("/metrics").text):
        for s in fam.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                total += s.value
    return total


def test_predict_returns_probability_and_version_header(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["found_probability"] <= 1 and body["features_found"] is True
    assert r.headers["X-Model-Version"] == body["model_version"] == "v0001"


def test_unseen_store_item_uses_defaults(client):
    r = client.post("/predict", json={**PAYLOAD, "store_id": 999})
    assert r.status_code == 200 and r.json()["features_found"] is False
    assert metric(client, "foundrate_feature_store_misses_total") == 1


def test_request_validation(client):
    for bad in ({**PAYLOAD, "order_hour": 24}, {**PAYLOAD, "quantity": 0}, {**PAYLOAD, "store_id": -1},
                {"store_id": 1}, {**PAYLOAD, "item_id": "abc"}):
        assert client.post("/predict", json=bad).status_code == 422


def test_metrics_expose_latency_and_prediction_histograms(client):
    for _ in range(3):
        client.post("/predict", json=PAYLOAD)
    assert metric(client, "foundrate_request_latency_seconds_count", endpoint="/predict") == 3
    assert metric(client, "foundrate_prediction_score_count", model_version="v0001") == 3
    assert metric(client, "foundrate_requests_total", status="200") == 3
    assert metric(client, "foundrate_model_info", role="production") == 1


def test_feedback_attaches_label(client):
    rid = client.post("/predict", json={**PAYLOAD, "request_id": "abc"}).json()["request_id"]
    assert client.post("/feedback", json={"request_id": rid, "found": True}).status_code == 200
    assert client.post("/feedback", json={"request_id": "nope", "found": True}).status_code == 404
    df = client.app.state.prediction_log.recent(10)
    assert df.loc[df.request_id == "abc", "label"].iloc[0] == 1


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["production"] == "v0001" and h["online_features_as_of"] == "2026-01-15"


def test_canary_routing_split_and_live_rollback(settings, client):
    reg = ModelRegistry(settings.registry_dir, settings.model_name)
    v = reg.register(ConstantModel(0.5), {"defaults": reg.metadata("v0001")["defaults"], "metrics": {"auc": 0.5}})
    reg.start_canary(v, 30)
    ids = [f"req-{i}" for i in range(1000)]
    versions = [client.post("/predict", json={**PAYLOAD, "request_id": i}).headers["X-Model-Version"] for i in ids]
    share = versions.count(v) / len(ids)
    assert 0.22 < share < 0.38
    # routing is deterministic per request id
    assert all((vv == v) == (bucket(i) < 30) for i, vv in zip(ids, versions))
    reg.rollback("test")
    assert {client.post("/predict", json={**PAYLOAD, "request_id": i}).headers["X-Model-Version"]
            for i in ids[:50]} == {"v0001"}


def test_failing_candidate_reports_500_with_its_version(settings, client):
    reg = ModelRegistry(settings.registry_dir, settings.model_name)
    v = reg.register(FaultyModel(), {"defaults": reg.metadata("v0001")["defaults"], "metrics": {"auc": 0.5}})
    reg.start_canary(v, 100)
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 500 and r.headers["X-Model-Version"] == v
    assert metric(client, "foundrate_requests_total", model_version=v, status="500") == 1


def test_no_model_returns_503(tmp_path):
    from fastapi.testclient import TestClient

    from foundrate.service.app import create_app

    from .conftest import small_settings

    with TestClient(create_app(small_settings(tmp_path / "empty"))) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=PAYLOAD).status_code == 503
