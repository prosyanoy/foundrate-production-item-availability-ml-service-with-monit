"""Train/serve parity: the online path must produce exactly the training features."""
from datetime import date, timedelta

import numpy as np

from foundrate.feature_store import OfflineStore, OnlineStore
from foundrate.features import FEATURE_COLUMNS, assemble_frame, assemble_vector, compute_defaults, to_matrix
from foundrate.registry import ModelRegistry

from .conftest import DS


def test_online_vectors_equal_offline_training_features(settings):
    offline = OfflineStore(settings.offline_db)
    as_of = DS - timedelta(days=7)  # a training snapshot (labels in [as_of, as_of+7))
    frame = offline.training_frame([as_of], 7).sample(500, random_state=0)
    defaults = ModelRegistry(settings.registry_dir, settings.model_name).metadata("v0001")["defaults"]

    online = OnlineStore(settings.online_db)
    online.publish(offline.snapshot(as_of), as_of)  # what the online store would hold at that time
    offline_x = to_matrix(assemble_frame(frame, defaults))
    online_x = np.array([
        assemble_vector(online.get(int(r.store_id), int(r.item_id)), int(r.order_hour), int(r.quantity), defaults)
        for r in frame.itertuples()
    ])
    assert offline_x.shape[1] == len(FEATURE_COLUMNS)
    np.testing.assert_array_equal(offline_x, online_x)
    online.close()


def test_unseen_entity_falls_back_to_defaults(settings):
    defaults = compute_defaults(OfflineStore(settings.offline_db).snapshot(DS))
    vec = assemble_vector(None, 10, 2, defaults)
    assert vec[:-2] == [defaults[c] for c in FEATURE_COLUMNS[:-2]]
    assert vec[-2:] == [10.0, 2.0]


def test_service_scores_match_offline_scoring(settings, client):
    """End to end: the API's score equals scoring the offline snapshot row with the same model."""
    offline = OfflineStore(settings.offline_db)
    reg = ModelRegistry(settings.registry_dir, settings.model_name)
    loaded = reg.load("v0001")
    snap = offline.snapshot(DS).iloc[0]
    row = snap.copy()
    row["order_hour"], row["quantity"] = 12, 2
    x = to_matrix(assemble_frame(row.to_frame().T, loaded.metadata["defaults"]))
    expected = loaded.model.predict_proba(x)[0, 1]
    r = client.post("/predict", json={"store_id": int(snap.store_id), "item_id": int(snap.item_id),
                                      "order_hour": 12, "quantity": 2})
    assert r.status_code == 200
    assert abs(r.json()["found_probability"] - expected) < 1e-12
