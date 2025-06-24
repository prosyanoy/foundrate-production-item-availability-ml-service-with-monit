"""FastAPI model service: validated requests, online feature lookup, canary routing, Prometheus metrics."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from foundrate.config import Settings
from foundrate.feature_store import OnlineStore
from foundrate.features import FEATURE_COLUMNS, assemble_vector
from foundrate.prediction_log import PredictionLog
from foundrate.registry import ModelRegistry
from foundrate.service.router import ModelRouter

log = logging.getLogger("foundrate.service")

LATENCY_BUCKETS = (0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
SCORE_BUCKETS = tuple(i / 10 for i in range(1, 10)) + (1.0,)
VERSION_HEADER = "X-Model-Version"


class PredictRequest(BaseModel):
    store_id: int = Field(ge=0, description="Store identifier")
    item_id: int = Field(ge=0, description="Item identifier")
    order_hour: int = Field(ge=0, le=23, description="Hour of day the order is picked")
    quantity: int = Field(default=1, ge=1, le=50)
    request_id: str | None = Field(default=None, min_length=1, max_length=64)


class PredictResponse(BaseModel):
    request_id: str
    found_probability: float
    model_version: str
    features_found: bool


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    found: bool


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    registry = ModelRegistry(settings.registry_dir, settings.model_name)
    router = ModelRouter(registry)
    online = OnlineStore(settings.online_db)
    pred_log = PredictionLog(settings.prediction_log_db)

    prom = CollectorRegistry()
    latency = Histogram("foundrate_request_latency_seconds", "Request latency", ["endpoint", "model_version"],
                        buckets=LATENCY_BUCKETS, registry=prom)
    requests_total = Counter("foundrate_requests_total", "Requests by outcome", ["endpoint", "model_version", "status"],
                             registry=prom)
    scores = Histogram("foundrate_prediction_score", "Predicted found-probability", ["model_version"],
                       buckets=SCORE_BUCKETS, registry=prom)
    fs_miss = Counter("foundrate_feature_store_misses_total", "Online feature lookups with no row", registry=prom)
    feedback_total = Counter("foundrate_feedback_total", "Labels received", ["found"], registry=prom)
    canary_pct = Gauge("foundrate_canary_percent", "Share of traffic routed to the candidate", registry=prom)
    model_info = Gauge("foundrate_model_info", "Models currently routable", ["model_version", "role"], registry=prom)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        online.close()
        pred_log.close()

    app = FastAPI(title="FoundRate", version="1.0.0", lifespan=lifespan)
    app.state.router, app.state.registry, app.state.online, app.state.prediction_log = (
        router, registry, online, pred_log)

    @app.get("/health")
    def health(response: Response) -> dict:
        s = router.state()
        if s.production is None:
            response.status_code = 503
            return {"status": "no_model"}
        as_of = online.as_of()
        return {"status": "ok", "production": s.production, "candidate": s.candidate,
                "canary_percent": s.canary_percent, "online_features_as_of": as_of.isoformat() if as_of else None}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, response: Response) -> PredictResponse:
        start = time.perf_counter()
        version, status = "none", "200"
        try:
            request_id = req.request_id or uuid.uuid4().hex
            loaded = router.select(request_id)
            if loaded is None:
                status = "503"
                raise HTTPException(503, "no production model registered")
            version = loaded.version
            try:
                row = online.get(req.store_id, req.item_id)
                if row is None:
                    fs_miss.inc()
                vec = assemble_vector(row, req.order_hour, req.quantity, loaded.metadata["defaults"])
                score = float(loaded.model.predict_proba(np.asarray([vec]))[0, 1])
            except Exception:
                log.exception("inference failed (model %s)", version)
                status = "500"
                raise HTTPException(500, "inference failed", headers={VERSION_HEADER: version})
            scores.labels(version).observe(score)
            try:
                pred_log.log(request_id, req.store_id, req.item_id, version, score,
                             dict(zip(FEATURE_COLUMNS, vec)), row is not None)
            except Exception:  # logging must never fail a prediction
                log.exception("prediction log write failed")
            response.headers[VERSION_HEADER] = version
            return PredictResponse(request_id=request_id, found_probability=score, model_version=version,
                                   features_found=row is not None)
        finally:
            latency.labels("/predict", version).observe(time.perf_counter() - start)
            requests_total.labels("/predict", version, status).inc()

    @app.post("/feedback")
    def feedback(fb: FeedbackRequest) -> dict:
        if not pred_log.set_label(fb.request_id, fb.found):
            raise HTTPException(404, "unknown request_id")
        feedback_total.labels(str(fb.found).lower()).inc()
        return {"status": "recorded"}

    @app.get("/metrics")
    def metrics() -> Response:
        s = router.state()
        canary_pct.set(s.canary_percent)
        model_info.clear()
        if s.production:
            model_info.labels(s.production, "production").set(1)
        if s.candidate:
            model_info.labels(s.candidate, "candidate").set(1)
        return Response(generate_latest(prom), media_type=CONTENT_TYPE_LATEST)

    return app
