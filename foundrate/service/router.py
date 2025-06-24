"""Model router: loads production/candidate from the registry and splits traffic deterministically."""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass

from foundrate.registry import LoadedModel, ModelRegistry


@dataclass(frozen=True)
class RouteState:
    production: str | None
    candidate: str | None
    canary_percent: int


def bucket(request_id: str) -> int:
    """Stable 0..99 bucket for a request id."""
    return int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 100


class ModelRouter:
    """Reloads automatically when aliases.json changes, so rollout/rollback needs no restart."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._lock = threading.Lock()
        self._sig: tuple | None = None
        self._state = RouteState(None, None, 0)
        self._models: dict[str, LoadedModel] = {}

    def state(self) -> RouteState:
        try:
            st = os.stat(self.registry.aliases_path)
            sig = (st.st_ino, st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            sig = None
        with self._lock:
            if sig != self._sig:
                a = self.registry.aliases()
                self._state = RouteState(a.get("production"), a.get("candidate"), int(a.get("canary_percent", 0)))
                self._sig = sig
            return self._state

    def _model(self, version: str) -> LoadedModel:
        with self._lock:
            if version not in self._models:
                self._models[version] = self.registry.load(version)
            return self._models[version]

    def select(self, request_id: str) -> LoadedModel | None:
        s = self.state()
        if s.production is None:
            return None
        if s.candidate and s.canary_percent > 0 and bucket(request_id) < s.canary_percent:
            return self._model(s.candidate)
        return self._model(s.production)
