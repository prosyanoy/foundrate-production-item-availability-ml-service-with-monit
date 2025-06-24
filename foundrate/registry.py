"""Filesystem model registry: immutable versioned artifacts + a small aliases file for rollout state."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


@dataclass
class LoadedModel:
    version: str
    model: Any
    metadata: dict[str, Any]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class ModelRegistry:
    """Layout: <root>/<name>/v0001/{model.joblib,metadata.json} and <root>/<name>/aliases.json.

    aliases.json = {"production": "v0001", "candidate": null, "canary_percent": 0, "history": [...]}
    The serving layer watches this file, so changing it is the rollout control plane.
    """

    def __init__(self, root: Path, name: str = "foundrate"):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def aliases_path(self) -> Path:
        return self.dir / "aliases.json"

    def versions(self) -> list[str]:
        return sorted(p.name for p in self.dir.iterdir() if p.is_dir() and p.name.startswith("v"))

    def register(self, model: Any, metadata: dict[str, Any]) -> str:
        existing = self.versions()
        version = f"v{(int(existing[-1][1:]) + 1) if existing else 1:04d}"
        vdir = self.dir / version
        vdir.mkdir()
        joblib.dump(model, vdir / "model.joblib")
        meta = {
            **metadata,
            "version": version,
            "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _atomic_write(vdir / "metadata.json", json.dumps(meta, indent=2, sort_keys=True))
        return version

    def load(self, version: str) -> LoadedModel:
        vdir = self.dir / version
        if not (vdir / "model.joblib").exists():
            raise KeyError(f"unknown model version {version!r}")
        meta = json.loads((vdir / "metadata.json").read_text())
        return LoadedModel(version, joblib.load(vdir / "model.joblib"), meta)

    def metadata(self, version: str) -> dict[str, Any]:
        return json.loads((self.dir / version / "metadata.json").read_text())

    # ---- aliases / rollout state -------------------------------------------------
    def aliases(self) -> dict[str, Any]:
        if not self.aliases_path.exists():
            return {"production": None, "candidate": None, "canary_percent": 0, "history": []}
        return json.loads(self.aliases_path.read_text())

    def _save(self, a: dict[str, Any], event: str, **details: Any) -> None:
        a["history"].append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **details})
        _atomic_write(self.aliases_path, json.dumps(a, indent=2))

    def promote(self, version: str, reason: str = "manual") -> None:
        self.load(version)  # validates the artifact exists
        a = self.aliases()
        previous = a["production"]
        a.update(production=version, candidate=None, canary_percent=0, previous_production=previous)
        self._save(a, "promote", version=version, previous=previous, reason=reason)

    def start_canary(self, candidate: str, percent: int) -> None:
        if not 0 <= percent <= 100:
            raise ValueError("percent must be within 0..100")
        self.load(candidate)
        a = self.aliases()
        if a["production"] is None:
            raise RuntimeError("cannot canary without a production model")
        a.update(candidate=candidate, canary_percent=percent)
        self._save(a, "canary", version=candidate, percent=percent)

    def rollback(self, reason: str) -> None:
        """Stop routing to the candidate; production is untouched."""
        a = self.aliases()
        rejected = a["candidate"]
        a.update(candidate=None, canary_percent=0)
        self._save(a, "rollback", version=rejected, reason=reason)
