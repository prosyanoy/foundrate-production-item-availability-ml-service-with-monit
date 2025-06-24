"""Canary rollout: ramp traffic to a candidate model and auto-roll back if guardrails degrade.

The controller drives the same control plane the service watches (the registry's aliases.json),
sends labelled traffic through the public API, and compares candidate vs production on metrics it
collects from the responses themselves (status, latency, model-version header) and the true outcomes.
Metrics accumulate across steps, so later (larger) steps judge the candidate on more evidence.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Sequence

import httpx
import numpy as np

from foundrate.monitoring.psi import safe_auc
from foundrate.registry import ModelRegistry
from foundrate.simulate import Outcome, Sample, send_traffic


@dataclass(frozen=True)
class Guardrails:
    max_error_rate: float = 0.01  # candidate 5xx share
    max_p95_latency_ms: float = 250.0
    max_latency_ratio: float = 3.0  # candidate p95 / production p95
    max_auc_drop: float = 0.02  # tolerated production AUC - candidate AUC on the same live traffic ...
    auc_z: float = 2.0  # ... rolled back only when the drop exceeds the tolerance by z standard errors
    max_mean_score_shift: float = 0.10  # |mean score candidate - production|
    min_samples: int = 300  # candidate requests needed before quality guardrails apply


@dataclass
class ArmStats:
    n: int = 0
    errors: int = 0
    p95_latency_ms: float | None = None
    auc: float | None = None
    auc_se: float | None = None
    mean_score: float | None = None
    mean_outcome: float | None = None


@dataclass
class StepResult:
    percent: int
    requests: int
    production: ArmStats
    candidate: ArmStats
    violations: list[str]


@dataclass
class CanaryResult:
    status: str  # promoted | rolled_back
    candidate: str
    production: str
    reason: str
    steps: list[StepResult] = field(default_factory=list)
    finished_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def auc_std_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of an AUC estimate."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def summarize(outcomes: Sequence[Outcome]) -> ArmStats:
    if not outcomes:
        return ArmStats()
    ok = [o for o in outcomes if o.status == 200]
    auc = safe_auc([o.found for o in ok], [o.score for o in ok]) if ok else None
    n_pos = sum(o.found for o in ok)
    return ArmStats(
        n=len(outcomes),
        errors=len(outcomes) - len(ok),
        p95_latency_ms=float(np.percentile([o.latency_s for o in outcomes], 95) * 1000),
        auc=auc,
        auc_se=auc_std_error(auc, n_pos, len(ok) - n_pos) if auc is not None else None,
        mean_score=float(np.mean([o.score for o in ok])) if ok else None,
        mean_outcome=float(np.mean([o.found for o in ok])) if ok else None,
    )


def check_guardrails(prod: ArmStats, cand: ArmStats, g: Guardrails) -> list[str]:
    """Return human-readable violations (empty list = healthy)."""
    v: list[str] = []
    if cand.n == 0:
        return ["candidate received no traffic"]
    err = cand.errors / cand.n
    if err > g.max_error_rate:
        v.append(f"error rate {err:.1%} > {g.max_error_rate:.1%}")
    if cand.p95_latency_ms is not None:
        if cand.p95_latency_ms > g.max_p95_latency_ms:
            v.append(f"p95 latency {cand.p95_latency_ms:.0f}ms > {g.max_p95_latency_ms:.0f}ms")
        elif prod.p95_latency_ms and cand.p95_latency_ms > g.max_latency_ratio * prod.p95_latency_ms \
                and cand.p95_latency_ms > 20:
            v.append(f"p95 latency {cand.p95_latency_ms:.0f}ms > {g.max_latency_ratio}x production")
    if cand.n >= g.min_samples:
        if cand.auc is not None and prod.auc is not None:
            drop = prod.auc - cand.auc
            se = math.hypot(cand.auc_se or 0.0, prod.auc_se or 0.0)
            if drop - g.auc_z * se > g.max_auc_drop:
                v.append(f"AUC {cand.auc:.3f} is {drop:.3f} below production {prod.auc:.3f} "
                         f"(significant beyond the {g.max_auc_drop} tolerance)")
        if cand.mean_score is not None and prod.mean_score is not None \
                and abs(cand.mean_score - prod.mean_score) > g.max_mean_score_shift:
            v.append(f"mean score shifted {cand.mean_score - prod.mean_score:+.3f} vs production")
    return v


class CanaryController:
    def __init__(self, registry: ModelRegistry, client: httpx.Client, traffic: Callable[[int], Iterator[Sample]],
                 guardrails: Guardrails = Guardrails(), steps: Sequence[int] = (5, 25, 50),
                 requests_per_step: int = 1500, log: Callable[[str], None] = lambda _: None):
        self.registry, self.client, self.traffic = registry, client, traffic
        self.g, self.steps, self.requests_per_step, self.log = guardrails, tuple(steps), requests_per_step, log

    def run(self, candidate: str) -> CanaryResult:
        production = self.registry.aliases()["production"]
        if production is None:
            raise RuntimeError("no production model to compare against")
        if candidate == production:
            raise ValueError("candidate is already the production model")
        result = CanaryResult("rolled_back", candidate, production, "")
        prod_all: list[Outcome] = []
        cand_all: list[Outcome] = []
        try:
            for pct in self.steps:
                self.registry.start_canary(candidate, pct)
                # enough requests that the candidate arm reaches min_samples at this percentage (30% headroom for split noise)
                n = max(self.requests_per_step, math.ceil(self.g.min_samples * 130 / pct))
                outcomes = send_traffic(self.client, self.traffic(n))
                prod_all += [o for o in outcomes if o.version == production]
                cand_all += [o for o in outcomes if o.version == candidate]
                p, c = summarize(prod_all), summarize(cand_all)
                violations = check_guardrails(p, c, self.g)
                result.steps.append(StepResult(pct, n, p, c, violations))
                self.log(f"step {pct:>3}%: candidate n={c.n} auc={c.auc} vs prod auc={p.auc} "
                         f"errors={c.errors} -> {'FAIL ' + '; '.join(violations) if violations else 'ok'}")
                if violations:
                    result.reason = "; ".join(violations)
                    self.registry.rollback(result.reason)
                    return result
        except Exception as exc:  # never leave a half-open canary behind
            result.reason = f"controller error: {exc}"
            self.registry.rollback(result.reason)
            return result
        result.status, result.reason = "promoted", "all guardrails held at every step"
        self.registry.promote(candidate, reason=f"canary passed ({self.steps})")
        return result


def write_report(path, result: CanaryResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2))
