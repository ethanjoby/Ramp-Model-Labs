from __future__ import annotations

from dataclasses import dataclass, field
import statistics
import time
from typing import Any, Awaitable, Callable

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Modal CPU list price, used to price classical-model inference honestly.
CPU_DOLLARS_PER_SEC = 0.0000131

Emit = Callable[..., Awaitable[None]]


@dataclass
class TaskSpec:
    name: str
    task_type: str  # classification
    description: str
    labels: list[str]
    label_descriptions: dict[str, str] = field(default_factory=dict)
    input_description: str = ""
    template: str | None = None
    target_accuracy: float = 0.85
    max_latency_ms: float = 1500
    priority: str = "balanced"  # cost|accuracy|latency|balanced

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class LabContext:
    lab_id: str
    spec: TaskSpec
    train: list[dict]
    dev: list[dict]
    test: list[dict]
    emit: Emit
    artifact_dir: str


@dataclass
class Result:
    predictions: list[str]
    latencies_ms: list[float]
    cost_per_1k: float
    train_cost: float = 0.0
    train_seconds: float = 0.0
    size_kb: float | None = None
    artifact_path: str | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Strategy:
    key = "base"
    display_name = "Base"
    family = "classical"
    blurb = ""

    def available(self) -> tuple[bool, str]:
        return True, ""

    async def run(self, ctx: LabContext) -> Result:  # pragma: no cover
        raise NotImplementedError


def score(spec: TaskSpec, test: list[dict], res: Result) -> dict[str, Any]:
    y = [r["label"] for r in test]
    p = res.predictions
    labels = spec.labels
    lat = sorted(res.latencies_ms) or [0.0]
    cm = confusion_matrix(y, p, labels=labels).tolist()
    per_label = f1_score(y, p, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, p, labels=labels, average="macro", zero_division=0)), 4),
        "per_label_f1": {l: round(float(v), 3) for l, v in zip(labels, per_label)},
        "confusion": cm,
        "latency_p50_ms": round(statistics.median(lat), 3),
        "latency_p95_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 3),
        "cost_per_1k": res.cost_per_1k,
        "train_cost": round(res.train_cost, 4),
        "train_seconds": round(res.train_seconds, 2),
        "size_kb": res.size_kb,
        "n_eval": len(y),
        "notes": res.notes,
        "errors": [
            {"text": t["text"], "expected": a, "predicted": b}
            for t, a, b in zip(test, y, p) if a != b
        ][:12],
        **res.extra,
    }


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
