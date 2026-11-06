from __future__ import annotations

from dataclasses import dataclass, field
import math
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
    eval_cost: float = 0.0  # LLM spend during evaluation and measurement, so agent_spend is honest
    extra: dict[str, Any] = field(default_factory=dict)


class Strategy:
    key = "base"
    display_name = "Base"
    family = "classical"
    blurb = ""
    uses_gpu: str | None = None  # "local": the Mac's GPU (serialized); "remote": a cloud GPU; None: CPU or an API

    def available(self) -> tuple[bool, str]:
        return True, ""

    async def run(self, ctx: LabContext) -> Result:  # pragma: no cover
        raise NotImplementedError

    async def refit(self, ctx: LabContext, rows: list[dict], artifact_path: str) -> bool:
        """Retrain the chosen candidate on every row (train + dev + test) and overwrite its artifact.

        The reported scores stay those of the held-out fit; this only makes the deployed model use all the data.
        Strategies that cannot do this cheaply (remote or prompted ones) leave the default no-op.
        """
        return False

    async def benchmark(self, ctx: LabContext, artifact_path: str) -> tuple[list[float], float] | None:
        """Re-measure single-request latency (ms) and cost per 1k on an idle machine, after all training finished.

        Latency sampled while sibling candidates train is noisy; the orchestrator calls this serially and
        overwrites latency_p50_ms, latency_p95_ms and cost_per_1k with the result. None keeps the training-time numbers.
        """
        return None


UNKNOWN = "__unknown__"  # a prompted model that answered off-list: always wrong, never silently remapped


def _wilson(acc: float, n: int, z: float = 1.96) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def score(spec: TaskSpec, test: list[dict], res: Result) -> dict[str, Any]:
    y = [r["label"] for r in test]
    p = [UNKNOWN if v is None else v for v in res.predictions]
    labels = spec.labels
    present = [l for l in labels if l in set(y) or l in set(p)]
    absent = [l for l in labels if l not in present]
    lat = sorted(res.latencies_ms) or [0.0]
    cm = confusion_matrix(y, p, labels=labels).tolist()
    per_label = dict(zip(present, f1_score(y, p, labels=present, average=None, zero_division=0))) if present else {}
    acc = float(accuracy_score(y, p))
    notes = res.notes
    if absent:
        notes = (notes + " · " if notes else "") + f"not evaluated (no held-out rows): {', '.join(absent[:6])}"
    off_list = sum(1 for v in p if v == UNKNOWN)
    if off_list:
        notes = (notes + " · " if notes else "") + f"{off_list} answers were not one of the labels (counted wrong)"
    return {
        "accuracy": round(acc, 4),
        "accuracy_ci": _wilson(acc, len(y)),
        # Macro F1 over the labels that could actually be scored; an absent label is null, not a zero that drags the mean.
        "macro_f1": round(float(f1_score(y, p, labels=present, average="macro", zero_division=0)), 4) if present else 0.0,
        "per_label_f1": {l: (round(float(per_label[l]), 3) if l in per_label else None) for l in labels},
        "confusion": cm,
        "latency_p50_ms": round(statistics.median(lat), 3),
        "latency_p95_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 3),
        "cost_per_1k": res.cost_per_1k,
        "train_cost": round(res.train_cost, 4),
        "eval_cost": round(res.eval_cost, 4),
        "train_seconds": round(res.train_seconds, 2),
        "size_kb": res.size_kb,
        "n_eval": len(y),
        "notes": notes,
        "errors": [
            {"text": t["text"], "expected": a, "predicted": ("(off-list answer)" if b == UNKNOWN else b)}
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
