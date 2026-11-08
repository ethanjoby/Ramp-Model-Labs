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
    confidences: list[float] | None = None  # P(predicted label) per test row when the model has one; unlocks calibration metrics
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


def _ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    """Expected calibration error: how far 'I am 90% sure' is from being right 90% of the time."""
    if not conf:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c <= hi)]
        if idx:
            total += len(idx) / len(conf) * abs(sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return total


def automation_rate(conf: list[float], correct: list[bool], target: float = 0.95) -> tuple[float, float | None]:
    """The share of traffic a model can decide on its own while staying at least `target` accurate on what it
    decides, and the confidence floor that achieves it. Rows below the floor go to a person. Selective
    prediction is the number a review queue actually runs on; a model can be 90% accurate overall and still
    automate 70% of rows at 95%, or almost none if its confidence is uninformative."""
    if not conf:
        return 0.0, None
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    best_k, hits = 0, 0
    for k, i in enumerate(order, start=1):
        hits += int(correct[i])
        if hits / k >= target:
            best_k = k
    if best_k == 0:
        return 0.0, None
    return best_k / len(conf), conf[order[best_k - 1]]


def score(spec: TaskSpec, test: list[dict], res: Result) -> dict[str, Any]:
    """Score one candidate on the held-out set.

    Rows may carry a `slice`: "core" (the plain held-out draw), "unseen" (entities and phrasings kept out of
    training), or "perturbed" (noisy copies of core rows). The headline numbers are computed on core + unseen,
    i.e. every row that is its own case; perturbed copies are reported as a robustness slice so a copy and its
    original never both count toward the headline.
    """
    p_all = [UNKNOWN if v is None else v for v in res.predictions]
    conf_all = list(res.confidences) if res.confidences and len(res.confidences) == len(test) else None
    slice_of = [r.get("slice") or "core" for r in test]
    head = [i for i, sl in enumerate(slice_of) if sl != "perturbed"]
    y = [test[i]["label"] for i in head]
    p = [p_all[i] for i in head]
    labels = spec.labels
    present = [l for l in labels if l in set(y) or l in set(p)]
    absent = [l for l in labels if l not in present]
    lat = sorted(res.latencies_ms) or [0.0]
    cm = confusion_matrix(y, p, labels=labels).tolist() if y else [[0] * len(labels) for _ in labels]
    per_label = dict(zip(present, f1_score(y, p, labels=present, average=None, zero_division=0))) if present and y else {}
    acc = float(accuracy_score(y, p)) if y else 0.0
    notes = res.notes
    if absent:
        notes = (notes + " · " if notes else "") + f"not evaluated (no held-out rows): {', '.join(absent[:6])}"
    off_list = sum(1 for v in p if v == UNKNOWN)
    if off_list:
        notes = (notes + " · " if notes else "") + f"{off_list} answers were not one of the labels (counted wrong)"

    slices: dict[str, dict[str, Any]] = {}
    for name in ("core", "unseen", "perturbed"):
        idx = [i for i, sl in enumerate(slice_of) if sl == name]
        if idx:
            slices[name] = {"accuracy": round(sum(p_all[i] == test[i]["label"] for i in idx) / len(idx), 4), "n": len(idx)}
    # Robustness: of the perturbed copies whose original was answered right, how many stayed right.
    robust = None
    origins = {test[i]["text"]: p_all[i] == test[i]["label"] for i in head}
    pert = [i for i, sl in enumerate(slice_of) if sl == "perturbed" and origins.get(test[i].get("origin"), False)]
    if pert:
        robust = round(sum(p_all[i] == test[i]["label"] for i in pert) / len(pert), 4)

    calib: dict[str, Any] = {"ece": None, "auto_rate_at_95": None, "confidence_floor_95": None}
    if conf_all is not None and head:
        conf = [float(conf_all[i]) for i in head]
        correct = [p_all[i] == test[i]["label"] for i in head]
        rate, floor = automation_rate(conf, correct, 0.95)
        calib = {"ece": round(_ece(conf, correct), 4), "auto_rate_at_95": round(rate, 4),
                 "confidence_floor_95": None if floor is None else round(floor, 4)}

    return {
        "accuracy": round(acc, 4),
        "accuracy_ci": _wilson(acc, len(y)),
        # Macro F1 over the labels that could actually be scored; an absent label is null, not a zero that drags the mean.
        "macro_f1": round(float(f1_score(y, p, labels=present, average="macro", zero_division=0)), 4) if present and y else 0.0,
        "per_label_f1": {l: (round(float(per_label[l]), 3) if l in per_label else None) for l in labels},
        "confusion": cm,
        "slices": slices,
        "unseen_accuracy": slices.get("unseen", {}).get("accuracy"),
        "robust_accuracy": robust,
        **calib,
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
            {"text": test[i]["text"], "expected": test[i]["label"], "slice": slice_of[i],
             "predicted": ("(off-list answer)" if p_all[i] == UNKNOWN else p_all[i])}
            for i in head if test[i]["label"] != p_all[i]
        ][:12],
        **res.extra,
    }


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
