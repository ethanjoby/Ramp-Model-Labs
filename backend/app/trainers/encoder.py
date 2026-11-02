"""A real language model, fine-tuned locally.

A small pretrained transformer encoder gets a fresh classification head and is trained end to end on the
lab's training split, on this machine (Apple-silicon GPU via MPS, CUDA, or CPU). Evaluation and serving run
on CPU, because that is what a cheap serving box has and that is what the cost number prices.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from typing import Any, Callable

from ..config import get_settings
from .base import CPU_DOLLARS_PER_SEC, LabContext, Result, Strategy

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, BERT architecture, 90 MB
MAX_LEN = 64            # financial descriptors are short; longer inputs are truncated
BATCH = 32  # halved for small datasets so they still get enough optimizer steps
MAX_TRAIN_SECONDS = 150  # time box so a lab never waits on one candidate; the best finished epoch is kept
ARTIFACT_DIR = "encoder"


def model_name() -> str:
    return getattr(get_settings(), "local_lm_model", DEFAULT_MODEL) or DEFAULT_MODEL


def device_name() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _encode(tok, texts: list[str]):
    return tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")


def _predict(model, tok, device, texts: list[str], labels: list[str], batch: int = 64, probs: bool = False):
    import torch

    out: list[Any] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            logits = model(**_encode(tok, texts[i:i + batch]).to(device)).logits
            if probs:
                out += torch.softmax(logits.float(), dim=-1).cpu().tolist()
            else:
                out += [labels[j] for j in logits.argmax(-1).tolist()]
    return out


class EncoderFinetune(Strategy):
    key = "finetuned_encoder"
    display_name = "Fine-tuned MiniLM (22M, local)"
    family = "finetuned"
    blurb = ("A pretrained transformer encoder with a new classification head, trained end to end on this task's data, "
             "on this machine. You own the weights.")

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False, "pip install torch transformers"
        return True, ""

    async def run(self, ctx: LabContext) -> Result:
        loop = asyncio.get_running_loop()

        def log(msg: str) -> None:
            asyncio.run_coroutine_threadsafe(ctx.emit("trainer", "log", msg), loop)

        return await asyncio.to_thread(self._train, ctx, log)

    def _train(self, ctx: LabContext, log: Callable[[str], None]) -> Result:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        torch.manual_seed(0)
        name, labels = model_name(), ctx.spec.labels
        idx = {l: i for i, l in enumerate(labels)}
        dev_name = device_name()
        device = torch.device(dev_name)
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(labels)).to(device)

        X = [r["text"] for r in ctx.train]
        y = torch.tensor([idx[r["label"]] for r in ctx.train])
        Xd = [r["text"] for r in ctx.dev]
        yd = [idx[r["label"]] for r in ctx.dev]
        n = len(X)
        batch_size = 16 if n < 600 else BATCH
        steps_per_epoch = math.ceil(n / batch_size)
        epochs = 2 if n > 1500 else 3 if n > 800 else 5  # fewer rows need more passes to move a 12-way head
        total = epochs * steps_per_epoch
        warm = max(1, int(0.06 * total))
        # The fresh head learns fast, the pretrained encoder moves gently.
        head = [p for n_, p in model.named_parameters() if n_.startswith(("classifier", "pre_classifier"))]
        head_ids = {id(p) for p in head}
        body = [p for p in model.parameters() if id(p) not in head_ids]
        opt = torch.optim.AdamW([{"params": body, "lr": 5e-5}, {"params": head, "lr": 2e-3}], weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / total))
        log(f"minilm · {n} examples · {epochs} epochs · batch {batch_size} · training on {dev_name}")

        t0 = time.perf_counter()
        best_acc, best_state, step, time_boxed = -1.0, None, 0, False
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            loss_sum = 0.0
            for i in range(0, n, batch_size):
                b = perm[i:i + batch_size]
                batch = _encode(tok, [X[j] for j in b.tolist()]).to(device)
                out = model(**batch, labels=y[b].to(device))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                loss_sum += out.loss.item()
                step += 1
                if time.perf_counter() - t0 > MAX_TRAIN_SECONDS:
                    time_boxed = True
                    break
            model.eval()
            acc = (sum(p == labels[t] for p, t in zip(_predict(model, tok, device, Xd, labels), yd)) / len(yd)) if yd else float("nan")
            log(f"minilm · epoch {ep + 1}/{epochs} · loss {loss_sum / max(1, steps_per_epoch):.3f}"
                + (f" · dev acc {acc:.1%}" if yd else "") + f" · {time.perf_counter() - t0:.0f}s")
            if not yd or acc >= best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if time_boxed:
                log(f"minilm · {MAX_TRAIN_SECONDS}s time box reached, keeping the best epoch so far")
                break
        train_s = time.perf_counter() - t0
        if best_state is not None:
            model.load_state_dict(best_state)

        # score and time it the way it will be served: on CPU, one request at a time for latency
        cpu = torch.device("cpu")
        model.to(cpu).eval()
        texts = [r["text"] for r in ctx.test]
        preds = _predict(model, tok, cpu, texts, labels)
        lat = []
        for t in texts[:60]:
            s = time.perf_counter()
            _predict(model, tok, cpu, [t], labels)
            lat.append((time.perf_counter() - s) * 1000)

        path = os.path.join(ctx.artifact_dir, ARTIFACT_DIR)
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        tok.save_pretrained(path)
        with open(os.path.join(path, "labels.json"), "w") as f:
            json.dump(labels, f)
        size_kb = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) / 1024
        p50_s = sorted(lat)[len(lat) // 2] / 1000
        return Result(
            predictions=preds, latencies_ms=lat, cost_per_1k=round(p50_s * 1000 * CPU_DOLLARS_PER_SEC, 7),
            train_cost=train_s * CPU_DOLLARS_PER_SEC, train_seconds=train_s, size_kb=round(size_kb, 1), artifact_path=path,
            notes=(f"{name.split('/')[-1]} · {epochs} epochs on {dev_name} in {train_s:.0f}s"
                   + (" (time-boxed)" if time_boxed else "") + " · served on CPU"),
            extra={"device": dev_name, "epochs": epochs, "steps": step,
                   "best_dev_acc": None if math.isnan(best_acc) else round(best_acc, 4)},
        )


    async def benchmark(self, ctx: LabContext, artifact_path: str) -> tuple[list[float], float] | None:
        def go():
            import torch

            model, tok, labels = load_encoder(artifact_path)
            cpu = torch.device("cpu")
            texts = [r["text"] for r in ctx.test][:60] or ["warm up"]
            _predict(model, tok, cpu, texts[:4], labels)  # warm-up
            lat = []
            for t in texts:
                s = time.perf_counter()
                _predict(model, tok, cpu, [t], labels)
                lat.append((time.perf_counter() - s) * 1000)
            p50_s = sorted(lat)[len(lat) // 2] / 1000
            return lat, round(p50_s * 1000 * CPU_DOLLARS_PER_SEC, 7)
        return await asyncio.to_thread(go)


# ------------------------------------------------------------------ serving

_cache: dict[str, tuple[Any, Any, list[str]]] = {}


def load_encoder(path: str):
    if path not in _cache:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        with open(os.path.join(path, "labels.json")) as f:
            labels = json.load(f)
        model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        _cache[path] = (model, AutoTokenizer.from_pretrained(path), labels)
    return _cache[path]


def predict_encoder(path: str, texts: list[str]) -> list[dict]:
    import torch

    model, tok, labels = load_encoder(path)
    out = []
    for p in _predict(model, tok, torch.device("cpu"), texts, labels, probs=True):
        ranked = sorted(zip(labels, p), key=lambda x: -x[1])
        out.append({"label": ranked[0][0], "confidence": round(float(ranked[0][1]), 4),
                    "top": [{"label": l, "p": round(float(v), 4)} for l, v in ranked[:3]]})
    return out
