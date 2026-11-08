"""Distillation: the strongest expensive candidate teaches the cheapest local one.

DeepSeek-R1's small models were not trained with RL; they were fine-tuned on the big model's outputs, and
that beat RL on the small models themselves (DeepSeek-R1 report, §4.2 and §G.1). The classifier version is
simple: take rows nobody labeled, let the best expensive model (a prompted frontier model, a GPU LoRA, the
150M encoder) label them, and train the cheap student on the original data plus those labels. The student is
scored on the same held-out set as everyone else, and the teacher's own labels never touch that set.

It runs only when it can pay off: the teacher must lead the best cheap model by a real margin and cost
several times more to serve. Otherwise the cheap model already is the answer.
"""
from __future__ import annotations

import asyncio
import json
import os
import pickle
import random
from typing import Any

from .. import llm
from ..config import get_settings
from ..datasets.templates import TEMPLATES, generate
from ..trainers.base import LabContext, Result, Strategy, TaskSpec, score
from .data import _fold, _norm, augment

MIN_GAP = 0.02       # teacher must lead the best cheap model by this much accuracy
COST_RATIO = 3.0     # and cost at least this many times more per call
CHEAP_KEYS = {"finetuned_encoder", "embedding_specialist"}  # the locally served tier, next to every classical model
STUDENT_KEY = "distilled_minilm"


def _acc(r: dict) -> float:
    return float(r["metrics"]["accuracy"])


def pick(results: list[dict]) -> tuple[dict, dict] | None:
    """(teacher, best cheap candidate) when distilling is worth a try, else None."""
    done = [r for r in results if r.get("metrics") and "accuracy" in r["metrics"] and r["key"] != STUDENT_KEY]
    cheap = [r for r in done if r["family"] == "classical" or r["key"] in CHEAP_KEYS]
    expensive = [r for r in done if r not in cheap]
    if not cheap or not expensive:
        return None
    best_cheap = max(cheap, key=_acc)
    teacher = max(expensive, key=_acc)
    if _acc(teacher) - _acc(best_cheap) < MIN_GAP:
        return None
    if teacher["metrics"]["cost_per_1k"] < COST_RATIO * max(best_cheap["metrics"]["cost_per_1k"], 1e-7):
        return None
    return teacher, best_cheap


def unlabeled_pool(spec: TaskSpec, train: list[dict], held: list[dict], rng: random.Random, n: int) -> list[str]:
    """Rows nobody has labeled: fresh generator rows (labels thrown away) and variants of the training rows.
    Nothing that mirrors a held-out row once digits are folded, nothing already in training."""
    texts: list[str] = []
    if spec.template:  # over-draw: some generator rows collide with training rows and are filtered below
        texts += [r["text"] for r in generate(TEMPLATES[spec.template], 2 * n, seed=9091, noise=False, pool="train")]
    texts += [r["text"] for r in augment(train, rng, n, source="distilled")]
    blocked = {_fold(r["text"]) for r in held}
    known = {_norm(r["text"]) for r in train}
    out, seen = [], set()
    for t in texts:
        k = _norm(t)
        if k in seen or k in known or _fold(t) in blocked:
            continue
        seen.add(k)
        out.append(t)
    return out[:n]


async def teacher_labels(teacher: dict, spec: TaskSpec, texts: list[str], usage: llm.Usage | None) -> list[str | None]:
    """Run the teacher's saved artifact over the pool, whatever family it is. None marks an answer to drop."""
    path = teacher.get("artifact_path") or ""
    if os.path.isdir(path):
        from ..trainers.encoder import predict_encoder

        return [p["label"] for p in await asyncio.to_thread(predict_encoder, path, texts)]
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if hasattr(obj, "route"):  # the cascade: only rows it would decide itself, never the ones it escalates
            return [r["label"] if r["route"] != "review" else None for r in obj.route(texts)]
        return [str(x) for x in await asyncio.to_thread(obj.predict, texts)]
    with open(path) as f:
        obj = json.load(f)
    if "adapter" in obj:
        from ..trainers.finetuned import serve

        out: list[str | None] = []
        for i in range(0, len(texts), 200):
            labels, _ = await serve(obj, texts[i:i + 200])
            out += labels
        return out
    from ..trainers.prompted import classify_batch

    sem = asyncio.Semaphore(4)

    async def chunk(start: int) -> list[str | None]:
        async with sem:
            try:
                return await classify_batch(obj["system"], texts[start:start + 25], obj["labels"], obj["model"], usage=usage)
            except Exception:
                return [None] * len(texts[start:start + 25])

    parts = await asyncio.gather(*(chunk(i) for i in range(0, len(texts), 25)))
    return [p for part in parts for p in part]


def student() -> Strategy:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        from ..trainers.classical import LinearSpecialist

        st = LinearSpecialist()
        st.key, st.display_name = "distilled_linear", "Distilled linear specialist"
        return st
    from ..trainers.encoder import EncoderFinetune

    return EncoderFinetune(key=STUDENT_KEY, tag="student", display="Distilled MiniLM (22M, local)")


async def run(ctx: LabContext, results: list[dict], gpu: asyncio.Semaphore, usage: llm.Usage,
              on_start, log) -> dict[str, Any] | None:
    """Distill when it can pay off. `on_start(strategy, teacher)` registers the student as a candidate and returns
    its id; the returned dict is a results entry like the ones the trainer produces, plus the student strategy."""
    picked = pick(results)
    if not picked:
        best = max((r for r in results if r.get("metrics") and "accuracy" in r["metrics"]), key=_acc, default=None)
        if best:
            await log("distillation · skipped: no expensive candidate leads the cheap local models by "
                      f"{MIN_GAP:.0%} at {COST_RATIO:.0f}x the cost")
        return None
    teacher, cheap = picked
    s = get_settings()
    st = student()
    st.display_name = f"{st.display_name.split(' (')[0]} · student of {teacher['name']}"
    st.blurb = (f"{teacher['name']} labeled {s.distill_rows} unlabeled rows; the student trained on them plus the original data. "
                "The teacher never saw the held-out set, so the student's score is as honest as everyone else's.")
    cid = await on_start(st, teacher)
    ratio = teacher["metrics"]["cost_per_1k"] / max(cheap["metrics"]["cost_per_1k"], 1e-7)
    await log(f"distillation · {teacher['name']} ({_acc(teacher):.1%}) leads {cheap['name']} ({_acc(cheap):.1%}) at "
              f"{ratio:,.0f}x the cost; labeling {s.distill_rows} unlabeled rows with it to train a cheap student")
    rng = random.Random(4242)
    texts = unlabeled_pool(ctx.spec, ctx.train, ctx.test + ctx.dev, rng, s.distill_rows)
    before = usage.cost
    labels = await teacher_labels(teacher, ctx.spec, texts, usage)
    pseudo = [{"text": t, "label": l, "source": "distilled"} for t, l in zip(texts, labels) if l in ctx.spec.labels]
    await log(f"distillation · teacher labeled {len(pseudo)} of {len(texts)} rows (${usage.cost - before:.3f}); training the student on "
              f"{len(ctx.train)} + {len(pseudo)} rows")
    ctx2 = LabContext(ctx.lab_id, ctx.spec, ctx.train + pseudo, ctx.dev, ctx.test, ctx.emit, ctx.artifact_dir)
    if getattr(st, "uses_gpu", None) == "local":
        async with gpu:
            res: Result = await st.run(ctx2)
    else:
        res = await st.run(ctx2)
    m = score(ctx.spec, ctx.test, res)
    m["teacher"] = teacher["name"]
    m["distilled_rows"] = len(pseudo)
    m["notes"] = (m.get("notes") or "") + f" · student of {teacher['name']}: +{len(pseudo)} teacher-labeled rows"
    return {"key": st.key, "id": cid, "name": st.display_name, "family": "finetuned", "metrics": m,
            "artifact_path": res.artifact_path, "strategy": st, "spend": res.train_cost + res.eval_cost}
