"""Prompted candidates: zero-shot LLMs and a DSPy-style optimized few-shot program.

They compete head-to-head with fine-tuned specialists. Sometimes prompting is enough.
"""
from __future__ import annotations

import asyncio
import difflib
import json
import os
import random
import time

from .. import llm
from ..config import get_settings
from .base import LabContext, Result, Strategy, TaskSpec

BATCH = 25


def _normalize(label: str, labels: list[str]) -> str:
    if label in labels:
        return label
    low = {l.lower(): l for l in labels}
    if label.lower().strip() in low:
        return low[label.lower().strip()]
    m = difflib.get_close_matches(label, labels, n=1, cutoff=0.0)
    return m[0] if m else labels[0]


def render_program(spec: TaskSpec, instruction: str, demos: list[dict]) -> str:
    lines = [instruction.strip(), "", "Allowed labels:"]
    for l in spec.labels:
        d = spec.label_descriptions.get(l)
        lines.append(f"- {l}" + (f": {d}" if d else ""))
    if demos:
        lines += ["", "Examples:"]
        lines += [f"Input: {d['text']}\nLabel: {d['label']}" for d in demos]
    return "\n".join(lines)


def default_instruction(spec: TaskSpec) -> str:
    return (f"You are a precise financial-operations classifier. Task: {spec.description} "
            f"Inputs look like: {spec.input_description or 'free text'}. Pick exactly one label per input.")


async def classify_batch(system: str, texts: list[str], labels: list[str], model: str,
                         usage: llm.Usage | None = None) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (f"Classify each of the {len(texts)} inputs.\n\n{numbered}\n\n"
              f'Return a JSON array of exactly {len(texts)} label strings, in order.')
    for attempt in range(3):
        try:
            out = await llm.complete_json(prompt, system=system, model=model, max_tokens=60 * len(texts) + 50,
                                          usage=usage, temperature=0)
            if isinstance(out, dict):
                out = next(iter(out.values()))
            out = [_normalize(str(x), labels) for x in out]
            if len(out) == len(texts):
                return out
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))
    return [labels[0]] * len(texts)


async def classify_one(system: str, text: str, labels: list[str], model: str,
                       usage: llm.Usage | None = None) -> str:
    out = await llm.complete(f"Input: {text}\nRespond with only the label.", system=system, model=model,
                             max_tokens=24, usage=usage, temperature=0)
    return _normalize(out.strip().strip('"').splitlines()[0] if out.strip() else "", labels)


async def evaluate_program(system: str, rows: list[dict], labels: list[str], model: str,
                           usage: llm.Usage) -> list[str]:
    chunks = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    sem = asyncio.Semaphore(4)

    async def run(chunk):
        async with sem:
            return await classify_batch(system, [r["text"] for r in chunk], labels, model, usage)

    results = await asyncio.gather(*(run(c) for c in chunks))
    return [p for chunk in results for p in chunk]


async def measure_single(system: str, rows: list[dict], labels: list[str], model: str) -> tuple[list[float], float]:
    """Real per-request latency and cost, the way production traffic would hit it."""
    u = llm.Usage()
    lat = []
    for r in rows[:4]:
        t0 = time.perf_counter()
        await classify_one(system, r["text"], labels, model, u)
        lat.append((time.perf_counter() - t0) * 1000)
    per_call = u.cost / max(1, len(lat))
    return lat, round(per_call * 1000, 5)


def _llm_ok() -> tuple[bool, str]:
    return (True, "") if llm.available() else (False, "Add ANTHROPIC_API_KEY to enable LLM candidates")


class ZeroShot(Strategy):
    family = "prompted"

    def __init__(self, model: str, key: str, name: str, family: str = "prompted", eval_cap: int = 200):
        self.model, self.key, self.display_name, self.family, self.eval_cap = model, key, name, family, eval_cap
        self.blurb = f"{model} with a task-specific instruction, no examples."

    def available(self):
        return _llm_ok()

    async def run(self, ctx: LabContext) -> Result:
        system = render_program(ctx.spec, default_instruction(ctx.spec), [])
        u = llm.Usage()
        preds = await evaluate_program(system, ctx.test, ctx.spec.labels, self.model, u)
        lat, cost_1k = await measure_single(system, ctx.test, ctx.spec.labels, self.model)
        return Result(predictions=preds, latencies_ms=lat, cost_per_1k=cost_1k, train_cost=0.0,
                      notes="no training", extra={"eval_spend": round(u.cost, 4)})


class OptimizedFewShot(Strategy):
    key = "optimized_prompt"
    display_name = "Optimized prompt program"
    family = "prompted"
    blurb = "DSPy-style search: the agent rewrites the instruction from error analysis and bootstraps few-shot demos."

    def available(self):
        return _llm_ok()

    async def run(self, ctx: LabContext) -> Result:
        s = get_settings()
        spec, model = ctx.spec, s.fast_model
        u = llm.Usage()
        rng = random.Random(13)
        dev = ctx.dev[:60]
        y_dev = [r["label"] for r in dev]

        def pick_demos(seed: int, k: int = 2) -> list[dict]:
            r = random.Random(seed)
            out = []
            for l in spec.labels:
                pool = [x for x in ctx.train if x["label"] == l]
                out += r.sample(pool, min(k, len(pool)))
            r.shuffle(out)
            return out

        base_instr = default_instruction(spec)
        base_preds = await evaluate_program(render_program(spec, base_instr, []), dev, spec.labels, model, u)
        base_acc = sum(a == b for a, b in zip(base_preds, y_dev)) / len(dev)
        await ctx.emit("trainer", "log", f"prompt optimizer · baseline instruction → dev acc {base_acc:.1%}")

        errors = [{"input": r["text"], "expected": a, "got": b}
                  for r, a, b in zip(dev, y_dev, base_preds) if a != b][:15]
        proposals = [base_instr]
        if errors:
            try:
                out = await llm.complete_json(
                    "You are optimizing a classification prompt. Current instruction:\n"
                    f"{base_instr}\n\nLabels: {spec.labels}\n\nMistakes on a dev set:\n{json.dumps(errors, indent=1)}\n\n"
                    "Write 2 improved instructions that fix these failure modes with concrete decision rules. "
                    'Return {"instructions": ["...", "..."]}.',
                    model=s.agent_model, max_tokens=1500, usage=u)
                proposals += [p for p in out.get("instructions", []) if isinstance(p, str)][:2]
            except Exception as e:  # keep going with what we have
                await ctx.emit("trainer", "log", f"prompt optimizer · proposal step failed ({e.__class__.__name__})")

        best = (base_acc, base_instr, [])
        trial = 0
        for instr in proposals:
            for seed in (rng.randint(0, 9999), rng.randint(0, 9999)):
                trial += 1
                demos = pick_demos(seed)
                preds = await evaluate_program(render_program(spec, instr, demos), dev, spec.labels, model, u)
                acc = sum(a == b for a, b in zip(preds, y_dev)) / len(dev)
                await ctx.emit("trainer", "log", f"prompt optimizer · trial {trial} ({len(demos)} demos) → dev acc {acc:.1%}")
                if acc > best[0]:
                    best = (acc, instr, demos)

        _, instr, demos = best
        system = render_program(spec, instr, demos)
        train_cost = u.cost
        preds = await evaluate_program(system, ctx.test, spec.labels, model, u)
        lat, cost_1k = await measure_single(system, ctx.test, spec.labels, model)
        path = os.path.join(ctx.artifact_dir, f"{self.key}.json")
        with open(path, "w") as f:
            json.dump({"model": model, "system": system, "labels": spec.labels}, f)
        return Result(predictions=preds, latencies_ms=lat, cost_per_1k=cost_1k, train_cost=train_cost,
                      artifact_path=path, notes=f"{trial} trials, {len(demos)} demos",
                      extra={"instruction": instr})
