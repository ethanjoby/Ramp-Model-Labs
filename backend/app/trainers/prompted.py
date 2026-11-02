"""Prompted candidates: zero-shot LLMs and a DSPy-style optimized few-shot program.

They compete head-to-head with fine-tuned specialists. Sometimes prompting is enough.
Every reported number comes from per-row requests, the shape the endpoint serves, so accuracy, latency
and cost per 1k are the deployed ones rather than those of a cheaper batched prompt.
"""
from __future__ import annotations

import asyncio
import difflib
import json
import os
import random
import re
import time

from .. import llm
from ..config import get_settings
from .base import LabContext, Result, Strategy, TaskSpec

BATCH = 25
CONCURRENCY = 8
UNKNOWN = "__unknown__"  # the model named no allowed label; scored as a wrong prediction, never served as one
_PREFIX = re.compile(r"^(?:label|answer|category|class|prediction)\s*[:\-]\s*", re.I)


def _normalize(label: object, labels: list[str]) -> str | None:
    """Map a model's answer onto one allowed label, or None when it did not name one.

    Inventing a label when nothing matches would inflate accuracy, so the fuzzy match has a floor and an
    answer that names two labels is ambiguous rather than the first one found.
    """
    s = _PREFIX.sub("", str(label or "").strip().strip("\"'`*.").strip()).strip().strip("\"'`*.")
    if not s:
        return None
    if s in labels:
        return s
    low = {l.lower(): l for l in labels}
    if s.lower() in low:
        return low[s.lower()]
    text = s.lower()
    hits = [l for l in labels if re.search(rf"(?<![a-z0-9_]){re.escape(l.lower())}(?![a-z0-9_])", text)]
    maximal = [h for h in hits if not any(h != o and h.lower() in o.lower() for o in hits)]
    if len(maximal) == 1:
        return maximal[0]
    if maximal:
        return None
    m = difflib.get_close_matches(text, list(low), n=1, cutoff=0.6)
    return low[m[0]] if m else None


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


def _parse_batch(out: object) -> list:
    """Line up a batch reply with its inputs, whatever shape the model chose.

    Accepted: a list, {"labels": [...]} (or any object with one list value), an int-keyed object, and a
    list of objects with a label field (sorted by their index when they carry one).
    """
    if isinstance(out, dict):
        if isinstance(out.get("labels"), list):
            out = out["labels"]
        elif out and all(re.fullmatch(r"-?\d+", str(k)) for k in out):
            out = [out[k] for k in sorted(out, key=lambda k: int(k))]
        else:
            lists = [v for v in out.values() if isinstance(v, list)]
            out = lists[0] if len(lists) == 1 else []
    if not isinstance(out, list):
        return []
    if out and all(isinstance(x, dict) for x in out):
        if all(isinstance(x.get("index"), int) for x in out):
            out = sorted(out, key=lambda x: x["index"])
        out = [x.get("label") for x in out]
    return out


async def classify_one(system: str, text: str, labels: list[str], model: str,
                       usage: llm.Usage | None = None) -> str | None:
    """One request, one input: the deployed path. None when the answer names no allowed label."""
    out = await llm.complete(f"Input: {text}\nRespond with only the label.", system=system, model=model,
                             purpose="classify", usage=usage)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    for candidate in (out.strip(), lines[0] if lines else "", lines[-1] if lines else ""):
        hit = _normalize(candidate, labels)
        if hit:
            return hit
    return None


async def _classify_rows(system: str, texts: list[str], labels: list[str], model: str,
                         usage: llm.Usage | None, sem: asyncio.Semaphore) -> tuple[list[str | None], list[float]]:
    async def one(t: str):
        async with sem:
            t0 = time.perf_counter()
            label = await classify_one(system, t, labels, model, usage)
            return label, (time.perf_counter() - t0) * 1000

    done = await asyncio.gather(*(one(t) for t in texts))
    return [d[0] for d in done], [d[1] for d in done]


async def classify_batch(system: str, texts: list[str], labels: list[str], model: str,
                         usage: llm.Usage | None = None) -> list[str | None]:
    """Many inputs in one request: the cheap path for bulk labeling, not the one that is scored.

    A reply that cannot be lined up with the inputs falls back to one request per row for this batch;
    a batch is never padded with a guess.
    """
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (f"Classify each of the {len(texts)} inputs.\n\n{numbered}\n\n"
              f'Return a JSON array of exactly {len(texts)} label strings, in order.')
    for attempt in range(3):
        try:
            out = await llm.complete_json(prompt, system=system, model=model, max_tokens=40 * len(texts) + 100,
                                          purpose="classify", usage=usage)
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        parsed = _parse_batch(out)
        if len(parsed) == len(texts):
            return [_normalize(x, labels) for x in parsed]
        break  # the shape is wrong, not the connection; per-row is more reliable than another batch
    preds, _ = await _classify_rows(system, texts, labels, model, usage, asyncio.Semaphore(CONCURRENCY))
    return preds


async def predict_rows(system: str, rows: list[dict], labels: list[str], model: str,
                       usage: llm.Usage | None = None) -> tuple[list[str], list[float]]:
    """Per-row predictions and the latency of each request. Unknown answers become UNKNOWN so scoring counts them wrong."""
    preds, lat = await _classify_rows(system, [r["text"] for r in rows], labels, model, usage,
                                      asyncio.Semaphore(CONCURRENCY))
    return [p or UNKNOWN for p in preds], lat


async def evaluate_program(system: str, rows: list[dict], labels: list[str], model: str,
                           usage: llm.Usage) -> list[str]:
    preds, _ = await predict_rows(system, rows, labels, model, usage)
    return preds


async def measure_single(system: str, rows: list[dict], labels: list[str], model: str,
                         n: int = 8) -> tuple[list[float], float]:
    """Serial single-request latency and the measured cost per 1k, the way production traffic would hit it."""
    u = llm.Usage()
    lat = []
    for r in rows[:n]:
        t0 = time.perf_counter()
        await classify_one(system, r["text"], labels, model, u)
        lat.append((time.perf_counter() - t0) * 1000)
    per_call = u.cost / max(1, len(lat))
    return lat, round(per_call * 1000, 5)


def _llm_ok() -> tuple[bool, str]:
    return (True, "") if llm.available() else (False, "Add ANTHROPIC_API_KEY to enable LLM candidates")


def _request_summary(model: str) -> dict:
    kw = llm.request_kwargs(model, "classify")
    return {"model": model, "effort": kw.get("output_config", {}).get("effort"),
            "thinking": kw.get("thinking", {}).get("type", "adaptive"), "max_tokens": kw["max_tokens"]}


def _write_program(path: str, model: str, system: str, labels: list[str]) -> str:
    with open(path, "w") as f:
        json.dump({"model": model, "system": system, "labels": labels}, f)
    return path


class _Prompted(Strategy):
    family = "prompted"

    async def benchmark(self, ctx: LabContext, artifact_path: str) -> tuple[list[float], float] | None:
        with open(artifact_path) as f:
            obj = json.load(f)
        return await measure_single(obj["system"], ctx.test, obj["labels"], obj["model"], n=16)


class ZeroShot(_Prompted):
    def __init__(self, model: str, key: str, name: str, family: str = "prompted"):
        self.model, self.key, self.display_name, self.family = model, key, name, family
        self.blurb = f"{model} with a task-specific instruction, no examples."

    def available(self):
        return _llm_ok()

    async def run(self, ctx: LabContext) -> Result:
        system = render_program(ctx.spec, default_instruction(ctx.spec), [])
        u = llm.Usage()
        preds, lat = await predict_rows(system, ctx.test, ctx.spec.labels, self.model, u)
        path = _write_program(os.path.join(ctx.artifact_dir, f"{self.key}.json"), self.model, system, ctx.spec.labels)
        return Result(predictions=preds, latencies_ms=lat, cost_per_1k=round(u.cost / max(1, len(preds)) * 1000, 5),
                      train_cost=0.0, eval_cost=round(u.cost, 4), artifact_path=path,
                      notes=f"no training · scored with {len(preds)} single requests",
                      extra={"eval_spend": round(u.cost, 4), "unknown_predictions": preds.count(UNKNOWN),
                             "llm_request": _request_summary(self.model)})


class OptimizedFewShot(_Prompted):
    key = "optimized_prompt"
    display_name = "Optimized prompt program"
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
                    model=s.agent_model, usage=u)
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
        ev = llm.Usage()  # the search is training spend; scoring the held-out set is evaluation spend
        preds, lat = await predict_rows(system, ctx.test, spec.labels, model, ev)
        path = _write_program(os.path.join(ctx.artifact_dir, f"{self.key}.json"), model, system, spec.labels)
        return Result(predictions=preds, latencies_ms=lat, cost_per_1k=round(ev.cost / max(1, len(preds)) * 1000, 5),
                      train_cost=train_cost, eval_cost=round(ev.cost, 4), artifact_path=path,
                      notes=f"{trial} trials, {len(demos)} demos · scored with {len(preds)} single requests",
                      extra={"instruction": instr, "eval_spend": round(ev.cost, 4),
                             "unknown_predictions": preds.count(UNKNOWN), "llm_request": _request_summary(model)})
