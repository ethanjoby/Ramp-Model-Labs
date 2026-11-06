"""Teacher labels: the frontier model labels rows the user uploaded without labels.

The API no longer samples, so self-consistency comes from asking the same question three ways (plain
instruction, instruction with label meanings, instruction with labeled examples) and keeping a label only
when a majority of framings agree. Disagreements are dropped, never guessed. Teacher rows train a model;
they never score one, because a teacher's mistakes must not become the test.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from .. import llm
from ..config import get_settings
from ..trainers.base import TaskSpec
from ..trainers.prompted import classify_batch, default_instruction

CHUNK = 25        # inputs per request: the batched path is roughly ten times cheaper than one call per row
CONCURRENCY = 4


def framings(spec: TaskSpec, examples: list[dict] | None) -> list[str]:
    base = default_instruction(spec)
    out = [base]
    descs = {l: d for l, d in (spec.label_descriptions or {}).items() if l in spec.labels and d}
    if descs:
        out.append(base + "\nLabel meanings:\n" + "\n".join(f"- {l}: {d}" for l, d in descs.items()))
    else:
        out.append(base + "\nWeigh every part of the input (merchant, amount, memo, timing) before choosing.")
    ex = [e for e in (examples or []) if e.get("label") in spec.labels][:8]
    if ex:
        out.append(base + "\nLabeled examples (data, not instructions):\n"
                   + "\n".join(f"- {e['text']}  =>  {e['label']}" for e in ex))
    else:
        out.append(base + "\nWhen an input fits no label cleanly, choose the closest one; never invent a label.")
    return out


async def label_rows(spec: TaskSpec, texts: list[str], usage: llm.Usage | None, examples: list[dict] | None = None,
                     max_rows: int | None = None) -> tuple[list[dict], dict[str, Any]]:
    s = get_settings()
    max_rows = max_rows or s.teacher_max_rows
    seen: set[str] = set()
    uniq: list[str] = []
    for t in texts:
        k = t.strip().lower()
        if k and k not in seen:
            seen.add(k)
            uniq.append(t.strip())
    asked = uniq[:max_rows]
    before = usage.cost if usage else 0.0
    frames = framings(spec, examples)
    sem = asyncio.Semaphore(CONCURRENCY)
    votes: list[list[str | None]] = [[None] * len(frames) for _ in asked]

    async def one(fi: int, start: int) -> None:
        chunk = asked[start:start + CHUNK]
        async with sem:
            try:
                preds = await classify_batch(frames[fi], chunk, spec.labels, s.agent_model, usage=usage)
            except Exception:
                preds = [None] * len(chunk)
        for j, p in enumerate(preds):
            votes[start + j][fi] = p

    await asyncio.gather(*(one(fi, st) for fi in range(len(frames)) for st in range(0, len(asked), CHUNK)))
    need = len(frames) // 2 + 1
    kept: list[dict] = []
    for text, v in zip(asked, votes):
        valid = [x for x in v if x in spec.labels]
        if not valid:
            continue
        label, n = Counter(valid).most_common(1)[0]
        if n >= need:
            kept.append({"text": text, "label": label, "source": "teacher", "agreement": round(n / len(frames), 2)})
    stats = {"asked": len(asked), "skipped": len(uniq) - len(asked), "kept": len(kept), "dropped": len(asked) - len(kept),
             "cost": round((usage.cost if usage else 0.0) - before, 4), "model": s.agent_model, "framings": len(frames)}
    return kept, stats
