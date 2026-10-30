"""Data curator: gathers seed examples, synthesizes more, dedupes, and builds honest splits."""
from __future__ import annotations

import asyncio
from collections import Counter
import json
import random
import re
from typing import Any

from .. import llm
from ..datasets.templates import TEMPLATES, generate
from ..trainers.base import Emit, TaskSpec

PROCEDURAL_N = 2400
TEST_CAP = 200
DEV_CAP = 150
AUGMENT_BELOW = 400  # training sets smaller than this get label-preserving variants of their own rows


def _norm(t: str) -> str:
    return re.sub(r"[\W_]+", " ", t.lower()).strip()


async def _synth_label(spec: TaskSpec, label: str, n: int, seeds: list[dict], usage: llm.Usage,
                       sem: asyncio.Semaphore) -> list[dict]:
    rows: list[dict] = []
    styles = ["terse and abbreviated like raw bank data", "messy with typos and odd casing",
              "verbose with extra context", "ambiguous edge cases near other labels", "typical, everyday cases"]
    batches = max(1, (n + 24) // 25)

    async def one(i: int):
        async with sem:
            ex = "\n".join(f"- {s['text']}  =>  {s['label']}" for s in seeds[:6])
            try:
                out = await llm.complete_json(
                    f"Generate 25 diverse, realistic inputs for a financial classifier.\nTask: {spec.description}\n"
                    f"Input format: {spec.input_description}\nAll labels: {spec.labels}\n"
                    f"Generate inputs whose correct label is: \"{label}\""
                    + (f" ({spec.label_descriptions[label]})" if label in spec.label_descriptions else "")
                    + f"\nStyle for this batch: {styles[i % len(styles)]}. Vary merchants, amounts, names, phrasing."
                    + (f"\nReal examples for reference:\n{ex}" if ex else "")
                    + '\nReturn {"inputs": ["...", ...]}', max_tokens=3000, usage=usage, temperature=1.0)
                return [{"text": t.strip(), "label": label, "source": "llm"}
                        for t in out.get("inputs", []) if isinstance(t, str) and 3 < len(t) < 600]
            except Exception:
                return []

    for chunk in await asyncio.gather(*(one(i) for i in range(batches))):
        rows += chunk
    return rows[:n]


def augment(rows: list[dict], rng: random.Random, target: int) -> list[dict]:
    """Label-preserving variants of training rows, so a tiny upload still teaches what to ignore."""
    if target <= 0 or not rows:
        return []
    out: list[dict] = []
    seen = {r["text"] for r in rows}
    attempts = 0
    while len(out) < target and attempts < target * 8:
        attempts += 1
        r = rng.choice(rows)
        t = r["text"]
        roll = rng.random()
        if roll < 0.3:
            t = re.sub(r"[#*|;:]+", " ", t)
        elif roll < 0.6:
            toks = t.split()
            if len(toks) >= 3:
                toks.pop(rng.randrange(len(toks)))
            t = " ".join(toks)
        elif roll < 0.85:
            t = re.sub(r"\$\s?\d[\d,]*(?:\.\d+)?", lambda _: f"${rng.uniform(1, 999):.2f}", t)
        else:
            t = re.sub(r"\d{3,}", lambda m: str(rng.randint(10 ** (len(m.group()) - 1), 10 ** len(m.group()) - 1)), t)
        t = re.sub(r"\s+", " ", t).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append({"text": t, "label": r["label"], "source": "augmented"})
    return out


def _stratified(rows: list[dict], n: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    take, rest = [], []
    frac = min(1.0, n / max(1, len(rows)))
    for items in by.values():
        rng.shuffle(items)
        k = max(1, round(len(items) * frac)) if len(items) > 2 else 0
        take += items[:k]
        rest += items[k:]
    return take, rest


async def curate(spec: TaskSpec, seed_examples: list[dict], emit: Emit, usage: llm.Usage) -> dict[str, Any]:
    rng = random.Random(42)
    real = [{"text": str(r["text"]).strip(), "label": r["label"], "source": "user"}
            for r in seed_examples if r.get("label") in spec.labels and str(r.get("text", "")).strip()]
    pool: list[dict] = []
    if real:
        await emit("data", "log", f"{len(real)} labeled examples from you, reserved for evaluation first")

    if spec.template:
        t = TEMPLATES[spec.template]
        rows = generate(t, PROCEDURAL_N, seed=rng.randint(0, 10_000))
        rows = [r for r in rows if r["label"] in spec.labels]
        pool += rows
        await emit("data", "log", f"procedural engine · {len(rows)} examples from the {t.name.lower()} generator (zero LLM spend)")

    if llm.available():
        per_label = 40 if spec.template else 110
        await emit("data", "log", f"synthesizer · asking Claude for ~{per_label} examples per label across 5 styles")
        sem = asyncio.Semaphore(6)
        seeds_by = {l: [r for r in real if r["label"] == l] for l in spec.labels}
        results = await asyncio.gather(*(_synth_label(spec, l, per_label, seeds_by[l] or real, usage, sem)
                                         for l in spec.labels))
        synth = [r for chunk in results for r in chunk]
        pool += synth
        await emit("data", "log", f"synthesizer · {len(synth)} examples generated (${usage.cost:.3f} so far)")

    # dedupe across everything; real data wins collisions
    seen, deduped_real, deduped_pool = set(), [], []
    for r in real:
        k = _norm(r["text"])
        if k not in seen:
            seen.add(k)
            deduped_real.append(r)
    for r in pool:
        k = _norm(r["text"])
        if k not in seen:
            seen.add(k)
            deduped_pool.append(r)
    dropped = len(real) + len(pool) - len(deduped_real) - len(deduped_pool)
    if dropped:
        await emit("data", "log", f"dedupe · removed {dropped} near-duplicates")

    # held-out test: prefer real examples so synthetic blind spots can't inflate scores
    n_real = len(deduped_real)
    if n_real >= 30:
        # A small upload keeps more of its rows for training; the test set is still only real data.
        frac = 0.5 if n_real >= 240 else 0.35 if n_real >= 90 else 0.3
        test, real_rest = _stratified(deduped_real, min(TEST_CAP, int(n_real * frac)), rng)
        eval_source = "your real examples"
        rest = real_rest + deduped_pool
    else:
        rest = deduped_real + deduped_pool
        test, rest = _stratified(rest, min(TEST_CAP, max(20, len(rest) // 6)), rng)
        eval_source = "held-out synthetic (upload real examples for a stricter test)"
    dev_n = min(DEV_CAP, max(2 * len(spec.labels), len(rest) // 8), max(1, len(rest) // 4))
    dev, train = _stratified(rest, dev_n, rng)

    if len(train) < 2 * len(spec.labels):
        raise ValueError("Not enough training data. Upload more labeled examples or add an ANTHROPIC_API_KEY.")

    if len(train) < AUGMENT_BELOW:
        extra = augment(train, rng, target=min(AUGMENT_BELOW, 4 * len(train)) - len(train))
        if extra:
            train += extra
            await emit("data", "log", f"augmenter · {len(extra)} variants of your training rows (separators, dropped "
                                      f"tokens, amounts); test and dev stay untouched")
    rng.shuffle(train)

    counts = Counter(r["label"] for r in train)
    stats = {
        "train": len(train), "dev": len(dev), "test": len(test),
        "sources": dict(Counter(r["source"] for r in train + dev + test)),
        "label_counts": {l: counts.get(l, 0) for l in spec.labels},
        "eval_source": eval_source,
        "samples": [{"text": r["text"], "label": r["label"], "source": r["source"]} for r in rng.sample(train, min(8, len(train)))],
        "imbalance": round(max(counts.values()) / max(1, min(counts.values() or [1])), 2),
    }
    await emit("data", "metric", f"dataset ready · {len(train)} train / {len(dev)} dev / {len(test)} test · eval on {eval_source}",
               {"dataset": stats})
    return {"train": train, "dev": dev, "test": test, "stats": stats}


def parse_examples(raw: str) -> list[dict]:
    """Accept CSV (text,label), JSONL, or 'text => label' lines."""
    import csv
    import io

    raw = raw.strip()
    if not raw:
        return []
    out: list[dict] = []
    if raw.startswith("[") or raw.startswith("{"):
        try:
            data = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
            for d in data:
                text = d.get("text") or d.get("input") or d.get("description")
                if text:
                    out.append({"text": str(text), "label": d.get("label") or d.get("category")})
            return out[:5000]
        except Exception:
            pass
    if "=>" in raw:
        for line in raw.splitlines():
            if "=>" in line:
                t, l = line.rsplit("=>", 1)
                out.append({"text": t.strip(), "label": l.strip() or None})
        return out[:5000]
    reader = csv.DictReader(io.StringIO(raw))
    if reader.fieldnames:
        cols = {c.lower().strip(): c for c in reader.fieldnames if c}
        tcol = next((cols[c] for c in ("text", "input", "description", "memo", "transaction", "message") if c in cols), None)
        lcol = next((cols[c] for c in ("label", "category", "class", "target", "gl", "intent") if c in cols), None)
        if tcol:
            for row in reader:
                if row.get(tcol):
                    out.append({"text": row[tcol], "label": (row.get(lcol) or "").strip() or None if lcol else None})
    return out[:5000]
