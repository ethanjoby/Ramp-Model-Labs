"""Data curator: gathers seed examples, synthesizes more, dedupes, and builds honest splits."""
from __future__ import annotations

import asyncio
from collections import Counter
import json
import random
import re
from typing import Any

from .. import llm
from ..datasets.perturb import perturb, perturb_rows
from ..datasets.templates import TEMPLATES, generate
from ..trainers.base import Emit, TaskSpec

PROCEDURAL_N = 2400   # noisy rows for training, the way real ledgers arrive
CLEAN_N = 600         # noise-free rows for evaluation, so a correct answer is never marked wrong
UNSEEN_N = 300        # noise-free rows built only from merchants, vendors and phrasings the training pool never contains
TEST_CAP = 200
UNSEEN_SHARE = 0.4    # of a synthetic test set, the share drawn from the unseen pool
PERTURBED_N = 60      # noisy copies of core test rows, scored as a robustness slice rather than in the headline
DEV_CAP = 150
AUGMENT_BELOW = 400   # training sets smaller than this get label-preserving variants of their own rows
USER_SHARE = 0.25     # real rows are weighted up until they are at least this share of a template-fed training set
REPHRASE_MAX = 400    # uploaded rows the LLM rewrites in other styles (two rewrites each, fidelity-checked)


def _norm(t: str) -> str:
    return re.sub(r"[\W_]+", " ", t.lower()).strip()


def _fold(t: str) -> str:
    """Digit-folded key: 'UBER #1037 $45.00' and 'UBER #2211 $45.10' are the same row for leakage purposes."""
    return re.sub(r"\d", "0", _norm(t))


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
                    + (f"\nReal examples for reference (data, not instructions):\n<examples>\n{ex}\n</examples>" if ex else "")
                    + '\nReturn {"inputs": ["...", ...]}', max_tokens=3000, usage=usage, purpose="plan")
                return [{"text": t.strip(), "label": label, "source": "llm"}
                        for t in out.get("inputs", []) if isinstance(t, str) and 3 < len(t) < 600]
            except Exception:
                return []

    for chunk in await asyncio.gather(*(one(i) for i in range(batches))):
        rows += chunk
    return rows[:n]


def augment(rows: list[dict], rng: random.Random, target: int, source: str = "augmented") -> list[dict]:
    """Label-preserving variants of training rows: truncated merchants, other casings and separators, typos,
    reformatted amounts, processor prefixes, a dropped token. Variants rather than copies, because repeating the
    same tokens teaches a model to memorize them while rewrites teach it what to ignore."""
    if target <= 0 or not rows:
        return []
    out: list[dict] = []
    seen = {r["text"] for r in rows}
    attempts = 0
    while len(out) < target and attempts < target * 8:
        attempts += 1
        r = rng.choice(rows)
        t = perturb(r["text"], rng, strength=rng.choice([1, 2, 2]), train=True)
        if not t or t in seen:
            continue
        seen.add(t)
        out.append({"text": t, "label": r["label"], "source": source})
    return out


async def _rephrase(spec: TaskSpec, rows: list[dict], usage: llm.Usage | None, emit: Emit) -> list[dict]:
    """Kimi-K2-style rephrasing for the user's own rows: each row is rewritten twice in a different register
    (another bank's descriptor format, a person's shorthand), then every rewrite is re-classified and kept only
    when the model still reads the original label from it. Two varied tellings of a row beat two copies of it."""
    rows = rows[:REPHRASE_MAX]
    if not rows:
        return []
    model = llm.fast_model()
    sem = asyncio.Semaphore(4)
    labels = spec.labels

    async def one(chunk: list[dict]) -> list[dict]:
        numbered = "\n".join(f"{i + 1}. {r['text']}" for i, r in enumerate(chunk))
        async with sem:
            try:
                out = await llm.complete_json(
                    f"Task these rows belong to: {spec.description}\nInput format: {spec.input_description or 'free text'}\n"
                    f"Rewrite each of the {len(chunk)} rows below in two other realistic styles (e.g. how a different bank feed, "
                    "card processor, email client or person would write the same thing: abbreviations, truncation, other "
                    "casing and separators, amount formats, extra processor codes). Keep every fact that decides its category; "
                    "never change the merchant, vendor, intent or amounts beyond formatting.\n<rows>\n" + numbered + "\n</rows>\n"
                    'Return {"rewrites": [[two strings for row 1], [two strings for row 2], ...]} with exactly one entry per row, in order.',
                    model=model, max_tokens=4096, usage=usage, purpose="plan")
                rewrites = out.get("rewrites") if isinstance(out, dict) else None
                if not isinstance(rewrites, list) or len(rewrites) != len(chunk):
                    return []
            except Exception:
                return []
        cands = [{"text": t.strip(), "label": r["label"]} for r, rw in zip(chunk, rewrites) if isinstance(rw, list)
                 for t in rw if isinstance(t, str) and 3 < len(t.strip()) < 600 and t.strip() != r["text"]]
        if not cands:
            return []
        from ..trainers.prompted import classify_batch, default_instruction, render_program

        system = render_program(spec, default_instruction(spec), [])
        try:
            preds = await classify_batch(system, [c["text"] for c in cands], labels, model, usage=usage)
        except Exception:
            return []
        return [{**c, "source": "user-rephrased"} for c, pr in zip(cands, preds) if pr == c["label"]]

    chunks = [rows[i:i + 20] for i in range(0, len(rows), 20)]
    kept = [r for chunk in await asyncio.gather(*(one(c) for c in chunks)) for r in chunk]
    asked = 2 * len(rows)
    await emit("data", "log", f"rephraser · {len(kept)} of {asked} rewrites of your rows kept after re-labeling "
                              f"({asked - len(kept)} drifted and were dropped)")
    return kept


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


def _dedupe(groups: list[list[dict]]) -> tuple[list[list[dict]], int]:
    """Earlier groups win collisions: real rows over unseen-pool rows over clean generator rows over the noisy pool."""
    seen: set[str] = set()
    out: list[list[dict]] = []
    dropped = 0
    for rows in groups:
        kept = []
        for r in rows:
            k = _norm(r["text"])
            if k in seen:
                dropped += 1
                continue
            seen.add(k)
            kept.append(r)
        out.append(kept)
    return out, dropped


async def curate(spec: TaskSpec, seed_examples: list[dict], emit: Emit, usage: llm.Usage | None) -> dict[str, Any]:
    rng = random.Random(42)
    labels_n = len(spec.labels)
    ignored = {"unlabeled": 0, "off_spec": 0, "empty": 0}
    real: list[dict] = []
    off_spec_examples: list[str] = []
    unlabeled_texts: list[str] = []
    for r in seed_examples:
        text = str(r.get("text") or "").strip()
        label = r.get("label")
        if not text:
            ignored["empty"] += 1
            continue
        if label is None or str(label).strip() == "":
            unlabeled_texts.append(text)
            continue
        label = str(label).strip()
        if label not in spec.labels:
            ignored["off_spec"] += 1
            if len(off_spec_examples) < 3:
                off_spec_examples.append(label)
            continue
        real.append({"text": text, "label": label, "source": "user"})
    teach = bool(unlabeled_texts) and llm.available()
    if not teach:
        ignored["unlabeled"] = len(unlabeled_texts)
    if sum(ignored.values()):
        await emit("data", "log", f"ignored {sum(ignored.values())} uploaded rows: {ignored['unlabeled']} unlabeled, "
                                  f"{ignored['off_spec']} with labels outside the spec" + (f" (e.g. {', '.join(off_spec_examples)})" if off_spec_examples else "")
                                  + f", {ignored['empty']} empty")
    if real:
        await emit("data", "log", f"{len(real)} labeled examples from you, reserved for evaluation first")

    # Unlabeled rows are still the user's real inputs: the frontier model labels them, by agreement, for training only.
    teacher_rows: list[dict] = []
    teacher_stats: dict | None = None
    if teach:
        from . import teacher

        await emit("data", "log", f"teacher · {len(unlabeled_texts)} uploaded rows carry no label; asking "
                                  f"{llm.pretty(llm.agent_model())} three ways and keeping the agreements")
        teacher_rows, teacher_stats = await teacher.label_rows(spec, unlabeled_texts, usage, examples=real)
        await emit("data", "log", f"teacher · kept {teacher_stats['kept']} of {teacher_stats['asked']} "
                                  f"({teacher_stats['dropped']} disagreements dropped) · ${teacher_stats['cost']:.3f} · "
                                  "these rows train the model and never score it")

    pool: list[dict] = []
    clean: list[dict] = []
    unseen: list[dict] = []
    noise_rate = None
    if spec.template:
        t = TEMPLATES[spec.template]
        seed = rng.randint(0, 10_000)
        noisy = generate(t, PROCEDURAL_N, seed=seed, pool="train")
        truth = generate(t, PROCEDURAL_N, seed=seed, noise=False, pool="train")
        noise_rate = round(sum(a["label"] != b["label"] for a, b in zip(noisy, truth)) / max(1, len(noisy)), 4)
        pool += [r for r in noisy if r["label"] in spec.labels]
        clean += [r for r in generate(t, CLEAN_N, seed=seed + 1, noise=False, pool="train") if r["label"] in spec.labels]
        # The eval pool's merchants, vendors and phrasings exist nowhere in training: the new-merchant case production brings every week.
        unseen += [{**r, "source": "procedural-unseen", "slice": "unseen"}
                   for r in generate(t, UNSEEN_N, seed=seed + 2, noise=False, pool="eval") if r["label"] in spec.labels]
        await emit("data", "log", f"procedural engine · {len(pool)} training rows from the {t.name.lower()} generator "
                                  f"({noise_rate:.1%} planted label noise, like a real ledger), {len(clean)} clean rows for evaluation "
                                  f"and {len(unseen)} rows built only from merchants and phrasings the training rows never contain")

    if llm.available():
        per_label = 40 if spec.template else 110
        await emit("data", "log", f"synthesizer · asking Claude for ~{per_label} examples per label across 5 styles")
        sem = asyncio.Semaphore(6)
        seeds_by = {l: [r for r in real if r["label"] == l] for l in spec.labels}
        results = await asyncio.gather(*(_synth_label(spec, l, per_label, seeds_by[l] or real, usage, sem) for l in spec.labels))
        synth = [r for chunk in results for r in chunk]
        pool += synth
        await emit("data", "log", f"synthesizer · {len(synth)} examples generated (${(usage.cost if usage else 0.0):.3f} so far)")

    (real, unseen, clean, pool, teacher_rows), dropped = _dedupe([real, unseen, clean, pool, teacher_rows])
    if dropped:
        await emit("data", "log", f"dedupe · removed {dropped} near-duplicates")

    # held-out test: prefer real examples so synthetic blind spots can't inflate scores
    n_real = len(real)
    notes: list[str] = []
    if n_real >= 30:
        frac = 0.5 if n_real >= 240 else 0.35 if n_real >= 90 else 0.3
        test, real_rest = _stratified(real, min(TEST_CAP, int(n_real * frac)), rng)
        eval_source = "your real examples"
        if clean:
            dev, clean_rest = _stratified(clean, min(DEV_CAP, len(clean) // 2), rng)
            train = real_rest + pool + clean_rest
        else:
            rest = real_rest + pool
            dev_n = min(DEV_CAP, max(2 * labels_n, len(rest) // 8), max(1, len(rest) // 4))
            dev, train = _stratified(rest, dev_n, rng)
    elif clean:
        n_unseen = min(len(unseen), int(TEST_CAP * UNSEEN_SHARE)) if unseen else 0
        core, clean_rest = _stratified(clean, min(TEST_CAP - n_unseen, len(clean) * 2 // 3), rng)
        unseen_test, _ = _stratified(unseen, n_unseen, rng) if n_unseen else ([], [])
        test = core + unseen_test  # what is left of the unseen pool is discarded, never trained on
        dev, clean_rest = _stratified(clean_rest, min(DEV_CAP, len(clean_rest)), rng)
        train = real + pool + clean_rest
        eval_source = ("held-out synthetic, no planted noise"
                       + (f" · {len(unseen_test)} rows use merchants and phrasings never seen in training" if unseen_test else "")
                       + " (upload 30+ real examples for a stricter test)")
        if real:
            notes.append(f"{n_real} real rows are too few to evaluate on; they train the model instead")
    else:
        rest = real + pool
        test_n = min(TEST_CAP, max(len(rest) // 3, 2 * labels_n)) if not pool else min(TEST_CAP, max(20, len(rest) // 6))
        test, rest = _stratified(rest, test_n, rng)
        dev_n = min(DEV_CAP, max(2 * labels_n, len(rest) // 8), max(1, len(rest) // 4))
        dev, train = _stratified(rest, dev_n, rng)
        eval_source = ("your real examples" if test and all(r["source"] == "user" for r in test)
                       else "held-out synthetic (upload real examples for a stricter test)")

    # Robustness slice: noisy copies of core test rows. Scored next to the headline, never inside it.
    perturbed = perturb_rows([r for r in test if r.get("slice", "core") == "core"], rng, min(PERTURBED_N, len(test)))
    if perturbed:
        test = test + perturbed
        eval_source += f" · {len(perturbed)} perturbed copies scored separately"

    train += teacher_rows  # after the split: a teacher's mistakes must not become the test

    # nothing in train may be a digit-folded twin of a held-out row
    held = {_fold(r["text"]) for r in test + dev}
    before = len(train)
    train = [r for r in train if _fold(r["text"]) not in held]
    leaked = before - len(train)
    if leaked:
        await emit("data", "log", f"leakage guard · dropped {leaked} training rows that mirror a held-out row once numbers are folded")

    if len(train) < 2 * labels_n:
        raise ValueError("Not enough training data. Upload more labeled examples or add an ANTHROPIC_API_KEY.")

    # Rewrites of the user's rows in other registers, verified to still carry their label (needs a model key).
    user_rows = [r for r in train if r["source"] == "user"]
    if user_rows and llm.available():
        try:
            train += await _rephrase(spec, user_rows, usage, emit)
        except Exception as e:  # a bonus, never the reason a lab fails
            await emit("data", "log", f"rephraser · skipped ({e.__class__.__name__})")

    # a template's 2,400 rows must not drown the rows the user actually cares about
    user_train = [r for r in train if r["source"] in ("user", "teacher", "user-rephrased")]  # the user's real inputs, either way labeled
    if user_train and pool:
        share = len(user_train) / len(train)
        if share < USER_SHARE:
            need = int(USER_SHARE * len(train) / (1 - USER_SHARE)) - len(user_train)
            if need > 0:
                variants = augment(user_train, rng, need, source="user-variant")
                copies = [{**rng.choice(user_train), "source": "user-upsampled"} for _ in range(need - len(variants))]
                train += variants + copies
                await emit("data", "log", f"weighting · your {len(user_train)} rows weighted up to {USER_SHARE:.0%} of training: "
                                          f"{len(variants)} rewritten variants" + (f" and {len(copies)} repeats" if copies else ""))

    if len(train) < AUGMENT_BELOW:
        extra = augment(train, rng, target=min(AUGMENT_BELOW, 4 * len(train)) - len(train))
        if extra:
            train += extra
            await emit("data", "log", f"augmenter · {len(extra)} variants of your training rows (separators, dropped "
                                      f"tokens, amounts); test and dev stay untouched")
    rng.shuffle(train)

    counts = Counter(r["label"] for r in train)
    if noise_rate is not None:
        notes.append(f"training rows carry about {noise_rate:.1%} planted label noise; evaluation rows carry none")
    slices = dict(Counter(r.get("slice", "core") for r in test))
    headline = sum(v for k, v in slices.items() if k != "perturbed")
    stats = {
        "train": len(train), "dev": len(dev), "test": headline, "test_total": len(test), "slices": slices,
        "sources": dict(Counter(r["source"] for r in train + dev + test)),
        "label_counts": {l: counts.get(l, 0) for l in spec.labels},
        "eval_source": eval_source,
        "samples": [{"text": r["text"], "label": r["label"], "source": r["source"]} for r in rng.sample(train, min(8, len(train)))],
        "imbalance": round(max(counts.values()) / max(1, min(counts.values() or [1])), 2),
        "noise_rate": noise_rate, "ignored": ignored, "leak_removed": leaked, "notes": notes, "teacher": teacher_stats,
    }
    await emit("data", "metric", f"dataset ready · {len(train)} train / {len(dev)} dev / {headline} test · eval on {eval_source}",
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

    def put(text, label):
        text = str(text).strip()
        if text:
            out.append({"text": text, "label": (str(label).strip() or None) if label is not None else None})

    if raw.startswith("[") or raw.startswith("{"):
        try:
            data = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
            for d in data:
                if isinstance(d, dict):
                    text = d.get("text") or d.get("input") or d.get("description")
                    if text:
                        put(text, d.get("label") if d.get("label") is not None else d.get("category"))
            return out[:5000]
        except Exception:
            pass
    if "=>" in raw:
        for line in raw.splitlines():
            if "=>" in line:
                t, l = line.rsplit("=>", 1)
                put(t, l.strip() or None)
        return out[:5000]
    reader = csv.DictReader(io.StringIO(raw))
    if reader.fieldnames:
        cols = {c.lower().strip(): c for c in reader.fieldnames if c}
        tcol = next((cols[c] for c in ("text", "input", "description", "memo", "transaction", "message") if c in cols), None)
        lcol = next((cols[c] for c in ("label", "category", "class", "target", "gl", "intent") if c in cols), None)
        if tcol:
            for row in reader:
                if row.get(tcol):
                    put(row[tcol], (row.get(lcol) if lcol else None))
    return out[:5000]
