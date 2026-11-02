"""Four-level cascade: learned rules, then a small model, then an LLM, then a human.

Most traffic stops at the cheap levels, so spend scales with how hard the input is rather than with volume.
"""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
import os
import pickle
import re
import time

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .. import llm
from ..config import get_settings
from .base import CPU_DOLLARS_PER_SEC, LabContext, Result, Strategy, Timer
from .classical import _features, _xy
from .prompted import default_instruction, evaluate_program, measure_single, render_program

TOKEN = re.compile(r"[a-z][a-z&'*-]{2,}")
RULE_MIN_SUPPORT = 8
RULE_MIN_PURITY = 0.99
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def _tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def learn_rules(rows: list[dict]) -> dict[str, str]:
    """Tokens that almost always mean one label in training become deterministic rules."""
    seen: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for tok in _tokens(r["text"]):
            seen[tok][r["label"]] += 1
    rules = {}
    for tok, counts in seen.items():
        label, n = counts.most_common(1)[0]
        total = sum(counts.values())
        if total >= RULE_MIN_SUPPORT and n / total >= RULE_MIN_PURITY:
            rules[tok] = label
    return rules


class CascadeModel:
    def __init__(self, rules: dict[str, str], pipe, threshold: float, llm_system: str | None, llm_model: str | None,
                 labels: list[str]):
        self.rules, self.pipe, self.threshold = rules, pipe, threshold
        self.llm_system, self.llm_model, self.labels = llm_system, llm_model, labels

    def _rule(self, text: str) -> str | None:
        hits = {self.rules[t] for t in _tokens(text) if t in self.rules}
        return hits.pop() if len(hits) == 1 else None

    def route(self, texts: list[str]) -> list[dict]:
        """Levels one and two. An uncertain row keeps the model's guess and is marked for escalation."""
        probs = self.pipe.predict_proba(texts)
        classes = self.pipe.classes_
        out = []
        for text, p in zip(texts, probs):
            rule = self._rule(text)
            if rule:
                out.append({"label": rule, "confidence": 1.0, "route": "rule"})
                continue
            i = int(p.argmax())
            out.append({"label": str(classes[i]), "confidence": round(float(p[i]), 4),
                        "route": "model" if p[i] >= self.threshold else "review"})
        return out


def _pick_threshold(pipe, dev: list[dict], target: float) -> float:
    """Lowest threshold whose auto-handled dev rows clear the bar, so escalation stays as rare as possible."""
    if not dev:
        return 0.9
    X, y = _xy(dev)
    probs = pipe.predict_proba(X)
    classes = pipe.classes_
    guesses = [(float(p.max()), classes[int(p.argmax())] == t) for p, t in zip(probs, y)]
    for th in THRESHOLDS:
        kept = [ok for conf, ok in guesses if conf >= th]
        if kept and sum(kept) / len(kept) >= min(0.98, target + 0.05):
            return th
    return THRESHOLDS[-1]


class Cascade(Strategy):
    key = "cascade"
    display_name = "Cascade (rules, model, LLM, review)"
    family = "classical"
    blurb = "Learned rules handle the obvious, a small model the routine, an LLM the uncertain, and a person the rest."

    async def run(self, ctx: LabContext) -> Result:
        def train():
            X, y = _xy(ctx.train)
            with Timer() as t:
                pipe = Pipeline([("f", _features(len(X))), ("clf", LogisticRegression(C=4.0, max_iter=2000))]).fit(X, y)
                real = [r for r in ctx.train if r.get("source") not in ("augmented", "user-upsampled")]
                # Rules mined from augmented copies would count one real row several times; only real rows vote.
                rules = learn_rules(real) if len(real) >= 40 else {}
                threshold = _pick_threshold(pipe, ctx.dev, ctx.spec.target_accuracy)
            return pipe, rules, threshold, t.s

        pipe, rules, threshold, train_s = await asyncio.to_thread(train)
        model = get_settings().fast_model if llm.available() else None
        system = render_program(ctx.spec, default_instruction(ctx.spec), []) if model else None
        cascade = CascadeModel(rules, pipe, threshold, system, model, ctx.spec.labels)

        texts = [r["text"] for r in ctx.test]
        routed = await asyncio.to_thread(cascade.route, texts)
        lat = []
        for t in texts[:200]:
            t0 = time.perf_counter()
            await asyncio.to_thread(cascade.route, [t])
            lat.append((time.perf_counter() - t0) * 1000)
        cost_1k = sorted(lat)[len(lat) // 2] * CPU_DOLLARS_PER_SEC

        uncertain = [i for i, r in enumerate(routed) if r["route"] == "review"]
        if model and uncertain:
            rows = [ctx.test[i] for i in uncertain]
            u = llm.Usage()
            try:
                preds = await evaluate_program(system, rows, ctx.spec.labels, model, u)
                for i, label in zip(uncertain, preds):
                    routed[i].update(label=label, route="llm")
                llm_lat, per_1k = await measure_single(system, rows, ctx.spec.labels, model)
                cost_1k += per_1k * len(uncertain) / len(texts)
                share = len(uncertain) / len(texts)
                lat = [v + (sum(llm_lat) / len(llm_lat)) * share for v in lat]
            except Exception as e:
                await ctx.emit("trainer", "log", f"cascade · LLM level unavailable, escalating to review ({e.__class__.__name__})")

        path = os.path.join(ctx.artifact_dir, f"{self.key}.pkl")
        with open(path, "wb") as f:
            pickle.dump(cascade, f)
        self._threshold = threshold

        routes = Counter(r["route"] for r in routed)
        n = max(1, len(routed))
        auto = [r["label"] == row["label"] for r, row in zip(routed, ctx.test) if r["route"] != "review"]
        auto_acc = round(sum(auto) / len(auto), 4) if auto else None
        return Result(
            predictions=[r["label"] for r in routed], latencies_ms=lat, cost_per_1k=round(cost_1k, 7),
            train_cost=train_s * CPU_DOLLARS_PER_SEC, train_seconds=train_s,
            size_kb=round(os.path.getsize(path) / 1024, 1), artifact_path=path,
            notes=(f"{len(rules)} rules" + ("" if rules else " (too few real rows to mine any)") + f", confidence floor {threshold:.0%}. Routed: "
                   + ", ".join(f"{routes[k] / n:.0%} {k}" for k in ("rule", "model", "llm", "review") if routes[k])
                   + (". Rows sent to review are scored on the model's guess; reviewer time is not priced."
                      if routes["review"] else ".")),
            extra={"routes": {k: round(v / n, 4) for k, v in routes.items()}, "auto_accuracy": auto_acc},
        )

    async def refit(self, ctx: LabContext, rows: list[dict], artifact_path: str) -> bool:
        def go():
            with open(artifact_path, "rb") as f:
                old: CascadeModel = pickle.load(f)
            X, y = _xy(rows)
            pipe = Pipeline([("f", _features(len(X))), ("clf", LogisticRegression(C=4.0, max_iter=2000))]).fit(X, y)
            real = [r for r in rows if r.get("source") not in ("augmented", "user-upsampled")]
            rules = learn_rules(real) if len(real) >= 40 else {}
            with open(artifact_path, "wb") as f:
                pickle.dump(CascadeModel(rules, pipe, old.threshold, old.llm_system, old.llm_model, old.labels), f)
            return True
        return await asyncio.to_thread(go)

    async def benchmark(self, ctx: LabContext, artifact_path: str) -> tuple[list[float], float] | None:
        def go():
            with open(artifact_path, "rb") as f:
                cascade: CascadeModel = pickle.load(f)
            texts = [r["text"] for r in ctx.test][:60] or ["warm up"]
            cascade.route(texts[:2])
            lat = []
            for t in texts:
                t0 = time.perf_counter()
                cascade.route([t])
                lat.append((time.perf_counter() - t0) * 1000)
            return lat, round(sorted(lat)[len(lat) // 2] * CPU_DOLLARS_PER_SEC, 7)
        return await asyncio.to_thread(go)
