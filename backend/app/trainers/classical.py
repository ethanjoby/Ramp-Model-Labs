"""Small specialist models trained locally in seconds. Fractions of a cent per million calls."""
from __future__ import annotations

import asyncio
import os
import pickle
import re
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .base import CPU_DOLLARS_PER_SEC, LabContext, Result, Strategy, Timer


def _prep(text: str) -> str:
    # Digits become shape, not identity: "#1037 $45.00" and "#2211 $45.10" should look alike.
    return re.sub(r"\d", "0", text.lower())


def _features(n_docs: int, char: bool = True, max_features: int = 60000) -> FeatureUnion:
    # On a small upload every n-gram is seen once; dropping those would leave nothing to learn from.
    min_df = 2 if n_docs >= 300 else 1
    parts = [("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=max_features,
                                      preprocessor=_prep, token_pattern=r"(?u)\b\w[\w&'.*-]*\b"))]
    if char:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                              min_df=min_df, max_features=max_features, preprocessor=_prep)))
    return FeatureUnion(parts)


def _measure(pipe, texts: list[str]) -> tuple[list[str], list[float]]:
    preds = list(pipe.predict(texts))
    lat = []
    for t in texts[: min(len(texts), 200)]:
        t0 = time.perf_counter()
        pipe.predict([t])
        lat.append((time.perf_counter() - t0) * 1000)
    return preds, lat


def _finish(ctx: LabContext, key: str, pipe, train_s: float, notes: str = "", extra=None) -> Result:
    texts = [r["text"] for r in ctx.test]
    preds, lat = _measure(pipe, texts)
    path = os.path.join(ctx.artifact_dir, f"{key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(pipe, f)
    sec_per_call = sorted(lat)[len(lat) // 2] / 1000
    return Result(
        predictions=preds, latencies_ms=lat, cost_per_1k=round(sec_per_call * 1000 * CPU_DOLLARS_PER_SEC, 7),
        train_cost=train_s * CPU_DOLLARS_PER_SEC, train_seconds=train_s,
        size_kb=round(os.path.getsize(path) / 1024, 1), artifact_path=path, notes=notes, extra=extra or {},
    )


def _xy(rows):
    return [r["text"] for r in rows], [r["label"] for r in rows]


class NaiveBayes(Strategy):
    key = "naive_bayes"
    display_name = "Naive Bayes baseline"
    family = "classical"
    blurb = "Word n-gram Complement NB. The floor every candidate has to beat."

    async def run(self, ctx: LabContext) -> Result:
        def go():
            X, y = _xy(ctx.train)
            with Timer() as t:
                pipe = Pipeline([("f", _features(len(X), char=False)), ("clf", ComplementNB(alpha=0.3))]).fit(X, y)
            return _finish(ctx, self.key, pipe, t.s)
        return await asyncio.to_thread(go)


class LinearSpecialist(Strategy):
    key = "linear_specialist"
    display_name = "Linear specialist (tuned)"
    family = "classical"
    blurb = "Word + char n-gram TF-IDF into logistic regression, with C chosen by dev-set search."

    async def run(self, ctx: LabContext) -> Result:
        loop = asyncio.get_running_loop()

        def go():
            X, y = _xy(ctx.train)
            Xd, yd = _xy(ctx.dev)
            best, best_acc, best_c, trials = None, -1.0, None, []
            t0 = time.perf_counter()
            for C in (1.0, 4.0, 16.0):
                pipe = Pipeline([("f", _features(len(X))), ("clf", LogisticRegression(C=C, max_iter=2000))]).fit(X, y)
                acc = sum(a == b for a, b in zip(pipe.predict(Xd), yd)) / max(1, len(yd))
                trials.append({"C": C, "dev_acc": round(acc, 4)})
                asyncio.run_coroutine_threadsafe(
                    ctx.emit("trainer", "log", f"linear specialist · C={C} → dev acc {acc:.1%}"), loop)
                if acc > best_acc:
                    best, best_acc, best_c = pipe, acc, C
            return _finish(ctx, self.key, best, time.perf_counter() - t0,
                           notes=f"best C={best_c}",
                           extra={"search": trials})
        return await asyncio.to_thread(go)


class SvmSpecialist(Strategy):
    key = "svm_specialist"
    display_name = "Margin specialist (SVM)"
    family = "classical"
    blurb = "Linear SVM over the same n-gram features. Robust on noisy, abbreviated descriptors."

    async def run(self, ctx: LabContext) -> Result:
        def go():
            X, y = _xy(ctx.train)
            with Timer() as t:
                pipe = Pipeline([("f", _features(len(X))), ("clf", LinearSVC(C=0.7))]).fit(X, y)
            return _finish(ctx, self.key, pipe, t.s)
        return await asyncio.to_thread(go)


class TinyNeural(Strategy):
    key = "tiny_neural"
    display_name = "Tiny neural net"
    family = "classical"
    blurb = "One hidden-layer MLP over char n-grams. Picks up non-linear combinations of signals."

    async def run(self, ctx: LabContext) -> Result:
        def go():
            X, y = _xy(ctx.train)
            with Timer() as t:
                pipe = Pipeline([
                    ("f", _features(len(X), max_features=12000)),
                    ("clf", MLPClassifier(hidden_layer_sizes=(128,), max_iter=40, early_stopping=len(X) >= 200,
                                          random_state=0, alpha=1e-4)),
                ]).fit(X, y)
            return _finish(ctx, self.key, pipe, t.s)
        return await asyncio.to_thread(go)
