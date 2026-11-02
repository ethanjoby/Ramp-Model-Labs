"""Small specialist models trained locally in seconds. Fractions of a cent per million calls."""
from __future__ import annotations

import asyncio
import os
import pickle
import re
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .base import CPU_DOLLARS_PER_SEC, LabContext, Result, Strategy, Timer

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # older scikit-learn
    FrozenEstimator = None

BENCH_N = 60


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


def _xy(rows):
    return [r["text"] for r in rows], [r["label"] for r in rows]


def _time_single(pipe, texts: list[str]) -> list[float]:
    """Single-request latency, the way the endpoint serves it, after a warm-up call."""
    pipe.predict(texts[:2] or ["warm up"])
    lat = []
    for t in texts[:BENCH_N]:
        t0 = time.perf_counter()
        pipe.predict([t])
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


def _cost_per_1k(lat: list[float]) -> float:
    sec = sorted(lat)[len(lat) // 2] / 1000
    return round(sec * 1000 * CPU_DOLLARS_PER_SEC, 7)


def _slim(pipe: Pipeline) -> None:
    """Training-only state (Adam moments, early-stopping snapshots, loss curves) is most of an MLP pickle."""
    clf = pipe.steps[-1][1]
    inner = getattr(clf, "estimator", clf)
    if isinstance(inner, MLPClassifier):
        for attr in ("_optimizer", "_best_coefs", "_best_intercepts", "loss_curve_", "validation_scores_", "_no_improvement_count"):
            inner.__dict__.pop(attr, None)
        inner.coefs_ = [c.astype(np.float32) for c in inner.coefs_]
        inner.intercepts_ = [b.astype(np.float32) for b in inner.intercepts_]


def _calibrate(pipe: Pipeline, dev: list[dict]) -> str:
    """Sigmoid calibration on the dev split, so '92% confident' means roughly 92% right. Needs every class in dev."""
    if FrozenEstimator is None or len(dev) < 30:
        return ""
    clf = pipe.steps[-1][1]
    Xd, yd = _xy(dev)
    if set(yd) != set(clf.classes_):
        return ""
    try:
        feats = pipe[:-1].transform(Xd)
        cal = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid").fit(feats, yd)
        pipe.steps[-1] = ("clf", cal)
        return f"calibrated on {len(dev)} dev rows"
    except Exception:
        return ""


def _finish(ctx: LabContext, key: str, pipe: Pipeline, train_s: float, notes: str = "", extra=None) -> Result:
    texts = [r["text"] for r in ctx.test]
    preds = list(pipe.predict(texts))
    lat = _time_single(pipe, texts)
    _slim(pipe)
    path = os.path.join(ctx.artifact_dir, f"{key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(pipe, f)
    return Result(
        predictions=preds, latencies_ms=lat, cost_per_1k=_cost_per_1k(lat),
        train_cost=train_s * CPU_DOLLARS_PER_SEC, train_seconds=train_s,
        size_kb=round(os.path.getsize(path) / 1024, 1), artifact_path=path, notes=notes, extra=extra or {},
    )


class _Classical(Strategy):
    family = "classical"

    def build(self, n_docs: int) -> Pipeline:  # pragma: no cover
        raise NotImplementedError

    def fit(self, rows: list[dict], dev: list[dict] | None = None) -> tuple[Pipeline, str]:
        X, y = _xy(rows)
        pipe = self.build(len(X)).fit(X, y)
        note = _calibrate(pipe, dev or [])
        return pipe, note

    async def run(self, ctx: LabContext) -> Result:
        def go():
            with Timer() as t:
                pipe, note = self.fit(ctx.train, ctx.dev)
            return _finish(ctx, self.key, pipe, t.s, notes=note)
        return await asyncio.to_thread(go)

    async def refit(self, ctx: LabContext, rows: list[dict], artifact_path: str) -> bool:
        def go():
            # No dev split is left once every row trains, so calibrate with 3-fold CV instead of a held-out set.
            X, y = _xy(rows)
            pipe = self.build(len(X)).fit(X, y)
            if FrozenEstimator is not None and len(rows) >= 90 and min(y.count(c) for c in set(y)) >= 3:
                try:
                    feats = pipe[:-1].transform(X)
                    pipe.steps[-1] = ("clf", CalibratedClassifierCV(self.build(len(X)).steps[-1][1], method="sigmoid", cv=3).fit(feats, y))
                except Exception:
                    pass
            _slim(pipe)
            with open(artifact_path, "wb") as f:
                pickle.dump(pipe, f)
            return True
        return await asyncio.to_thread(go)

    async def benchmark(self, ctx: LabContext, artifact_path: str) -> tuple[list[float], float] | None:
        def go():
            with open(artifact_path, "rb") as f:
                pipe = pickle.load(f)
            lat = _time_single(pipe, [r["text"] for r in ctx.test] or ["warm up"])
            return lat, _cost_per_1k(lat)
        return await asyncio.to_thread(go)


class NaiveBayes(_Classical):
    key = "naive_bayes"
    display_name = "Naive Bayes baseline"
    blurb = "Word n-gram Complement NB. The floor every candidate has to beat."

    def build(self, n_docs: int) -> Pipeline:
        return Pipeline([("f", _features(n_docs, char=False)), ("clf", ComplementNB(alpha=0.3))])


class LinearSpecialist(_Classical):
    key = "linear_specialist"
    display_name = "Linear specialist (tuned)"
    blurb = "Word + char n-gram TF-IDF into logistic regression, with C chosen by dev-set search."
    _best_c = 4.0

    def build(self, n_docs: int, C: float | None = None) -> Pipeline:
        return Pipeline([("f", _features(n_docs)), ("clf", LogisticRegression(C=C or self._best_c, max_iter=2000))])

    async def run(self, ctx: LabContext) -> Result:
        loop = asyncio.get_running_loop()

        def go():
            X, y = _xy(ctx.train)
            Xd, yd = _xy(ctx.dev)
            use_cv = len(yd) < 30
            best, best_acc, best_c, trials = None, -1.0, None, []
            t0 = time.perf_counter()
            for C in (1.0, 4.0, 16.0):
                pipe = self.build(len(X), C)
                if use_cv:
                    folds = max(2, min(3, min(y.count(c) for c in set(y))))
                    acc = float(cross_val_score(pipe, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0)).mean())
                    pipe.fit(X, y)
                else:
                    pipe.fit(X, y)
                    acc = sum(a == b for a, b in zip(pipe.predict(Xd), yd)) / max(1, len(yd))
                trials.append({"C": C, "dev_acc": round(acc, 4)})
                asyncio.run_coroutine_threadsafe(
                    ctx.emit("trainer", "log", f"linear specialist · C={C} → {'3-fold' if use_cv else 'dev'} acc {acc:.1%}"), loop)
                if acc > best_acc:
                    best, best_acc, best_c = pipe, acc, C
            self._best_c = best_c
            note = _calibrate(best, ctx.dev)
            return _finish(ctx, self.key, best, time.perf_counter() - t0,
                           notes=f"best C={best_c}" + (" chosen by 3-fold CV (dev set too small)" if use_cv else "") + (f" · {note}" if note else ""),
                           extra={"search": trials})
        return await asyncio.to_thread(go)


class SvmSpecialist(_Classical):
    key = "svm_specialist"
    display_name = "Margin specialist (SVM)"
    blurb = "Linear SVM over the same n-gram features. Robust on noisy, abbreviated descriptors."

    def build(self, n_docs: int) -> Pipeline:
        return Pipeline([("f", _features(n_docs)), ("clf", LinearSVC(C=0.7))])


class TinyNeural(_Classical):
    key = "tiny_neural"
    display_name = "Tiny neural net"
    blurb = "One hidden-layer MLP over char n-grams. Picks up non-linear combinations of signals."

    def build(self, n_docs: int) -> Pipeline:
        return Pipeline([
            ("f", _features(n_docs, max_features=8000)),
            ("clf", MLPClassifier(hidden_layer_sizes=(128,), max_iter=40, early_stopping=n_docs >= 200, random_state=0, alpha=1e-4)),
        ])
