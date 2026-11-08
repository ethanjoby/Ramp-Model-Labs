"""Frozen sentence embeddings with a linear head.

A pretrained embedding encoder (bge-small, 33M parameters) turns each row into a vector once; logistic
regression learns the labels on top. Nothing in the encoder moves, so the model generalizes from what the
text means rather than from merchant strings it has memorized, which is what matters when a merchant shows
up that was never in training. It trains in seconds and is the strongest cheap option on a tiny upload.

The pickled pipeline holds only the encoder's name; the weights are loaded from the Hugging Face cache on
first use, on whichever device the process has.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from ..config import get_settings
from .classical import _Classical, _calibrate, _finish

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
MAX_LEN = 64
_models: dict[tuple[str, str], tuple[Any, Any]] = {}


def model_name() -> str:
    return getattr(get_settings(), "local_embed_model", DEFAULT_MODEL) or DEFAULT_MODEL


def _load(name: str, device: str):
    key = (name, device)
    if key not in _models:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(torch.device(device)).eval()
        _models[key] = (tok, model)
    return _models[key]


def embed(name: str, texts: list[str], device: str = "cpu", batch: int = 64) -> np.ndarray:
    """L2-normalized embeddings. bge models are trained for CLS pooling; everything else gets mean pooling."""
    import torch

    tok, model = _load(name, device)
    out: list[np.ndarray] = []
    cls = "bge" in name.lower()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(model.device)
            h = model(**enc).last_hidden_state
            if cls:
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            v = torch.nn.functional.normalize(v.float(), dim=-1)
            out.append(v.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """sklearn transformer around `embed`, so the whole model is one picklable Pipeline the server already knows."""

    def __init__(self, name: str = DEFAULT_MODEL, device: str = "cpu"):
        self.name = name
        self.device = device  # used for the bulk training pass; serving overrides it with cpu

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return embed(self.name, list(X), self.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["device"] = "cpu"  # a pickled model is served on CPU, whatever trained it
        return state


def _bulk_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class EmbeddingSpecialist(_Classical):
    key = "embedding_specialist"
    uses_gpu = "local"  # the one-off embedding pass of the training set runs on the Mac's GPU when there is one

    def __init__(self, model: str | None = None):
        self.model = model or model_name()
        short = self.model.split("/")[-1]
        self.display_name = f"Embedding specialist ({short} + linear)"
        self.blurb = (f"Frozen {short} sentence embeddings with a logistic-regression head. The encoder never moves, so it "
                      "generalizes from meaning rather than memorized merchant strings; trains in seconds, strongest cheap option on tiny uploads.")

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False, "pip install torch transformers"
        return True, ""

    def build(self, n_docs: int, C: float = 8.0) -> Pipeline:
        return Pipeline([("emb", SentenceEmbedder(self.model, "cpu")), ("clf", LogisticRegression(C=C, max_iter=3000))])

    async def run(self, ctx):
        loop = asyncio.get_running_loop()

        def log(msg: str) -> None:
            asyncio.run_coroutine_threadsafe(ctx.emit("trainer", "log", msg), loop)

        def go():
            t0 = time.perf_counter()
            dev_name = _bulk_device()
            X = [r["text"] for r in ctx.train]
            y = [r["label"] for r in ctx.train]
            E = embed(self.model, X, dev_name)
            Xd = [r["text"] for r in ctx.dev]
            yd = [r["label"] for r in ctx.dev]
            Ed = embed(self.model, Xd, dev_name) if Xd else None
            log(f"embedding · {len(X)} rows embedded with {self.model.split('/')[-1]} on {dev_name} in {time.perf_counter() - t0:.0f}s")
            use_cv = len(yd) < 30
            best_c, best_acc, trials = 8.0, -1.0, []
            for C in (2.0, 8.0, 32.0):
                clf = LogisticRegression(C=C, max_iter=3000)
                if use_cv:
                    folds = max(2, min(3, min(y.count(c) for c in set(y))))
                    acc = float(cross_val_score(clf, E, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0)).mean())
                else:
                    acc = float(clf.fit(E, y).score(Ed, yd))
                trials.append({"C": C, "dev_acc": round(acc, 4)})
                if acc > best_acc:
                    best_acc, best_c = acc, C
            pipe = self.build(len(X), best_c)
            pipe.steps[-1][1].fit(E, y)  # the embedder is stateless; only the head needs the vectors
            note = _calibrate(pipe, ctx.dev) if Ed is not None else ""
            train_s = time.perf_counter() - t0
            return _finish(ctx, self.key, pipe, train_s,
                           notes=f"{self.model.split('/')[-1]} frozen · best C={best_c}" + (" by 3-fold CV" if use_cv else "") + (f" · {note}" if note else ""),
                           extra={"search": trials, "device": dev_name})

        return await asyncio.to_thread(go)

    async def refit(self, ctx, rows: list[dict], artifact_path: str) -> bool:
        def go():
            import pickle

            X = [r["text"] for r in rows]
            y = [r["label"] for r in rows]
            E = embed(self.model, X, _bulk_device())
            pipe = self.build(len(X))
            pipe.steps[-1][1].fit(E, y)
            with open(artifact_path, "wb") as f:
                pickle.dump(pipe, f)
            return True
        return await asyncio.to_thread(go)
