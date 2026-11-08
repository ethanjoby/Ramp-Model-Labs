"""Load a deployed winner and serve predictions, whatever family it came from."""
from __future__ import annotations

import asyncio
import json
import math
import os
import pickle
import time

from .config import get_settings
from .db import Candidate

_cache: dict[str, object] = {}


def _load(c: Candidate):
    if c.id not in _cache:
        path = c.artifact_path
        if not path:
            raise ValueError("This model has no deployable artifact.")
        if os.path.isdir(path):  # a fine-tuned transformer saved with save_pretrained
            _cache[c.id] = ("encoder", path)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                _cache[c.id] = ("sk", pickle.load(f))  # artifacts are written only by our trainers
        else:
            with open(path) as f:
                _cache[c.id] = ("json", json.load(f))
    return _cache[c.id]


async def predict(c: Candidate, texts: list[str]) -> tuple[list[dict], float]:
    kind, obj = _load(c)
    t0 = time.perf_counter()
    out: list[dict] = []
    if kind == "encoder":
        from .trainers.encoder import predict_encoder

        out = await asyncio.to_thread(predict_encoder, obj, texts)
    elif kind == "sk" and hasattr(obj, "route"):
        from .trainers.prompted import classify_one

        out = obj.route(texts)
        if obj.llm_system:
            for t, r in zip(texts, out):
                if r["route"] == "review":
                    try:
                        r.update(label=await classify_one(obj.llm_system, t, obj.labels, obj.llm_model), route="llm")
                    except Exception:
                        pass  # stays in review; a person is the last level of the cascade
        for r in out:
            r["top"] = None
    elif kind == "sk":
        labels = list(obj.predict(texts))
        clf = obj.steps[-1][1]
        conf = [None] * len(texts)
        if hasattr(clf, "predict_proba"):
            conf = [float(max(p)) for p in obj.predict_proba(texts)]
        elif hasattr(clf, "decision_function"):
            for i, row in enumerate(obj.decision_function(texts)):
                row = list(row) if hasattr(row, "__len__") else [row, -row]
                ex = [math.exp(v - max(row)) for v in row]
                conf[i] = max(ex) / sum(ex)
        top = []
        if hasattr(clf, "predict_proba"):
            for p in obj.predict_proba(texts):
                ranked = sorted(zip(clf.classes_, p), key=lambda x: -x[1])[:3]
                top.append([{"label": l, "p": round(float(v), 4)} for l, v in ranked])
        for i, l in enumerate(labels):
            out.append({"label": str(l), "confidence": round(conf[i], 4) if conf[i] is not None else None,
                        "top": top[i] if top else None})
    elif "adapter" in obj:
        from .trainers.finetuned import serve

        labels, confs = await serve(obj, texts)
        out = [{"label": l, "confidence": (round(c, 4) if c is not None else None), "top": None} for l, c in zip(labels, confs)]
    else:
        from .trainers.prompted import classify_one

        for t in texts:
            out.append({"label": await classify_one(obj["system"], t, obj["labels"], obj["model"]),
                        "confidence": None, "top": None})
    return out, (time.perf_counter() - t0) * 1000


def evict(candidate_id: str) -> None:
    """Drop a deleted or retrained candidate's model from memory."""
    _cache.pop(candidate_id, None)
