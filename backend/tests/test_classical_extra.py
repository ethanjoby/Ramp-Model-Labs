"""The committee and the embedding specialist train, calibrate, pickle and serve through the generic sklearn path."""
import asyncio
import pickle

from app.datasets.templates import TEMPLATES, generate
from app.inference import predict as serve
from app.trainers.base import LabContext, TaskSpec, score
from app.trainers.classical import Committee
from app.trainers.embedding import EmbeddingSpecialist


def ctx(tmp_path, n=600):
    t = TEMPLATES["support_routing"]
    rows = generate(t, n + 120, seed=9, noise=False)
    spec = TaskSpec(name=t.name, task_type="classification", description=t.description, labels=list(t.labels), template=t.key)

    async def emit(*a, **k):
        pass

    return LabContext("lab_cls", spec, rows[:n], rows[n:n + 60], rows[n + 60:], emit, str(tmp_path))


def _serve(res, texts):
    class C:
        id = f"c-{res.artifact_path}"
        artifact_path = res.artifact_path

    return asyncio.run(serve(C(), texts))[0]


def test_committee_votes_with_probabilities(tmp_path):
    c = ctx(tmp_path)
    res = asyncio.run(Committee().run(c))
    m = score(c.spec, c.test, res)
    assert m["accuracy"] >= 0.85 and m["ece"] is not None and len(res.confidences) == len(c.test)
    assert "members: lr, svm, nb, mlp" in res.notes
    out = _serve(res, [r["text"] for r in c.test[:5]])
    assert [o["label"] for o in out] == res.predictions[:5] and all(o["confidence"] and o["top"] for o in out)


def test_embedding_specialist_is_small_fast_and_serves_from_a_pickle(tmp_path):
    c = ctx(tmp_path)
    res = asyncio.run(EmbeddingSpecialist().run(c))
    m = score(c.spec, c.test, res)
    assert m["accuracy"] >= 0.85 and res.size_kb < 500, (m["accuracy"], res.size_kb)  # the pickle holds the head and a model name only
    assert res.extra["search"] and "frozen" in res.notes
    with open(res.artifact_path, "rb") as f:
        pipe = pickle.load(f)
    assert pipe.steps[0][1].device == "cpu"  # served on CPU whatever trained it
    out = _serve(res, [r["text"] for r in c.test[:5]])
    assert [o["label"] for o in out] == res.predictions[:5] and all(0 < o["confidence"] <= 1 for o in out)
