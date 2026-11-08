"""Distillation runs only when an expensive candidate clearly leads the cheap tier, and the student is scored honestly."""
import asyncio
import os
import pickle

from app.agents import distill
from app.datasets.templates import TEMPLATES, generate
from app.trainers.base import LabContext, TaskSpec, score
from app.trainers.classical import LinearSpecialist


def cand(key, acc, cost, family="classical", path=None):
    return {"key": key, "name": key, "family": family, "artifact_path": path,
            "metrics": {"accuracy": acc, "cost_per_1k": cost, "macro_f1": acc, "n_eval": 200, "latency_p50_ms": 1.0}}


def test_pick_requires_a_real_gap_and_a_real_cost_difference():
    cheap = cand("linear_specialist", 0.90, 0.001)
    assert distill.pick([cheap, cand("zero_shot_x", 0.95, 0.5, "prompted")])[0]["key"] == "zero_shot_x"
    assert distill.pick([cheap, cand("zero_shot_x", 0.91, 0.5, "prompted")]) is None  # one point is noise
    assert distill.pick([cheap, cand("finetuned_modernbert", 0.95, 0.002, "finetuned")]) is None  # not expensive enough
    assert distill.pick([cheap, cand("finetuned_encoder", 0.99, 0.5, "finetuned")]) is None  # MiniLM is the cheap tier itself
    assert distill.pick([cand("zero_shot_x", 0.95, 0.5, "prompted")]) is None


def test_student_trains_on_teacher_labels_and_reports_its_teacher(tmp_path, monkeypatch):
    t = TEMPLATES["fraud_detection"]
    rows = generate(t, 700, seed=21, noise=False)
    spec = TaskSpec(name=t.name, task_type="classification", description=t.description, labels=list(t.labels), template=t.key)
    logs = []

    async def emit(agent, kind, message="", data=None):
        logs.append(message)

    ctx = LabContext("lab_d", spec, rows[:500], rows[500:560], rows[560:], emit, str(tmp_path))
    # the "teacher" is a strong classical pipeline saved to disk; pretend it is an expensive prompted model
    teacher_res = asyncio.run(LinearSpecialist().run(ctx))
    teacher = cand("zero_shot_x", 0.99, 0.5, "prompted", path=teacher_res.artifact_path)
    cheap = cand("naive_bayes", 0.80, 0.0005, path=None)
    monkeypatch.setattr(distill, "student", lambda: _linear_student())
    monkeypatch.setattr(distill.get_settings(), "distill_rows", 300)
    started = []

    async def on_start(st, tch):
        started.append((st.key, tch["key"]))
        return "cand_student"

    out = asyncio.run(distill.run(ctx, [teacher, cheap], asyncio.Semaphore(1), distill.llm.Usage(), on_start, lambda m: emit("trainer", "log", m)))
    assert started == [("distilled_linear", "zero_shot_x")]
    assert out["key"] == "distilled_linear" and out["family"] == "finetuned" and out["id"] == "cand_student"
    m = out["metrics"]
    assert m["teacher"] == "zero_shot_x" and 0 < m["distilled_rows"] <= 300 and m["accuracy"] >= 0.85
    assert os.path.exists(out["artifact_path"]) and any("teacher labeled" in l for l in logs)
    with open(out["artifact_path"], "rb") as f:
        assert hasattr(pickle.load(f), "predict")


def test_unlabeled_pool_avoids_training_and_held_out_rows():
    t = TEMPLATES["invoice_triage"]
    rows = generate(t, 400, seed=3, noise=False)
    spec = TaskSpec(name=t.name, task_type="classification", description=t.description, labels=list(t.labels), template=t.key)
    import random

    pool = distill.unlabeled_pool(spec, rows[:300], rows[300:], random.Random(1), 200)
    assert len(pool) == 200 and len(set(pool)) == 200
    known = {distill._norm(r["text"]) for r in rows[:300]}
    held = {distill._fold(r["text"]) for r in rows[300:]}
    assert not any(distill._norm(p) in known or distill._fold(p) in held for p in pool)


def _linear_student():
    st = LinearSpecialist()
    st.key, st.display_name = "distilled_linear", "Distilled linear specialist"
    return st
