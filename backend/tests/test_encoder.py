"""The local encoder candidates train, score, save and serve; the bigger one warms up on the task's own rows first."""
import asyncio

import pytest

from app.datasets.templates import TEMPLATES, generate
from app.trainers.base import LabContext, TaskSpec, score
from app.trainers.encoder import EncoderFinetune, predict_encoder
from app.trainers.registry import strategies


def ctx(tmp_path, n=480):
    t = TEMPLATES["fraud_detection"]
    rows = generate(t, n + 120, seed=5, noise=False)
    spec = TaskSpec(name=t.name, task_type="classification", description=t.description, labels=list(t.labels),
                    input_description=t.input_hint, template=t.key)

    async def emit(*a, **k):
        pass

    return LabContext("lab_enc", spec, rows[:n], rows[n:n + 60], rows[n + 60:], emit, str(tmp_path))


def test_registry_has_the_local_encoders_serialized_on_the_gpu():
    enc = [st for st in strategies() if isinstance(st, EncoderFinetune)]
    assert [st.key for st in enc][:2] == ["finetuned_encoder", "finetuned_modernbert"]
    assert all(st.uses_gpu == "local" for st in enc) and enc[1].tacp and not enc[0].tacp
    # the local decoder LLM joins the roster when configured (the test environment leaves it off)
    from app.trainers.registry import local_llm

    q = local_llm("Qwen/Qwen3-0.6B")
    assert q.key == "finetuned_local_llm" and q.unfreeze_last == 4 and q.max_rows == 1500 and "Qwen3-0.6B" in q.display_name


@pytest.mark.parametrize("strategy", [
    EncoderFinetune(),
    EncoderFinetune(model="answerdotai/ModernBERT-base", key="finetuned_modernbert", tag="modernbert",
                    display="Fine-tuned ModernBERT-base (150M, local)", max_train_seconds=120, tacp=True),
], ids=["minilm", "modernbert"])
def test_encoder_trains_and_serves(tmp_path, strategy):
    c = ctx(tmp_path)
    res = asyncio.run(strategy.run(c))
    m = score(c.spec, c.test, res)
    assert m["accuracy"] >= 0.8, m["notes"]
    assert res.size_kb > 1000 and res.train_seconds > 0 and len(res.latencies_ms) == 60
    assert res.extra["tacp_steps"] > 0 if strategy.tacp else res.extra["tacp_steps"] == 0
    served = predict_encoder(res.artifact_path, [r["text"] for r in c.test[:10]])
    assert [p["label"] for p in served] == res.predictions[:10] and all(0 < p["confidence"] <= 1 for p in served)
