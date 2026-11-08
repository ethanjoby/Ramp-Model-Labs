"""The GPU LoRA candidate's backend half, without a GPU: roster selection, result mapping, the serving branch."""
import asyncio
import json
import os
import sys
import types

from app import inference
from app.trainers.base import LabContext, TaskSpec, score
import pytest

from app.trainers.finetuned import GPU_DOLLARS_PER_SEC, L4_DOLLARS_PER_SEC, LoraFinetune, parse_spec, roster, roster_models
from app.trainers.prompted import UNKNOWN

ns = types.SimpleNamespace


def test_roster_models_prefers_the_explicit_list_and_dedupes():
    assert roster_models(ns(modal_models="", modal_base_model="Qwen/Qwen3-1.7B", modal_heavy_model="")) == ["Qwen/Qwen3-1.7B"]
    assert roster_models(ns(modal_models="", modal_base_model="Qwen/Qwen3-1.7B", modal_heavy_model="Qwen/Qwen3-4B-Instruct-2507")) \
        == ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B-Instruct-2507"]
    assert roster_models(ns(modal_models=" Qwen/Qwen3-0.6B, Qwen/Qwen3-4B-Instruct-2507 ,Qwen/Qwen3-0.6B",
                            modal_base_model="ignored", modal_heavy_model="ignored")) == ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B-Instruct-2507"]


def test_specs_pick_a_gpu_from_the_size_and_quantize_when_bf16_would_not_fit():
    assert parse_spec("Qwen/Qwen3-1.7B") == ("Qwen/Qwen3-1.7B", "L4", False)
    assert parse_spec("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") == ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "L4", False)
    assert parse_spec("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") == ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "A100-40GB", False)
    assert parse_spec("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B@L4").quantize is True  # 15 GB of bf16 weights on a 24 GB card
    assert parse_spec("moonshotai/Moonlight-16B-A3B-Instruct") == ("moonshotai/Moonlight-16B-A3B-Instruct", "A100-80GB", False)
    assert parse_spec("moonshotai/Moonlight-16B-A3B-Instruct@a100-40gb:4bit") == ("moonshotai/Moonlight-16B-A3B-Instruct", "A100-40GB", True)
    assert parse_spec("moonshotai/Kimi-Linear-48B-A3B-Instruct") == ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "H100", True)
    assert parse_spec("Qwen/Qwen3-4B-Instruct-2507:bf16").quantize is False
    with pytest.raises(ValueError, match="Unknown GPU"):
        parse_spec("Qwen/Qwen3-1.7B@RTX4090")
    specs = roster(ns(modal_models="Qwen/Qwen3-1.7B, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B@L4 ,Qwen/Qwen3-1.7B@H100",
                      modal_base_model="x", modal_heavy_model=""))
    assert [(sp.short, sp.gpu, sp.method) for sp in specs] == [("Qwen3-1.7B", "L4", "LoRA"), ("DeepSeek-R1-Distill-Qwen-7B", "L4", "QLoRA")]
    st = LoraFinetune(specs[1], key="lora_x")
    assert st.display_name == "Fine-tuned DeepSeek-R1-Distill-Qwen-7B (QLoRA, L4)" and "DeepSeek" in st.blurb and "4-bit" in st.blurb


def test_unavailable_without_modal_enabled():
    ok, why = LoraFinetune().available()
    assert not ok and "MODAL_ENABLED" in why


class _FakeRemote:
    def __init__(self, out):
        self.out, self.calls = out, []

    async def aio(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.out


def _fake_modal(out):
    """A stand-in for `modal.Cls.from_name(app, "Trainer").with_options(gpu=...)()` whose methods record their calls."""
    remote = _FakeRemote(out)
    remote.gpus = []

    class Handle:
        def __init__(self, gpu):
            remote.gpus.append(gpu)
            self.train_and_eval = types.SimpleNamespace(remote=remote)
            self.predict = types.SimpleNamespace(remote=remote)

    cls = types.SimpleNamespace(with_options=lambda gpu=None, **kw: (lambda: Handle(gpu)), hydrate=lambda: None)
    Cls = types.SimpleNamespace(from_name=lambda app, name: cls)
    return types.SimpleNamespace(Cls=Cls), remote


def _ctx(tmp_path):
    spec = TaskSpec(name="Fraud", task_type="classification", description="Flag fraud.", labels=["fraud", "legitimate"],
                    input_description="transaction + risk signals")
    train = [{"text": f"row {i}", "label": "fraud" if i % 2 else "legitimate"} for i in range(10)]
    test = [{"text": "t0", "label": "fraud"}, {"text": "t1", "label": "legitimate"}, {"text": "t2", "label": "fraud"},
            {"text": "T0", "label": "fraud", "slice": "perturbed", "origin": "t0"}]
    logs = []

    async def emit(agent, kind, message="", data=None):
        logs.append(message)

    return LabContext("lab_lora", spec, train, [], test, emit, str(tmp_path)), logs


def test_run_maps_a_scored_job_into_a_result_with_confidences(tmp_path, monkeypatch):
    out = {"predictions": ["fraud", "legitimate", "banana", "fraud"], "confidences": [0.91, 0.84, 0.52, 0.88],
           "latencies_ms": [48.0, 51.0, 47.5], "batch_rows_per_s": 40.0, "train_seconds": 100.0, "total_seconds": 130.0,
           "adapter": "/adapters/lab_lora-lora_qwen/adapter", "adapter_kb": 2048.0, "base_model": "Qwen/Qwen3-1.7B",
           "epochs": 2, "rank": 8, "trainable_params": 17_000_000, "total_params": 1_700_000_000, "final_loss": 0.08, "gpu": "L4"}
    fake, remote = _fake_modal(out)
    monkeypatch.setitem(sys.modules, "modal", fake)
    ctx, logs = _ctx(tmp_path)
    st = LoraFinetune("Qwen/Qwen3-4B-Instruct-2507", key="lora_qwen3_4b_instruct_2507")
    assert st.display_name == "Fine-tuned Qwen3-4B-Instruct-2507 (LoRA, L4)" and st.uses_gpu == "remote"
    res = asyncio.run(st.run(ctx))

    args, kwargs = remote.calls[0]
    assert remote.gpus == ["L4"] and kwargs["gpu"] == "L4" and kwargs["quantize"] is False
    assert args[0] == ctx.train and args[1] == ctx.test and args[2] == ["fraud", "legitimate"] and kwargs["base_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert res.predictions == ["fraud", "legitimate", UNKNOWN, "fraud"]  # off-list answers are counted wrong, never remapped
    assert res.confidences == [0.91, 0.84, 0.52, 0.88]
    assert res.cost_per_1k == round(1000 * L4_DOLLARS_PER_SEC / 40.0, 6)
    assert res.train_cost == 130.0 * L4_DOLLARS_PER_SEC and res.train_seconds == 100.0 and res.size_kb == 2048.0
    assert res.extra["cost_estimated"] is True and res.extra["unknown_predictions"] == 1 and res.extra["batch_rows_per_s"] == 40.0
    assert "label scoring" in res.notes
    with open(res.artifact_path) as f:
        art = json.load(f)
    assert art["adapter"] == out["adapter"] and art["base_model"] == "Qwen/Qwen3-1.7B" and art["labels"] == ["fraud", "legitimate"]
    assert art["gpu"] == "L4" and art["quantize"] is False
    assert "Flag fraud." in art["instruction"]
    assert any("sending 10 rows" in m for m in logs) and any("40 rows/s" in m for m in logs)

    m = score(ctx.spec, ctx.test, res)
    assert m["n_eval"] == 3 and m["accuracy"] == round(2 / 3, 4)
    assert m["ece"] is not None and m["auto_rate_at_95"] is not None and m["robust_accuracy"] == 1.0


def test_run_without_confidences_still_scores(tmp_path, monkeypatch):
    out = {"predictions": ["fraud", "legitimate", "fraud", "fraud"], "latencies_ms": [50.0], "batch_rows_per_s": 0,
           "train_seconds": 10.0, "total_seconds": 12.0, "adapter": "/adapters/x/adapter", "adapter_kb": 1.0}
    fake, _ = _fake_modal(out)
    monkeypatch.setitem(sys.modules, "modal", fake)
    ctx, _ = _ctx(tmp_path)
    res = asyncio.run(LoraFinetune().run(ctx))
    assert res.confidences is None and res.cost_per_1k == round(1000 * L4_DOLLARS_PER_SEC / 40, 6)  # fallback throughput
    assert score(ctx.spec, ctx.test, res)["ece"] is None


def _candidate(tmp_path, cid, **extra):
    path = os.path.join(tmp_path, f"{cid}.json")
    with open(path, "w") as f:
        json.dump({"adapter": "/adapters/a/adapter", "base_model": "Qwen/Qwen3-1.7B", "instruction": "Flag fraud.",
                   "labels": ["fraud", "legitimate"], **extra}, f)
    return types.SimpleNamespace(id=cid, artifact_path=path)


def test_a_qlora_job_on_a_big_gpu_is_priced_at_that_gpu(tmp_path, monkeypatch):
    out = {"predictions": ["fraud", "legitimate", "fraud", "fraud"], "confidences": [0.9, 0.8, 0.7, 0.9], "latencies_ms": [90.0],
           "batch_rows_per_s": 20.0, "train_seconds": 300.0, "total_seconds": 400.0, "adapter": "/adapters/m/adapter", "adapter_kb": 4096.0,
           "base_model": "moonshotai/Moonlight-16B-A3B-Instruct", "epochs": 2, "rank": 8, "gpu": "A100-80GB", "quantized": False, "moe": True}
    fake, remote = _fake_modal(out)
    monkeypatch.setitem(sys.modules, "modal", fake)
    ctx, logs = _ctx(tmp_path)
    res = asyncio.run(LoraFinetune("moonshotai/Moonlight-16B-A3B-Instruct", key="lora_moonlight").run(ctx))
    rate = GPU_DOLLARS_PER_SEC["A100-80GB"]
    assert remote.gpus == ["A100-80GB"] and res.train_cost == 400.0 * rate and res.cost_per_1k == round(1000 * rate / 20.0, 6)
    assert "except the routed experts" in res.notes and res.extra["method"] == "LoRA" and res.extra["gpu"] == "A100-80GB"
    assert any("Modal A100-80GB" in m for m in logs)


def test_serving_accepts_scored_and_legacy_predict_shapes(tmp_path, monkeypatch):
    fake, remote = _fake_modal({"labels": ["fraud", "Legitimate"], "confidences": [0.9731, 0.61]})
    monkeypatch.setitem(sys.modules, "modal", fake)
    out, ms = asyncio.run(inference.predict(_candidate(tmp_path, "cand_new", gpu="A100-40GB", quantize=True), ["a", "b"]))
    assert out == [{"label": "fraud", "confidence": 0.9731, "top": None}, {"label": "legitimate", "confidence": 0.61, "top": None}]
    assert remote.calls[0][0][3] == ["a", "b"] and remote.calls[0][1]["base_model"] == "Qwen/Qwen3-1.7B" and ms >= 0
    assert remote.gpus == ["A100-40GB"] and remote.calls[0][1]["quantize"] is True  # served on the GPU it was trained on

    fake, _ = _fake_modal(["fraud", "nonsense"])  # an adapter served by a deploy that predates label scoring
    monkeypatch.setitem(sys.modules, "modal", fake)
    out, _ = asyncio.run(inference.predict(_candidate(tmp_path, "cand_old"), ["a", "b"]))
    assert out == [{"label": "fraud", "confidence": None, "top": None}, {"label": None, "confidence": None, "top": None}]
