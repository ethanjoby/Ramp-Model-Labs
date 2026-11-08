"""LoRA / QLoRA fine-tunes of open language models on Modal GPUs (see modal_app.py for the recipe).

The roster is a list of base models, each with the GPU it trains and serves on and whether the base is loaded
in 4-bit (QLoRA). `MODAL_MODELS` entries read `org/model[@GPU][:4bit]`; the GPU defaults from the model's
size and 4-bit switches on by itself when the bf16 weights would not fit half the card:

    Qwen/Qwen3-1.7B                                -> L4, LoRA
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B        -> A100-40GB, LoRA        (or @L4 -> QLoRA)
    moonshotai/Moonlight-16B-A3B-Instruct          -> A100-80GB, LoRA
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
from typing import NamedTuple

from ..config import get_settings
from .base import LabContext, Result, Strategy
from .prompted import UNKNOWN, _normalize, default_instruction

# Modal list prices per GPU-second (modal.com/pricing) and card memory; training cost is measured GPU seconds at these rates.
GPU_DOLLARS_PER_SEC = {"T4": 0.000164, "L4": 0.000222, "A10G": 0.000306, "L40S": 0.000542, "A100-40GB": 0.000583,
                       "A100-80GB": 0.000694, "H100": 0.001097, "H200": 0.001261, "B200": 0.001736}
GPU_MEMORY_GB = {"T4": 16, "L4": 24, "A10G": 24, "L40S": 48, "A100-40GB": 40, "A100-80GB": 80, "H100": 80, "H200": 141, "B200": 192}
L4_DOLLARS_PER_SEC = GPU_DOLLARS_PER_SEC["L4"]
FALLBACK_REQ_PER_SEC = 40         # only if the job somehow reports no measured throughput
VOLUME = "rml-adapters"           # the volume modal_app.py writes adapters to
VOLUME_MOUNT = "/adapters/"
TRAINER = "Trainer"               # the Modal class modal_app.py deploys
_deployed: dict[str, tuple[float, bool, str]] = {}


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]  # Qwen/Qwen3-1.7B -> Qwen3-1.7B


def params_b(base_model: str) -> float:
    """Billions of parameters read off the model name: 'Qwen3-1.7B' -> 1.7, 'Moonlight-16B-A3B' -> 16 (total, not active)."""
    found = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", short_name(base_model))]
    return max(found) if found else 2.0


def default_gpu(base_model: str) -> str:
    p = params_b(base_model)
    return "L4" if p <= 4.5 else "A100-40GB" if p <= 9 else "A100-80GB" if p <= 20 else "H100"


def needs_quantization(base_model: str, gpu: str) -> bool:
    """bf16 weights plus optimizer state and activations need about twice the weight bytes; past half the card, go 4-bit."""
    return params_b(base_model) * 2 > 0.5 * GPU_MEMORY_GB.get(gpu, 24)


class ModelSpec(NamedTuple):
    base: str
    gpu: str
    quantize: bool

    @property
    def short(self) -> str:
        return short_name(self.base)

    @property
    def method(self) -> str:
        return "QLoRA" if self.quantize else "LoRA"

    @property
    def family_name(self) -> str:
        org = self.base.split("/")[0].lower()
        return {"deepseek-ai": "DeepSeek", "moonshotai": "Moonshot (Kimi)", "qwen": "Qwen", "meta-llama": "Llama",
                "google": "Gemma", "mistralai": "Mistral"}.get(org, org)


def parse_spec(entry: str) -> ModelSpec:
    """'org/model[@GPU][:4bit]' -> ModelSpec, with the GPU defaulted from the size and 4-bit from the fit."""
    entry = entry.strip()
    quantize: bool | None = None
    if entry.lower().endswith((":4bit", ":qlora")):
        entry, quantize = entry.rsplit(":", 1)[0], True
    elif entry.lower().endswith(":bf16"):
        entry, quantize = entry.rsplit(":", 1)[0], False
    base, _, gpu = entry.partition("@")
    gpu = gpu.strip().upper() or default_gpu(base)
    if gpu not in GPU_DOLLARS_PER_SEC:
        raise ValueError(f"Unknown GPU {gpu!r} for {base}; choose one of {', '.join(GPU_DOLLARS_PER_SEC)}")
    if quantize is None:
        quantize = needs_quantization(base, gpu)
    return ModelSpec(base.strip(), gpu, quantize)


def roster(settings) -> list[ModelSpec]:
    """The GPU roster: MODAL_MODELS when set, else the base plus the optional heavy one. Deduplicated by base."""
    listed = [m.strip() for m in str(getattr(settings, "modal_models", "") or "").split(",") if m.strip()]
    if not listed:
        listed = [settings.modal_base_model]
        if getattr(settings, "modal_heavy_model", ""):
            listed.append(settings.modal_heavy_model)
    out: list[ModelSpec] = []
    for e in listed:
        spec = parse_spec(e)
        if all(spec.base != o.base for o in out):
            out.append(spec)
    return out


def roster_models(settings) -> list[str]:
    return [s.base for s in roster(settings)]


def deployed(app_name: str) -> tuple[bool, str]:
    """Whether `modal deploy modal_app.py` has been run for this account. Cached (5 min when found, 1 min
    when not) because strategies() runs on every chat turn and capabilities check."""
    now = time.monotonic()
    hit = _deployed.get(app_name)
    if hit and now - hit[0] < (300 if hit[1] else 60):
        return hit[1], hit[2]

    def check() -> tuple[bool, str]:
        try:
            import modal

            modal.Cls.from_name(app_name, TRAINER).hydrate()
            return True, ""
        except Exception as e:
            return False, f"Run `uv run modal deploy modal_app.py` first ({e.__class__.__name__})"

    # Modal's sync client must not run on the API's event loop; a thread keeps the lookup isolated.
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        try:
            ok, why = ex.submit(check).result(timeout=20)
        except concurrent.futures.TimeoutError:
            ok, why = False, "Modal did not answer in 20s"
    _deployed[app_name] = (now, ok, why)
    return ok, why


def trainer(gpu: str):
    """A handle on the deployed Trainer class pinned to one GPU type for this call."""
    import modal

    return modal.Cls.from_name(get_settings().modal_app_name, TRAINER).with_options(gpu=gpu)()


async def serve(artifact: dict, texts: list[str]) -> tuple[list[str | None], list[float | None]]:
    """Run a saved adapter over `texts` on the GPU it was trained on; (labels, confidences), legacy list shape accepted."""
    gpu = artifact.get("gpu") or "L4"
    res = await trainer(gpu).predict.remote.aio(artifact["adapter"], artifact["instruction"], artifact["labels"], texts,
                                                base_model=artifact.get("base_model") or get_settings().modal_base_model,
                                                quantize=bool(artifact.get("quantize", False)))
    if isinstance(res, dict):
        labels, confs = res.get("labels") or [], res.get("confidences") or []
    else:
        labels, confs = list(res), []
    out_labels = [_normalize(p, artifact["labels"]) for p in labels]
    out_conf = [float(c) for c in confs] if len(confs) == len(out_labels) else [None] * len(out_labels)
    return out_labels, out_conf


async def adapter_size_kb(out: dict, volume: str = VOLUME) -> float | None:
    """Size of the saved adapter, from the job's own measurement or a volume listing; None rather than a guess."""
    if isinstance(out.get("adapter_kb"), (int, float)) and out["adapter_kb"] > 0:
        return round(float(out["adapter_kb"]), 1)
    adapter = str(out.get("adapter") or "")
    if not adapter.startswith(VOLUME_MOUNT):
        return None
    try:
        import modal

        entries = await modal.Volume.from_name(volume).listdir.aio(adapter[len(VOLUME_MOUNT):], recursive=True)
        total = sum(int(getattr(e, "size", 0) or 0) for e in entries)
        return round(total / 1024, 1) if total else None
    except Exception:
        return None


class LoraFinetune(Strategy):
    family = "finetuned"
    uses_gpu = "remote"

    def __init__(self, base_model: str | ModelSpec | None = None, key: str = "lora_qwen"):
        if isinstance(base_model, ModelSpec):
            self.spec = base_model
        else:
            self.spec = parse_spec(base_model or get_settings().modal_base_model)
        self.base_model, self.gpu, self.quantize = self.spec.base, self.spec.gpu, self.spec.quantize
        self.key = key
        self.display_name = f"Fine-tuned {self.spec.short} ({self.spec.method}, {self.gpu})"
        how = ("the base loaded in 4-bit NF4 with low-rank adapters trained on top (QLoRA)" if self.quantize
               else "low-rank adapters on every linear layer")
        self.blurb = (f"{self.spec.family_name}'s open-weight {self.spec.short}: {how}, trained on a Modal {self.gpu} with label-only "
                      "targets and served by scoring every allowed label, so it never answers off-list and every prediction "
                      "carries a confidence. A real language model you own, served from a GPU.")

    def available(self) -> tuple[bool, str]:
        s = get_settings()
        if not s.modal_enabled:
            return False, "Set MODAL_ENABLED=true after `modal deploy modal_app.py`"
        try:
            import modal  # noqa: F401
        except ImportError:
            return False, "uv add modal"
        return deployed(s.modal_app_name)

    async def run(self, ctx: LabContext) -> Result:
        instruction = default_instruction(ctx.spec)
        await ctx.emit("trainer", "log", f"{self.spec.method.lower()} · {self.spec.short} · sending {len(ctx.train)} rows to a Modal {self.gpu}")
        t0 = time.perf_counter()
        out = await trainer(self.gpu).train_and_eval.remote.aio(ctx.train, ctx.test, ctx.spec.labels, instruction, f"{ctx.lab_id}-{self.key}",
                                                                 base_model=self.base_model, quantize=self.quantize, gpu=self.gpu)
        wall_s = time.perf_counter() - t0
        gpu = out.get("gpu") if out.get("gpu") in GPU_DOLLARS_PER_SEC else self.gpu
        rate = GPU_DOLLARS_PER_SEC[gpu]
        quantized = bool(out.get("quantized", self.quantize))
        preds = [_normalize(p, ctx.spec.labels) or UNKNOWN for p in out["predictions"]]
        confs = out.get("confidences")
        confidences = ([float(c) for c in confs] if isinstance(confs, list) and len(confs) == len(preds) else None)
        path = os.path.join(ctx.artifact_dir, f"{self.key}.json")
        with open(path, "w") as f:
            json.dump({"adapter": out["adapter"], "base_model": out.get("base_model", self.base_model), "gpu": gpu,
                       "quantize": quantized, "instruction": instruction, "labels": ctx.spec.labels}, f)
        rps = float(out.get("batch_rows_per_s") or 0) or FALLBACK_REQ_PER_SEC
        # Serving cost: the batched throughput measured on the same GPU, at list price, assuming a warm replica.
        cost_1k = round(1000 * rate / rps, 6)
        extra = {"cost_estimated": True, "unknown_predictions": preds.count(UNKNOWN), "batch_rows_per_s": round(rps, 1),
                 "trainable_params": out.get("trainable_params"), "total_params": out.get("total_params"),
                 "final_loss": out.get("final_loss"), "epochs": out.get("epochs"), "rank": out.get("rank"), "gpu": gpu,
                 "quantized": quantized, "method": "QLoRA" if quantized else "LoRA"}
        cold_s = wall_s - float(out.get("total_seconds") or 0)  # container start, image pull, data transfer
        if cold_s > 0:
            extra["cold_start_ms"] = round(cold_s * 1000, 1)
        await ctx.emit("trainer", "log", f"{extra['method'].lower()} · {self.spec.short} trained in {out['train_seconds']:.0f}s · "
                                         f"final loss {out.get('final_loss')} · {rps:.0f} rows/s batched on the {gpu}")
        scoring = " · label scoring (no off-list answers)" if confidences is not None else ""
        where = "every linear layer" + (" except the routed experts" if out.get("moe") else "")
        return Result(
            predictions=preds, latencies_ms=out["latencies_ms"], cost_per_1k=cost_1k,
            train_cost=float(out["total_seconds"]) * rate, train_seconds=float(out["train_seconds"]),
            size_kb=await adapter_size_kb(out), artifact_path=path, confidences=confidences,
            notes=(f"{extra['method']} r={out.get('rank', 8)} on {where} of {self.spec.short}"
                   + (" (4-bit NF4 base)" if quantized else "") + f" · {len(ctx.train)} examples · "
                   f"{out.get('epochs')} epochs on a {gpu}{scoring} · latency measured warm in-container · "
                   f"serving cost est. from the measured {rps:.0f} rows/s at {gpu} list price"),
            extra=extra,
        )
