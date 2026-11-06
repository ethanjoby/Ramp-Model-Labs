"""LoRA fine-tune of a small open language model on Modal GPUs (see modal_app.py for the recipe)."""
from __future__ import annotations

import concurrent.futures
import json
import os
import time

from ..config import get_settings
from .base import LabContext, Result, Strategy
from .prompted import UNKNOWN, _normalize, default_instruction

L4_DOLLARS_PER_SEC = 0.80 / 3600  # Modal L4 list price; training cost is measured GPU seconds at this rate
FALLBACK_REQ_PER_SEC = 40         # only if the job somehow reports no measured throughput
VOLUME = "rml-adapters"           # the volume modal_app.py writes adapters to
VOLUME_MOUNT = "/adapters/"
_deployed: dict[str, tuple[float, bool, str]] = {}


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]  # Qwen/Qwen3-1.7B -> Qwen3-1.7B


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

            modal.Function.from_name(app_name, "train_and_eval").hydrate()
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

    def __init__(self, base_model: str | None = None, key: str = "lora_qwen"):
        self.base_model = base_model or get_settings().modal_base_model
        self.key = key
        self.display_name = f"Fine-tuned {short_name(self.base_model)} (LoRA, GPU)"
        self.blurb = (f"Low-rank adapters on every linear layer of {short_name(self.base_model)}, trained on a Modal L4 "
                      "with label-only targets. A real language model you own, served from a GPU.")

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
        import modal

        s = get_settings()
        fn = modal.Function.from_name(s.modal_app_name, "train_and_eval")
        instruction = default_instruction(ctx.spec)
        await ctx.emit("trainer", "log", f"lora · {short_name(self.base_model)} · sending {len(ctx.train)} rows to a Modal L4")
        t0 = time.perf_counter()
        out = await fn.remote.aio(ctx.train, ctx.test, ctx.spec.labels, instruction, f"{ctx.lab_id}-{self.key}",
                                  base_model=self.base_model)
        wall_s = time.perf_counter() - t0
        preds = [_normalize(p, ctx.spec.labels) or UNKNOWN for p in out["predictions"]]
        path = os.path.join(ctx.artifact_dir, f"{self.key}.json")
        with open(path, "w") as f:
            json.dump({"adapter": out["adapter"], "base_model": out.get("base_model", self.base_model),
                       "instruction": instruction, "labels": ctx.spec.labels}, f)
        rps = float(out.get("batch_rows_per_s") or 0) or FALLBACK_REQ_PER_SEC
        # Serving cost: the batched throughput measured on the same L4, at list price, assuming a warm replica.
        cost_1k = round(1000 * L4_DOLLARS_PER_SEC / rps, 6)
        extra = {"cost_estimated": True, "unknown_predictions": preds.count(UNKNOWN), "batch_rows_per_s": round(rps, 1),
                 "trainable_params": out.get("trainable_params"), "total_params": out.get("total_params"),
                 "final_loss": out.get("final_loss"), "epochs": out.get("epochs"), "rank": out.get("rank"), "gpu": out.get("gpu")}
        cold_s = wall_s - float(out.get("total_seconds") or 0)  # container start, image pull, data transfer
        if cold_s > 0:
            extra["cold_start_ms"] = round(cold_s * 1000, 1)
        await ctx.emit("trainer", "log", f"lora · trained in {out['train_seconds']:.0f}s · final loss {out.get('final_loss')} · "
                                         f"{rps:.0f} rows/s batched on the L4")
        return Result(
            predictions=preds, latencies_ms=out["latencies_ms"], cost_per_1k=cost_1k,
            train_cost=float(out["total_seconds"]) * L4_DOLLARS_PER_SEC, train_seconds=float(out["train_seconds"]),
            size_kb=await adapter_size_kb(out), artifact_path=path,
            notes=(f"LoRA r={out.get('rank', 8)} on every linear layer of {short_name(self.base_model)} · {len(ctx.train)} examples · "
                   f"{out.get('epochs')} epochs on an L4 · latency measured warm in-container · "
                   f"serving cost est. from the measured {rps:.0f} rows/s at L4 list price"),
            extra=extra,
        )
