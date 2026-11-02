"""LoRA fine-tune of a small open model on Modal GPUs."""
from __future__ import annotations

import json
import os
import time

from ..config import get_settings
from .base import LabContext, Result, Strategy
from .prompted import UNKNOWN, _normalize, default_instruction

L4_DOLLARS_PER_SEC = 0.000222  # Modal L4 list price; training cost is measured GPU seconds at this rate
# Serving is not measured by the lab: one warm L4 replica at an assumed sustained rate. Every number
# derived from it is labelled an estimate.
SERVING_REQ_PER_SEC = 40
VOLUME = "rml-adapters"  # the volume modal_app.py writes adapters to
VOLUME_MOUNT = "/adapters/"


async def adapter_size_kb(out: dict, volume: str = VOLUME) -> float | None:
    """Size of the saved adapter from a directory listing, or None rather than a guess.

    The job's reply is used when it carries a listing (adapter_files as {name: bytes} or [{size}], or
    adapter_bytes); otherwise the volume is listed. Anything that fails leaves the size unknown.
    """
    files = out.get("adapter_files")
    if isinstance(files, dict) and files:
        return round(sum(float(v) for v in files.values()) / 1024, 1)
    if isinstance(files, list) and files:
        return round(sum(float(f.get("size") or 0) for f in files if isinstance(f, dict)) / 1024, 1)
    if isinstance(out.get("adapter_bytes"), (int, float)):
        return round(out["adapter_bytes"] / 1024, 1)
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
    key = "lora_qwen"
    display_name = "Fine-tuned Qwen2.5-0.5B (LoRA)"
    family = "finetuned"
    blurb = "Parameter-efficient fine-tune of an open 0.5B model on Modal L4 GPUs. You own the weights."

    def available(self):
        s = get_settings()
        if not s.modal_enabled:
            return False, "Run `modal deploy modal_app.py` and set MODAL_ENABLED=true"
        try:
            import modal  # noqa: F401
        except ImportError:
            return False, "pip install modal"
        return True, ""

    async def run(self, ctx: LabContext) -> Result:
        import modal

        s = get_settings()
        fn = modal.Function.from_name(s.modal_app_name, "train_and_eval")
        instruction = default_instruction(ctx.spec)
        await ctx.emit("trainer", "log", "lora · dispatching to Modal L4 GPU")
        t0 = time.perf_counter()
        out = await fn.remote.aio(ctx.train, ctx.test, ctx.spec.labels, instruction, ctx.lab_id)
        wall_s = time.perf_counter() - t0
        preds = [_normalize(p, ctx.spec.labels) or UNKNOWN for p in out["predictions"]]
        path = os.path.join(ctx.artifact_dir, f"{self.key}.json")
        with open(path, "w") as f:
            json.dump({"adapter": out["adapter"], "instruction": instruction, "labels": ctx.spec.labels}, f)
        extra = {"cost_estimated": True, "unknown_predictions": preds.count(UNKNOWN)}
        # Wall time the job's own clock did not see: container start, image pull, data transfer.
        cold_s = wall_s - float(out.get("total_seconds") or 0)
        if out.get("total_seconds") and cold_s > 0:
            extra["cold_start_ms"] = round(cold_s * 1000, 1)
        return Result(
            predictions=preds, latencies_ms=out["latencies_ms"],
            cost_per_1k=round(1000 * L4_DOLLARS_PER_SEC / SERVING_REQ_PER_SEC, 5),
            train_cost=out["total_seconds"] * L4_DOLLARS_PER_SEC, train_seconds=out["train_seconds"],
            size_kb=await adapter_size_kb(out), artifact_path=path,
            notes=(f"LoRA r=16 · {len(ctx.train)} examples · L4 · latency measured warm in-container · "
                   f"serving cost est. from L4 list price at {SERVING_REQ_PER_SEC} req/s"),
            extra=extra,
        )
