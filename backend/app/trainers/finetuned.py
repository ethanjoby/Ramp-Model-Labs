"""LoRA fine-tune of a small open model on Modal GPUs."""
from __future__ import annotations

import json
import os

from ..config import get_settings
from .base import LabContext, Result, Strategy
from .prompted import _normalize, default_instruction

L4_DOLLARS_PER_SEC = 0.000222
# Warm L4 replica serving batched traffic: ~40 req/s sustained at 0.5B params.
SERVING_REQ_PER_SEC = 40


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
        out = await fn.remote.aio(ctx.train, ctx.test, ctx.spec.labels, instruction, ctx.lab_id)
        preds = [_normalize(p, ctx.spec.labels) for p in out["predictions"]]
        path = os.path.join(ctx.artifact_dir, f"{self.key}.json")
        with open(path, "w") as f:
            json.dump({"adapter": out["adapter"], "instruction": instruction, "labels": ctx.spec.labels}, f)
        return Result(
            predictions=preds, latencies_ms=out["latencies_ms"],
            cost_per_1k=round(1000 * L4_DOLLARS_PER_SEC / SERVING_REQ_PER_SEC, 5),
            train_cost=out["total_seconds"] * L4_DOLLARS_PER_SEC, train_seconds=out["train_seconds"],
            size_kb=35_000, artifact_path=path, notes=f"LoRA r=16 · {len(ctx.train)} examples · L4",
        )
