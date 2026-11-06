"""GPU fine-tuning + serving for Ramp Model Labs, on Modal.

    uv run modal deploy modal_app.py      # once per code change: builds the image, registers the functions
    # then MODAL_ENABLED=true in backend/.env and restart the API
    uv run modal run modal_app.py         # optional smoke test on a small synthetic task (a few cents)

train_and_eval LoRA-tunes a small open instruct model on the lab's training split and scores it on the
held-out split in the same container, so every number the lab shows was measured on the GPU that would
serve it. predict serves saved adapters. The recipe follows what the evidence supports for small models:
label-only targets (chain-of-thought hurts under ~10B parameters), low-rank adapters on every linear
layer, two or three epochs, greedy decoding.
"""
import os
import time

import modal

APP_NAME = "ramp-model-labs"
DEFAULT_BASE = "Qwen/Qwen3-1.7B"
GPU = "L4"
L4_DOLLARS_PER_SEC = 0.80 / 3600  # Modal list price; the backend prices measured GPU seconds at this rate
MAX_NEW_TOKENS = 16
HF_CACHE = "/models"

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "transformers>=4.56,<5", "peft>=0.17", "trl>=0.23", "datasets", "accelerate")
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false"})
)
adapters = modal.Volume.from_name("rml-adapters", create_if_missing=True)
weights = modal.Volume.from_name("rml-hf-cache", create_if_missing=True)  # base weights are downloaded once
VOLUMES = {"/adapters": adapters, HF_CACHE: weights}

_cache: dict = {}


def _system(instruction: str, labels: list[str]) -> str:
    return f"{instruction}\nLabels: {', '.join(labels)}. Answer with the label only."


def render_prompt(tok, instruction: str, labels: list[str], text: str) -> str:
    """The exact string the model sees at training and at serving time. Thinking is switched off for
    Qwen3 (templates without the switch ignore it): a classifier answers, it does not deliberate."""
    msgs = [{"role": "system", "content": _system(instruction, labels)}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _load(base_model: str, adapter_dir: str | None):
    key = (base_model, adapter_dir)
    if key in _cache:
        return _cache[key]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.eval()
    _cache[key] = (tok, model)
    return tok, model


def _generate(tok, model, instruction: str, labels: list[str], texts: list[str]) -> list[str]:
    import torch

    prompts = [render_prompt(tok, instruction, labels, t) for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


def _dir_kb(path: str) -> float:
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs) / 1024


@app.function(image=image, gpu=GPU, timeout=60 * 45, volumes=VOLUMES)
def train_and_eval(train_rows: list[dict], test_rows: list[dict], labels: list[str], instruction: str,
                   run_id: str, base_model: str = DEFAULT_BASE, epochs: int | None = None, rank: int = 8) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(base_model)
    eos = tok.eos_token or "<|im_end|>"
    # Prompt/completion pairs: the loss is taken on the label tokens only, and the prompt is the serving prompt.
    ds = Dataset.from_list([{"prompt": render_prompt(tok, instruction, labels, r["text"]),
                             "completion": f"{r['label']}{eos}"} for r in train_rows])
    n = len(train_rows)
    epochs = epochs or (3 if n < 800 else 2)
    big = any(tag in base_model for tag in ("4B", "7B", "8B"))
    out_dir = f"/adapters/{run_id}"
    cfg = SFTConfig(
        output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=8 if big else 16,
        gradient_accumulation_steps=2 if big else 1, learning_rate=2e-4, warmup_ratio=0.03,
        lr_scheduler_type="cosine", bf16=True, logging_steps=10, save_strategy="no", report_to=[],
        max_length=256, completion_only_loss=True, model_init_kwargs={"torch_dtype": torch.bfloat16},
    )
    # Rank 8 on every linear layer: the LoRA paper found more matrices at low rank beat one matrix at high rank.
    lora = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=base_model, train_dataset=ds, args=cfg, peft_config=lora)
    weights.commit()  # the base weights just landed in the cache volume; the next container skips the download
    trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    adapter_dir = f"{out_dir}/adapter"
    trainer.save_model(adapter_dir)
    adapters.commit()
    train_s = time.time() - t0
    del trainer
    torch.cuda.empty_cache()

    tok, model = _load(base_model, adapter_dir)
    texts = [r["text"] for r in test_rows]
    _generate(tok, model, instruction, labels, texts[:2])  # warm-up: CUDA graphs and allocator
    tb = time.perf_counter()
    preds: list[str] = []
    for i in range(0, len(texts), 32):
        preds += _generate(tok, model, instruction, labels, texts[i:i + 32])
    batch_rows_per_s = len(texts) / max(1e-6, time.perf_counter() - tb)
    lat = []
    for t in texts[:20]:
        s = time.perf_counter()
        _generate(tok, model, instruction, labels, [t])
        lat.append((time.perf_counter() - s) * 1000)
    return {
        "predictions": preds, "latencies_ms": lat, "batch_rows_per_s": round(batch_rows_per_s, 2),
        "train_seconds": round(train_s, 1), "total_seconds": round(time.time() - t0, 1),
        "adapter": adapter_dir, "adapter_kb": round(_dir_kb(adapter_dir), 1), "base_model": base_model,
        "epochs": epochs, "rank": rank, "trainable_params": trainable, "total_params": total,
        "final_loss": losses[-1] if losses else None, "gpu": GPU,
    }


@app.function(image=image, gpu=GPU, timeout=600, volumes=VOLUMES, scaledown_window=120)
def predict(adapter: str, instruction: str, labels: list[str], texts: list[str], base_model: str = DEFAULT_BASE) -> list[str]:
    adapters.reload()
    tok, model = _load(base_model, adapter)
    out: list[str] = []
    for i in range(0, len(texts), 32):
        out += _generate(tok, model, instruction, labels, texts[i:i + 32])
    return out


@app.local_entrypoint()
def smoke(base_model: str = DEFAULT_BASE, n: int = 300):
    """A real end-to-end run on a small synthetic task. Prints accuracy, timings and what the GPU time cost."""
    try:
        from app.datasets.templates import TEMPLATES, generate

        t = TEMPLATES["fraud_detection"]
        rows = generate(t, n + 100, seed=11, noise=False)
        labels, instruction = list(t.labels), f"{t.description} Inputs look like: {t.input_hint}"
    except Exception:  # run from outside the backend folder: a toy task is still a real GPU run
        rows = [{"text": f"card-not-present ${i * 7}.00 at NEWVENDOR{i} 03:1{i % 6} local; new device", "label": "fraud"}
                if i % 3 else {"text": f"chip $12.{i % 100:02d} at WHOLE FOODS 12:0{i % 6} local; recurring", "label": "legitimate"}
                for i in range(n + 100)]
        labels, instruction = ["fraud", "legitimate"], "Decide whether a card transaction looks fraudulent."
    train, test = rows[:n], rows[n:]
    t0 = time.time()
    out = train_and_eval.remote(train, test, labels, instruction, f"smoke-{int(t0)}", base_model=base_model)
    acc = sum(p.strip().lower() == r["label"].lower() for p, r in zip(out["predictions"], test)) / len(test)
    lat = sorted(out["latencies_ms"])
    print(f"\n{base_model}  LoRA r={out['rank']}  {out['trainable_params'] / 1e6:.1f}M trainable of {out['total_params'] / 1e9:.2f}B")
    print(f"accuracy on {len(test)} held-out rows: {acc:.1%}   final train loss {out['final_loss']}")
    print(f"train {out['train_seconds']}s · total in-container {out['total_seconds']}s · wall {time.time() - t0:.0f}s")
    print(f"single-request p50 {lat[len(lat) // 2]:.0f} ms · batched {out['batch_rows_per_s']} rows/s · adapter {out['adapter_kb'] / 1024:.1f} MB")
    print(f"GPU cost of this run ≈ ${out['total_seconds'] * L4_DOLLARS_PER_SEC:.3f} at the L4 list price")
