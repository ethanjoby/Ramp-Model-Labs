"""GPU fine-tuning + serving for Ramp Model Labs, on Modal.

    uv run modal deploy modal_app.py      # once per code change: builds the image, registers the Trainer class
    # then MODAL_ENABLED=true in backend/.env and restart the API
    uv run modal run modal_app.py         # optional smoke test on a small synthetic task (a few cents)
    uv run modal run modal_app.py --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --gpu A100-40GB

Trainer.train_and_eval LoRA-tunes an open language model on the lab's training split and scores it on the
held-out split in the same container, so every number the lab shows was measured on the GPU that would
serve it. Trainer.predict serves saved adapters. The GPU is chosen per call (`Trainer.with_options(gpu=...)`),
so a 1.7B model trains on an L4 while a 7B DeepSeek distill gets an A100 and a 16B Moonshot MoE an 80 GB one.
When the bf16 weights would not fit the GPU, the base is loaded in 4-bit NF4 and the adapters train on top
(QLoRA, Dettmers et al. 2023); the adapter is the same either way.

The recipe follows what the evidence supports for small models: label-only targets (chain-of-thought hurts
under ~10B parameters), low-rank adapters on every linear layer (experts excluded on MoE bases, where the
attention and shared projections carry the adaptation), two or three epochs, NEFTune embedding noise.

Inference scores labels instead of generating text: for each input, the log-likelihood of every allowed
label (plus the end-of-turn token) as the completion is computed in one batched forward pass, and the
best-scoring label is the answer. That means no off-list answers to count wrong, a confidence per row
(softmax over the label scores, the quantity the SFT loss actually optimised) that the review queue can
use, and one forward pass per (input, label) pair rather than autoregressive decoding.
"""
import math
import os
import re
import time

import modal

APP_NAME = "ramp-model-labs"
DEFAULT_BASE = "Qwen/Qwen3-1.7B"
GPU = "L4"  # the default; callers pick another per base with Trainer.with_options(gpu=...)
GPU_MEMORY_GB = {"T4": 16, "L4": 24, "A10G": 24, "L40S": 48, "A100-40GB": 40, "A100-80GB": 80, "H100": 80, "H200": 141, "B200": 192}
# Modal list prices per GPU-second (modal.com/pricing); the backend prices measured seconds at these rates.
GPU_DOLLARS_PER_SEC = {"T4": 0.000164, "L4": 0.000222, "A10G": 0.000306, "L40S": 0.000542, "A100-40GB": 0.000583,
                       "A100-80GB": 0.000694, "H100": 0.001097, "H200": 0.001261, "B200": 0.001736}
L4_DOLLARS_PER_SEC = GPU_DOLLARS_PER_SEC["L4"]
PAIR_BATCH = 32  # (input, label) sequences per forward pass; logits for 32 x ~160 tokens x 150k vocab fit an L4 in bf16
HF_CACHE = "/models"
TRUSTED_REMOTE_CODE = ("deepseek-ai/", "moonshotai/", "Qwen/")  # orgs whose repo code may run when transformers has no native class

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "transformers>=4.56,<5", "peft>=0.17", "trl>=0.23", "datasets", "accelerate",
                    "bitsandbytes>=0.46")
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false"})
)
adapters = modal.Volume.from_name("rml-adapters", create_if_missing=True)
weights = modal.Volume.from_name("rml-hf-cache", create_if_missing=True)  # base weights are downloaded once
VOLUMES = {"/adapters": adapters, HF_CACHE: weights}

_cache: dict = {}


def _system(instruction: str, labels: list[str]) -> str:
    return f"{instruction}\nLabels: {', '.join(labels)}. Answer with the label only."


def _eos(tok) -> str:
    return tok.eos_token or "<|im_end|>"


def render_prompt(tok, instruction: str, labels: list[str], text: str) -> str:
    """The exact string the model sees at training and at serving time. Thinking is switched off for
    Qwen3 (templates without the switch ignore it): a classifier answers, it does not deliberate."""
    msgs = [{"role": "system", "content": _system(instruction, labels)}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def params_b(base_model: str) -> float:
    """Billions of parameters read off the model name: 'Qwen3-1.7B' -> 1.7, 'Moonlight-16B-A3B' -> 16 (total, not active)."""
    found = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", base_model.split("/")[-1])]
    return max(found) if found else 2.0


def needs_quantization(base_model: str, gpu: str) -> bool:
    """bf16 weights plus optimizer state and activations need about twice the weight bytes; past half the card, go 4-bit."""
    return params_b(base_model) * 2 > 0.5 * GPU_MEMORY_GB.get(gpu, 24)


def _detect_gpu(name: str, memory_gb: float) -> str:
    """Map torch's device name ('NVIDIA A100-SXM4-40GB') onto a price-table key, by family then by memory."""
    compact = name.upper().replace(" ", "")
    family = [k for k in GPU_MEMORY_GB if k.split("-")[0] in compact] or list(GPU_MEMORY_GB)
    return min(family, key=lambda k: abs(GPU_MEMORY_GB[k] - memory_gb))


def _remote_code(base_model: str) -> bool:
    """Run the repo's own modeling code only for trusted orgs, and only when transformers has no native class."""
    if not base_model.startswith(TRUSTED_REMOTE_CODE):
        return False
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(base_model)
        return False
    except Exception:
        return True


def _is_moe(base_model: str) -> bool:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=_remote_code(base_model))
        return bool(getattr(cfg, "n_routed_experts", None) or getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None))
    except Exception:
        return "A3B" in base_model or "MoE" in base_model


def _quant_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)


def _load_base(base_model: str, quantize: bool):
    import torch
    from transformers import AutoModelForCausalLM

    kw = {"device_map": "cuda", "trust_remote_code": _remote_code(base_model)}
    if quantize:
        kw["quantization_config"] = _quant_config()
        kw["torch_dtype"] = torch.bfloat16
    else:
        kw["torch_dtype"] = torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(base_model, **kw)


def _tokenizer(base_model: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=_remote_code(base_model))
    tok.padding_side = "right"  # scoring reads positions by index; nothing is generated
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _load(base_model: str, adapter_dir: str | None, quantize: bool = False):
    key = (base_model, adapter_dir, quantize)
    if key in _cache:
        return _cache[key]
    tok = _tokenizer(base_model)
    model = _load_base(base_model, quantize)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        if not quantize:
            model = model.merge_and_unload()  # a 4-bit base cannot absorb the adapter; it is applied on the fly instead
    model.eval()
    _cache[key] = (tok, model)
    return tok, model


def _pair_ids(tok, prompt: str, completion: str) -> tuple[list[int], int]:
    """Token ids of prompt+completion and the index where the completion starts, tokenized the way TRL's
    prompt/completion SFT does it (prompt alone, then prompt+completion, completion = the suffix)."""
    p = tok(text=prompt)["input_ids"]
    full = tok(text=prompt + completion)["input_ids"]
    if full[: len(p)] != p:  # a merge across the boundary: fall back to concatenating the two tokenizations
        full = p + tok(text=completion, add_special_tokens=False)["input_ids"]
    return full, len(p)


def score_labels(tok, model, instruction: str, labels: list[str], texts: list[str]) -> tuple[list[str], list[float], list[list[float]]]:
    """For every input, log P(label + eos | prompt) for each allowed label, batched over (input, label) pairs.

    Returns (predicted labels, confidence of the prediction, full per-label probability rows). The score is
    the summed log-probability of the complete answer, end-of-turn token included: that is exactly what the
    SFT loss trained, and the eos makes "approve" and "approved" complete, comparable answers rather than
    one being a prefix of the other. No per-token length normalisation: a longer label is not a worse one.
    """
    import torch

    dev = next(model.parameters()).device
    eos = _eos(tok)
    prompts = [render_prompt(tok, instruction, labels, t) for t in texts]
    pairs: list[tuple[list[int], int]] = [_pair_ids(tok, p, f"{l}{eos}") for p in prompts for l in labels]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    totals: list[float] = []
    with torch.no_grad():
        for i in range(0, len(pairs), PAIR_BATCH):
            chunk = pairs[i:i + PAIR_BATCH]
            T = max(len(ids) for ids, _ in chunk)
            input_ids = torch.full((len(chunk), T), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), T), dtype=torch.long)
            rows, cols, tgts = [], [], []
            for b, (ids, start) in enumerate(chunk):
                input_ids[b, : len(ids)] = torch.tensor(ids)
                attn[b, : len(ids)] = 1
                for t in range(start, len(ids)):  # position t-1 predicts token t
                    rows.append(b)
                    cols.append(t - 1)
                    tgts.append(ids[t])
            rows_t = torch.tensor(rows, device=dev)
            logits = model(input_ids=input_ids.to(dev), attention_mask=attn.to(dev)).logits
            picked = logits[rows_t, torch.tensor(cols, device=dev)].float()  # [P, V]: only the completion positions
            lp = torch.log_softmax(picked, dim=-1).gather(-1, torch.tensor(tgts, device=dev).unsqueeze(-1)).squeeze(-1)
            per_seq = torch.zeros(len(chunk), dtype=torch.float32, device=dev).index_add_(0, rows_t, lp)
            totals += per_seq.tolist()
    scores = torch.tensor(totals).view(len(texts), len(labels))
    probs = torch.softmax(scores, dim=-1)
    best = probs.argmax(dim=-1).tolist()
    return [labels[j] for j in best], [round(float(probs[i, j]), 6) for i, j in enumerate(best)], [[round(float(v), 6) for v in row] for row in probs.tolist()]


def _dir_kb(path: str) -> float:
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs) / 1024


@app.cls(image=image, gpu=GPU, timeout=60 * 60, volumes=VOLUMES, scaledown_window=120)
class Trainer:
    """One class, any GPU: `Trainer.with_options(gpu="A100-80GB")().train_and_eval.remote(...)`."""

    @modal.method()
    def train_and_eval(self, train_rows: list[dict], test_rows: list[dict], labels: list[str], instruction: str,
                       run_id: str, base_model: str = DEFAULT_BASE, epochs: int | None = None, rank: int = 8,
                       quantize: bool | None = None, gpu: str | None = None) -> dict:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from trl import SFTConfig, SFTTrainer

        t0 = time.time()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_key = gpu if gpu in GPU_MEMORY_GB else _detect_gpu(gpu_name, torch.cuda.get_device_properties(0).total_memory / 2**30)
        if quantize is None:
            quantize = needs_quantization(base_model, gpu_key)
        tok = _tokenizer(base_model)
        eos = _eos(tok)
        # Prompt/completion pairs: the loss is taken on the label tokens only, and the prompt is the serving prompt.
        ds = Dataset.from_list([{"prompt": render_prompt(tok, instruction, labels, r["text"]),
                                 "completion": f"{r['label']}{eos}"} for r in train_rows])
        n = len(train_rows)
        epochs = epochs or (3 if n < 800 else 2)
        size = params_b(base_model)
        big = size >= 4
        moe = _is_moe(base_model)
        out_dir = f"/adapters/{run_id}"
        model = _load_base(base_model, quantize)
        weights.commit()  # the base weights just landed in the cache volume; the next container skips the download
        if quantize:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        cfg = SFTConfig(
            output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=8 if big else 16,
            gradient_accumulation_steps=2 if big else 1, learning_rate=2e-4 if size < 10 else 1e-4, warmup_ratio=0.03,
            lr_scheduler_type="cosine", bf16=True, logging_steps=10, save_strategy="no", report_to=[],
            max_length=256, completion_only_loss=True, gradient_checkpointing=big or quantize,
            # NEFTune (Jain et al. 2023): uniform noise on the input embeddings during training only. Free at
            # serving time and a consistent generalisation gain for instruction tuning of small models.
            neftune_noise_alpha=5.0,
        )
        # Rank 8 on every linear layer: the LoRA paper found more matrices at low rank beat one matrix at high rank.
        # On a mixture-of-experts base the routed experts are left alone: thousands of tiny adapters for no gain.
        lora = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM",
                          **({"exclude_modules": r".*experts.*"} if moe else {}))
        trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=cfg, peft_config=lora)
        trainer.train()
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        adapter_dir = f"{out_dir}/adapter"
        trainer.save_model(adapter_dir)
        adapters.commit()
        train_s = time.time() - t0
        del trainer, model
        torch.cuda.empty_cache()

        tok, model = _load(base_model, adapter_dir, quantize)
        texts = [r["text"] for r in test_rows]
        score_labels(tok, model, instruction, labels, texts[:2])  # warm-up: allocator and kernels
        tb = time.perf_counter()
        preds: list[str] = []
        confs: list[float] = []
        for i in range(0, len(texts), 32):
            p, c, _ = score_labels(tok, model, instruction, labels, texts[i:i + 32])
            preds += p
            confs += c
        batch_rows_per_s = len(texts) / max(1e-6, time.perf_counter() - tb)
        lat = []
        for t in texts[:20]:
            s = time.perf_counter()
            score_labels(tok, model, instruction, labels, [t])
            lat.append((time.perf_counter() - s) * 1000)
        return {
            "predictions": preds, "confidences": confs, "latencies_ms": lat, "batch_rows_per_s": round(batch_rows_per_s, 2),
            "train_seconds": round(train_s, 1), "total_seconds": round(time.time() - t0, 1),
            "adapter": adapter_dir, "adapter_kb": round(_dir_kb(adapter_dir), 1), "base_model": base_model,
            "epochs": epochs, "rank": rank, "trainable_params": trainable, "total_params": total,
            "final_loss": losses[-1] if losses else None, "gpu": gpu_key, "gpu_name": gpu_name, "quantized": bool(quantize),
            "moe": bool(moe), "inference": "label scoring",
        }

    @modal.method()
    def predict(self, adapter: str, instruction: str, labels: list[str], texts: list[str], base_model: str = DEFAULT_BASE,
                quantize: bool = False) -> dict:
        """{"labels": [...], "confidences": [...]} for `texts`, one entry per input, always an allowed label."""
        adapters.reload()
        tok, model = _load(base_model, adapter, quantize)
        out_labels: list[str] = []
        out_conf: list[float] = []
        for i in range(0, len(texts), 32):
            p, c, _ = score_labels(tok, model, instruction, labels, texts[i:i + 32])
            out_labels += p
            out_conf += c
        return {"labels": out_labels, "confidences": out_conf}


def _ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c <= hi)]
        if idx:
            total += len(idx) / len(conf) * abs(sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return total


@app.local_entrypoint()
def smoke(base_model: str = DEFAULT_BASE, n: int = 300, gpu: str = GPU, quantize: bool = False):
    """A real end-to-end run on a small synthetic task. Prints accuracy, calibration, timings and what the GPU time cost."""
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
    trainer = Trainer.with_options(gpu=gpu)()
    out = trainer.train_and_eval.remote(train, test, labels, instruction, f"smoke-{int(t0)}", base_model=base_model,
                                        quantize=quantize or None, gpu=gpu)
    correct = [p == r["label"] for p, r in zip(out["predictions"], test)]
    acc = sum(correct) / len(test)
    conf = out["confidences"]
    right = [c for c, ok in zip(conf, correct) if ok]
    wrong = [c for c, ok in zip(conf, correct) if not ok]
    mean = lambda xs: sum(xs) / len(xs) if xs else math.nan  # noqa: E731
    lat = sorted(out["latencies_ms"])
    print(f"\n{base_model}  {'QLoRA (4-bit base)' if out['quantized'] else 'LoRA'} r={out['rank']} on {out['gpu']}  "
          f"{out['trainable_params'] / 1e6:.1f}M trainable of {out['total_params'] / 1e9:.2f}B")
    print(f"accuracy on {len(test)} held-out rows: {acc:.1%}   final train loss {out['final_loss']}")
    print(f"calibration: mean confidence {mean(right):.3f} when right vs {mean(wrong):.3f} when wrong · ECE {_ece(conf, correct):.3f}")
    print(f"train {out['train_seconds']}s · total in-container {out['total_seconds']}s · wall {time.time() - t0:.0f}s")
    print(f"single-request p50 {lat[len(lat) // 2]:.0f} ms · batched {out['batch_rows_per_s']} rows/s · adapter {out['adapter_kb'] / 1024:.1f} MB")
    rate = GPU_DOLLARS_PER_SEC.get(out["gpu"], L4_DOLLARS_PER_SEC)
    print(f"GPU cost of this run ≈ ${out['total_seconds'] * rate:.3f} at the {out['gpu']} list price")
