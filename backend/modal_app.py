"""GPU fine-tuning + serving for Ramp Model Labs.

    pip install modal && modal token new
    modal deploy modal_app.py
    # then set MODAL_ENABLED=true in backend/.env

train_and_eval LoRA-tunes a small instruct model on the lab's training split, then
scores it on the held-out split in the same container. predict serves deployed adapters.
"""
import time

import modal

APP_NAME = "ramp-model-labs"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GPU = "L4"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch==2.8.0", "transformers>=4.56", "peft>=0.17", "trl>=0.23", "datasets", "accelerate"
)
vol = modal.Volume.from_name("rml-adapters", create_if_missing=True)

_cache: dict = {}


def _prompt(instruction: str, labels: list[str], text: str) -> list[dict]:
    return [
        {"role": "system", "content": f"{instruction}\nLabels: {', '.join(labels)}. Answer with the label only."},
        {"role": "user", "content": text},
    ]


def _load(adapter_dir: str):
    if adapter_dir in _cache:
        return _cache[adapter_dir]
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload().eval()
    _cache[adapter_dir] = (tok, model)
    return tok, model


def _generate(tok, model, instruction, labels, texts):
    import torch

    tok.padding_side = "left"
    prompts = [tok.apply_chat_template(_prompt(instruction, labels, t), tokenize=False, add_generation_prompt=True)
               for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


@app.function(image=image, gpu=GPU, timeout=60 * 45, volumes={"/adapters": vol})
def train_and_eval(train_rows: list[dict], test_rows: list[dict], labels: list[str], instruction: str,
                   run_id: str, epochs: int = 2) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    ds = Dataset.from_list([
        {"messages": _prompt(instruction, labels, r["text"]) + [{"role": "assistant", "content": r["label"]}]}
        for r in train_rows
    ])
    out_dir = f"/adapters/{run_id}"
    cfg = SFTConfig(output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=16,
                    learning_rate=2e-4, bf16=True, logging_steps=20, save_strategy="no", report_to=[],
                    assistant_only_loss=False, max_length=384)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=BASE_MODEL, train_dataset=ds, args=cfg, peft_config=lora)
    trainer.train()
    trainer.save_model(f"{out_dir}/adapter")
    vol.commit()
    train_s = time.time() - t0
    del trainer
    torch.cuda.empty_cache()

    tok, model = _load(f"{out_dir}/adapter")
    texts = [r["text"] for r in test_rows]
    preds = []
    for i in range(0, len(texts), 32):
        preds += _generate(tok, model, instruction, labels, texts[i:i + 32])
    lat = []
    for t in texts[:20]:
        s = time.perf_counter()
        _generate(tok, model, instruction, labels, [t])
        lat.append((time.perf_counter() - s) * 1000)
    return {"predictions": preds, "latencies_ms": lat, "train_seconds": train_s,
            "total_seconds": time.time() - t0, "adapter": f"{out_dir}/adapter"}


@app.function(image=image, gpu=GPU, timeout=600, volumes={"/adapters": vol}, scaledown_window=120)
def predict(adapter: str, instruction: str, labels: list[str], texts: list[str]) -> list[str]:
    vol.reload()
    tok, model = _load(adapter)
    return _generate(tok, model, instruction, labels, texts)
