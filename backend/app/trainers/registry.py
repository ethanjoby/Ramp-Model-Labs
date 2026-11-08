from .. import llm
from ..config import get_settings
from .base import Strategy
from .cascade import Cascade
from .classical import Committee, LinearSpecialist, NaiveBayes, SvmSpecialist, TinyNeural
from .embedding import EmbeddingSpecialist
from .encoder import EncoderFinetune
from .finetuned import LoraFinetune, roster, short_name
from .prompted import OptimizedFewShot, ZeroShot

LOCAL_LLM_SECONDS = 150  # a 0.6B decoder on a laptop GPU: two passes over 1,500 rows fit in this box


def local_llm(model: str, key: str = "finetuned_local_llm") -> EncoderFinetune:
    """A small decoder LLM as a classifier: its top four blocks and a fresh head train on this machine, and only
    those weights are saved (a delta over the base). Bigger than the encoders, slower to serve, but it brings
    a language model's prior knowledge of what 'LUFTHANSA' or 'DATADOG' is to merchants it has never seen labeled.
    A 1.5B base (DeepSeek-R1-Distill-Qwen-1.5B trains at ~0.8 steps/s on an M-series GPU) gets a longer box."""
    from .finetuned import params_b

    short = short_name(model)
    seconds = LOCAL_LLM_SECONDS if params_b(model) <= 1.0 else 2 * LOCAL_LLM_SECONDS
    return EncoderFinetune(model=model, key=key, tag=short.lower(), display=f"Fine-tuned {short} (local, top blocks)",
                           max_train_seconds=seconds, unfreeze_last=4, max_rows=1500, batch=16,
                           blurb=(f"{short}, a decoder language model, with a classification head; its top four blocks train on this "
                                  "machine and are saved as a delta over the base. Served on CPU."))


def local_llms(setting: str) -> list[EncoderFinetune]:
    """LOCAL_LLM_MODEL is one model or a comma-separated list; the first keeps the historical key."""
    models = [m.strip() for m in (setting or "").split(",") if m.strip()]
    return [local_llm(m, "finetuned_local_llm" if i == 0 else f"finetuned_local_{llm.slug(m)}") for i, m in enumerate(models)]


def strategies() -> list[Strategy]:
    s = get_settings()
    together = [ZeroShot(m, f"zero_shot_{llm.slug(m)}", f"{llm.pretty(m)} · zero-shot") for m in llm.together_models()]
    lora = [LoraFinetune(spec, key=("lora_qwen" if i == 0 else f"lora_{llm.slug(spec.base)}")) for i, spec in enumerate(roster(s))]
    ref = llm.reference_model()
    return [
        NaiveBayes(), LinearSpecialist(), SvmSpecialist(), TinyNeural(), Committee(), Cascade(),
        EmbeddingSpecialist(),
        EncoderFinetune(),
        EncoderFinetune(model="answerdotai/ModernBERT-base", key="finetuned_modernbert", tag="modernbert",
                        display="Fine-tuned ModernBERT-base (150M, local)", max_train_seconds=180, tacp=True, max_rows=1200, batch=16),
        *local_llms(s.local_llm_model),
        ZeroShot(s.fast_model, "zero_shot_haiku", f"{llm.pretty(s.fast_model)} · zero-shot"),
        OptimizedFewShot(),
        *together,
        *lora,
        ZeroShot(ref, "frontier_reference", f"{llm.pretty(ref)} · frontier reference", family="reference"),
    ]
