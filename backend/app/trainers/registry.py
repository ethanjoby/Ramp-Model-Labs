from .. import llm
from ..config import get_settings
from .base import Strategy
from .cascade import Cascade
from .classical import Committee, LinearSpecialist, NaiveBayes, SvmSpecialist, TinyNeural
from .embedding import EmbeddingSpecialist
from .encoder import EncoderFinetune
from .finetuned import LoraFinetune, roster_models, short_name
from .prompted import OptimizedFewShot, ZeroShot

LOCAL_LLM_SECONDS = 150  # a 0.6B decoder on a laptop GPU: two passes over 1,500 rows fit in this box


def local_llm(model: str) -> EncoderFinetune:
    """A small decoder LLM as a classifier: its top four blocks and a fresh head train on this machine, and only
    those weights are saved (a delta over the base). Bigger than the encoders, slower to serve, but it brings
    a language model's prior knowledge of what 'LUFTHANSA' or 'DATADOG' is to merchants it has never seen labeled."""
    short = short_name(model)
    return EncoderFinetune(model=model, key="finetuned_local_llm", tag=short.lower(), display=f"Fine-tuned {short} (local, top blocks)",
                           max_train_seconds=LOCAL_LLM_SECONDS, unfreeze_last=4, max_rows=1500, batch=16,
                           blurb=(f"{short}, a decoder language model, with a classification head; its top four blocks train on this "
                                  "machine and are saved as a delta over the base. Served on CPU."))


def strategies() -> list[Strategy]:
    s = get_settings()
    together = [ZeroShot(m, f"zero_shot_{llm.slug(m)}", f"{llm.pretty(m)} · zero-shot") for m in llm.together_models()]
    lora = [LoraFinetune(m, key=("lora_qwen" if i == 0 else f"lora_{llm.slug(m)}")) for i, m in enumerate(roster_models(s))]
    ref = llm.reference_model()
    return [
        NaiveBayes(), LinearSpecialist(), SvmSpecialist(), TinyNeural(), Committee(), Cascade(),
        EmbeddingSpecialist(),
        EncoderFinetune(),
        EncoderFinetune(model="answerdotai/ModernBERT-base", key="finetuned_modernbert", tag="modernbert",
                        display="Fine-tuned ModernBERT-base (150M, local)", max_train_seconds=300, tacp=True),
        *([local_llm(s.local_llm_model)] if s.local_llm_model else []),
        ZeroShot(s.fast_model, "zero_shot_haiku", f"{llm.pretty(s.fast_model)} · zero-shot"),
        OptimizedFewShot(),
        *together,
        *lora,
        ZeroShot(ref, "frontier_reference", f"{llm.pretty(ref)} · frontier reference", family="reference"),
    ]
