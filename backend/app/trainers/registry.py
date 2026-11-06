from ..config import get_settings
from .base import Strategy
from .cascade import Cascade
from .classical import LinearSpecialist, NaiveBayes, SvmSpecialist, TinyNeural
from .encoder import EncoderFinetune
from .finetuned import LoraFinetune
from .prompted import OptimizedFewShot, ZeroShot


def _pretty(model_id: str) -> str:
    """claude-haiku-5-5 -> Claude Haiku 5.5, so the leaderboard names the model that actually ran."""
    parts = model_id.replace("claude-", "").split("-")
    name, ver = parts[0].capitalize(), ".".join(p for p in parts[1:] if p.isdigit())
    return f"Claude {name} {ver}".strip()


def strategies() -> list[Strategy]:
    s = get_settings()
    return [
        NaiveBayes(), LinearSpecialist(), SvmSpecialist(), TinyNeural(), Cascade(),
        EncoderFinetune(),
        EncoderFinetune(model="answerdotai/ModernBERT-base", key="finetuned_modernbert", tag="modernbert",
                        display="Fine-tuned ModernBERT-base (150M, local)", max_train_seconds=300, tacp=True),
        ZeroShot(s.fast_model, "zero_shot_haiku", f"{_pretty(s.fast_model)} · zero-shot"),
        OptimizedFewShot(),
        LoraFinetune(),
        *([LoraFinetune(s.modal_heavy_model, key="lora_heavy")] if s.modal_heavy_model else []),
        ZeroShot(s.agent_model, "frontier_reference", f"{_pretty(s.agent_model)} · frontier reference", family="reference"),
    ]
