from ..config import get_settings
from .base import Strategy
from .cascade import Cascade
from .classical import LinearSpecialist, NaiveBayes, SvmSpecialist, TinyNeural
from .finetuned import LoraFinetune
from .prompted import OptimizedFewShot, ZeroShot


def strategies() -> list[Strategy]:
    s = get_settings()
    return [
        NaiveBayes(), LinearSpecialist(), SvmSpecialist(), TinyNeural(), Cascade(),
        ZeroShot(s.fast_model, "zero_shot_haiku", "Claude Haiku 4.5 · zero-shot"),
        OptimizedFewShot(),
        LoraFinetune(),
        ZeroShot(s.agent_model, "frontier_reference", "Claude Sonnet 5.5 · frontier reference", family="reference"),
    ]
