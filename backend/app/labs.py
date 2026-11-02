"""One place that starts a lab, so every entry point (API, copilot, retrain, workflows) shares the same limits."""
from __future__ import annotations

from sqlmodel import Session

from .agents import orchestrator
from .db import Lab, User
from .security import limiter

LABS_PER_HOUR = 12


def create_lab(s: Session, user: User, prompt: str, seeds: list[dict] | None = None) -> Lab:
    """Raises HTTPException(429) when the user is over the hourly lab limit."""
    limiter.check(f"lab:{user.id}", LABS_PER_HOUR, 3600)
    lab = Lab(user_id=user.id, prompt=prompt.strip()[:4000], seed_examples=seeds or [])
    s.add(lab)
    s.commit()
    s.refresh(lab)
    orchestrator.start(lab.id)
    return lab
