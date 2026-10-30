"""Per-lab event bus. Events are persisted (so the UI can replay) and fanned out live over SSE."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from sqlmodel import Session, func, select

from .db import LabEvent, engine


class Bus:
    def __init__(self) -> None:
        self.subs: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self.seq: dict[str, int] = {}

    def subscribe(self, lab_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.subs[lab_id].add(q)
        return q

    def unsubscribe(self, lab_id: str, q: asyncio.Queue) -> None:
        self.subs[lab_id].discard(q)

    def _next_seq(self, s: Session, lab_id: str) -> int:
        if lab_id not in self.seq:
            self.seq[lab_id] = s.exec(select(func.max(LabEvent.seq)).where(LabEvent.lab_id == lab_id)).one() or 0
        self.seq[lab_id] += 1
        return self.seq[lab_id]

    async def emit(self, lab_id: str, agent: str, kind: str, message: str = "", data: dict[str, Any] | None = None):
        with Session(engine) as s:
            ev = LabEvent(lab_id=lab_id, seq=self._next_seq(s, lab_id), agent=agent, kind=kind,
                          message=message, data=data or {})
            s.add(ev)
            s.commit()
            s.refresh(ev)
            payload = serialize(ev)
        for q in list(self.subs.get(lab_id, ())):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass


def serialize(ev: LabEvent) -> dict[str, Any]:
    return {"seq": ev.seq, "agent": ev.agent, "kind": ev.kind, "message": ev.message, "data": ev.data,
            "ts": ev.created_at.isoformat()}


bus = Bus()
