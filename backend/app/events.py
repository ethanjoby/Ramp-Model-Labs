"""Per-channel event bus (labs and workflows share it). Events are persisted so the UI can replay, and fanned out live."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from sqlmodel import Session, func, select

from .db import LabEvent, engine

TERMINAL_STAGES = {"done", "failed"}


class Bus:
    def __init__(self) -> None:
        self.subs: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self.seq: dict[str, int] = {}

    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.subs[channel].add(q)
        return q

    def unsubscribe(self, channel: str, q: asyncio.Queue) -> None:
        subs = self.subs.get(channel)
        if subs is None:
            return
        subs.discard(q)
        if not subs:
            del self.subs[channel]  # a channel nobody listens to must not live forever

    def _next_seq(self, s: Session, channel: str) -> int:
        if channel not in self.seq:
            self.seq[channel] = s.exec(select(func.max(LabEvent.seq)).where(LabEvent.lab_id == channel)).one() or 0
        self.seq[channel] += 1
        return self.seq[channel]

    async def emit(self, channel: str, agent: str, kind: str, message: str = "", data: dict[str, Any] | None = None):
        with Session(engine) as s:
            ev = LabEvent(lab_id=channel, seq=self._next_seq(s, channel), agent=agent, kind=kind,
                          message=message, data=data or {})
            s.add(ev)
            s.commit()
            s.refresh(ev)
            payload = serialize(ev)
        for q in list(self.subs.get(channel, ())):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # A slow reader loses its oldest buffered event, never the terminal one it is waiting for.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                q.put_nowait(payload)
        if kind == "error" or (data or {}).get("stage") in TERMINAL_STAGES:
            self.seq.pop(channel, None)  # the next emit (a retrain never reuses the id) re-reads the max from the table


def is_terminal(ev: dict[str, Any]) -> bool:
    return ev.get("kind") == "error" or (ev.get("data") or {}).get("stage") in TERMINAL_STAGES


def serialize(ev: LabEvent) -> dict[str, Any]:
    return {"seq": ev.seq, "agent": ev.agent, "kind": ev.kind, "message": ev.message, "data": ev.data,
            "ts": ev.created_at.isoformat()}


bus = Bus()
