from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field, field_validator
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from .. import llm
from ..agents import orchestrator
from ..agents.chat import Copilot
from ..agents.data import parse_examples
from ..config import get_settings
from ..datasets.templates import TEMPLATES, generate
from ..db import ApiKey, Candidate, ChatMessage, Deployment, Feedback, Lab, LabEvent, User, get_session
from ..events import bus, serialize
from ..firebase import verify_id_token
from ..inference import predict
from ..security import SESSION_COOKIE, api_key_user, client_ip, create_session_token, current_user, limiter, new_api_key

api = APIRouter(prefix="/api")
public = APIRouter(prefix="/v1")


# ------------------------------------------------------------------ auth

class FirebaseIn(BaseModel):
    id_token: str = Field(min_length=20, max_length=4096)


def _user_out(u: User) -> dict:
    return {"id": u.id, "email": u.email, "name": u.name, "avatar_url": u.avatar_url, "provider": u.provider}


def _set_session(resp: Response, user: User) -> None:
    s = get_settings()
    resp.set_cookie(SESSION_COOKIE, create_session_token(user), httponly=True, secure=s.cookie_secure,
                    samesite="lax", max_age=s.session_ttl_hours * 3600, path="/")


@api.post("/auth/firebase")
async def firebase_login(body: FirebaseIn, request: Request, response: Response, s: Session = Depends(get_session)):
    """Google sign-in: the browser hands over a Firebase ID token, we verify it and issue our own session."""
    limiter.check(f"fb:{client_ip(request)}", 30, 300)
    ident = await verify_id_token(body.id_token)
    if ident.provider != "google.com":
        raise HTTPException(403, "Only Google sign-in is enabled.")
    if not ident.email or not ident.email_verified:
        raise HTTPException(403, "Your Google account needs a verified email address.")
    user = (s.exec(select(User).where(User.firebase_uid == ident.uid)).first()
            or s.exec(select(User).where(User.email == ident.email)).first())
    if not user:
        user = User(email=ident.email, name=ident.name or ident.email.split("@")[0])
    user.firebase_uid, user.provider, user.avatar_url = ident.uid, "google", ident.picture
    if ident.name:
        user.name = ident.name
    s.add(user); s.commit(); s.refresh(user)
    _set_session(response, user)
    return _user_out(user)


@api.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@api.post("/auth/logout-all")
def logout_all(response: Response, user: User = Depends(current_user), s: Session = Depends(get_session)):
    user.token_version += 1
    s.add(user); s.commit()
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@api.get("/auth/me")
def me(user: User = Depends(current_user)):
    return _user_out(user)


# ------------------------------------------------------------------ platform info

@api.get("/capabilities")
def capabilities(user: User = Depends(current_user)):
    from ..trainers.registry import strategies

    return {
        "llm": llm.available(), "modal": get_settings().modal_enabled,
        "strategies": [{"key": st.key, "name": st.display_name, "family": st.family, "blurb": st.blurb,
                        "available": st.available()[0], "reason": st.available()[1]} for st in strategies()],
        "templates": [{"key": t.key, "name": t.name, "description": t.description, "labels": t.labels,
                       "samples": generate(t, 3, seed=3)} for t in TEMPLATES.values()],
    }


# ------------------------------------------------------------------ labs

class LabIn(BaseModel):
    prompt: str = Field(min_length=8, max_length=4000)
    examples: str | None = Field(default=None, max_length=5_000_000)


def _own_lab(lab_id: str, user: User, s: Session) -> Lab:
    lab = s.get(Lab, lab_id)
    if not lab or lab.user_id != user.id:
        raise HTTPException(404, "Lab not found")
    return lab


def _lab_out(lab: Lab, s: Session, full: bool = False) -> dict[str, Any]:
    out = {"id": lab.id, "name": lab.name, "prompt": lab.prompt, "status": lab.status, "error": lab.error,
           "created_at": lab.created_at.isoformat(), "finished_at": lab.finished_at.isoformat() if lab.finished_at else None,
           "spec": lab.spec, "summary": {k: (lab.result or {}).get(k) for k in ("winner_name", "savings", "deployment")},
           "winner_metrics": {k: ((lab.result or {}).get("winner_metrics") or {}).get(k)
                              for k in ("accuracy", "macro_f1", "cost_per_1k", "latency_p50_ms")}}
    if full:
        cands = s.exec(select(Candidate).where(Candidate.lab_id == lab.id)).all()
        out.update(result=lab.result, dataset=lab.dataset_stats, seed_count=len(lab.seed_examples or []),
                   candidates=[{"id": c.id, "key": c.strategy, "name": c.display_name, "family": c.family,
                                "status": c.status, "metrics": c.metrics, "is_winner": c.is_winner} for c in cands])
    return out


@api.post("/labs")
async def create_lab(body: LabIn, request: Request, user: User = Depends(current_user), s: Session = Depends(get_session)):
    limiter.check(f"lab:{user.id}", 12, 3600)
    seeds = parse_examples(body.examples) if body.examples else []
    lab = Lab(user_id=user.id, prompt=body.prompt.strip(), seed_examples=seeds)
    s.add(lab); s.commit(); s.refresh(lab)
    orchestrator.start(lab.id)
    return _lab_out(lab, s)


@api.post("/labs/parse-examples")
async def parse_upload(file: UploadFile = File(...), user: User = Depends(current_user)):
    raw = await file.read(get_settings().max_upload_bytes + 1)
    if len(raw) > get_settings().max_upload_bytes:
        raise HTTPException(413, "File too large (5 MB max).")
    text = raw.decode("utf-8", errors="replace")
    rows = parse_examples(text)
    if not rows:
        raise HTTPException(422, "Couldn't find a text column. Use CSV with text,label columns, JSONL, or 'text => label' lines.")
    labels = sorted({r["label"] for r in rows if r.get("label")})
    return {"raw": text, "count": len(rows), "labeled": sum(1 for r in rows if r.get("label")), "labels": labels[:50],
            "preview": rows[:5]}


@api.get("/labs")
def list_labs(user: User = Depends(current_user), s: Session = Depends(get_session)):
    labs = s.exec(select(Lab).where(Lab.user_id == user.id).order_by(Lab.created_at.desc()).limit(100)).all()
    return [_lab_out(l, s) for l in labs]


@api.get("/labs/{lab_id}")
def get_lab(lab_id: str, user: User = Depends(current_user), s: Session = Depends(get_session)):
    return _lab_out(_own_lab(lab_id, user, s), s, full=True)


@api.delete("/labs/{lab_id}")
def delete_lab(lab_id: str, user: User = Depends(current_user), s: Session = Depends(get_session)):
    lab = _own_lab(lab_id, user, s)
    if lab.status in ("queued", "running"):
        raise HTTPException(409, "Wait for the lab to finish first.")
    for model in (LabEvent, Candidate):
        for row in s.exec(select(model).where(model.lab_id == lab_id)).all():
            s.delete(row)
    for d in s.exec(select(Deployment).where(Deployment.lab_id == lab_id)).all():
        for f in s.exec(select(Feedback).where(Feedback.deployment_id == d.id)).all():
            s.delete(f)
        s.delete(d)
    s.delete(lab); s.commit()
    return {"ok": True}


@api.get("/labs/{lab_id}/events")
async def lab_events(lab_id: str, request: Request, after: int = 0, user: User = Depends(current_user),
                     s: Session = Depends(get_session)):
    _own_lab(lab_id, user, s)
    q = bus.subscribe(lab_id)
    past = [serialize(e) for e in s.exec(select(LabEvent).where(LabEvent.lab_id == lab_id, LabEvent.seq > after)
                                         .order_by(LabEvent.seq)).all()]
    s.close()

    async def gen():
        try:
            last = after
            for ev in past:
                last = ev["seq"]
                yield {"event": "lab", "id": str(ev["seq"]), "data": json.dumps(ev)}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                if ev["seq"] <= last:
                    continue
                last = ev["seq"]
                yield {"event": "lab", "id": str(ev["seq"]), "data": json.dumps(ev)}
        finally:
            bus.unsubscribe(lab_id, q)

    return EventSourceResponse(gen(), headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache, no-transform"})


# ------------------------------------------------------------------ deployments

def _own_dep(slug: str, user: User, s: Session) -> Deployment:
    d = s.exec(select(Deployment).where(Deployment.slug == slug)).first()
    if not d or d.user_id != user.id:
        raise HTTPException(404, "Model not found")
    return d


def _dep_out(d: Deployment, s: Session) -> dict:
    c = s.get(Candidate, d.candidate_id)
    lab = s.get(Lab, d.lab_id)
    fb = len(s.exec(select(Feedback).where(Feedback.deployment_id == d.id)).all())
    return {"id": d.id, "slug": d.slug, "name": d.name, "status": d.status, "lab_id": d.lab_id,
            "created_at": d.created_at.isoformat(), "requests": d.requests,
            "avg_latency_ms": round(d.total_latency_ms / d.requests, 2) if d.requests else None,
            "model": c.display_name if c else None, "family": c.family if c else None,
            "metrics": {k: (c.metrics or {}).get(k) for k in ("accuracy", "macro_f1", "cost_per_1k", "latency_p50_ms", "size_kb")} if c else {},
            "labels": (lab.spec or {}).get("labels", []) if lab else [],
            "savings": (lab.result or {}).get("savings") if lab else None, "feedback_count": fb}


@api.get("/deployments")
def list_deployments(user: User = Depends(current_user), s: Session = Depends(get_session)):
    deps = s.exec(select(Deployment).where(Deployment.user_id == user.id).order_by(Deployment.created_at.desc())).all()
    return [_dep_out(d, s) for d in deps]


@api.get("/deployments/{slug}")
def get_deployment(slug: str, user: User = Depends(current_user), s: Session = Depends(get_session)):
    return _dep_out(_own_dep(slug, user, s), s)


class FeedbackIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    predicted: str = Field(max_length=200)
    correct: str = Field(max_length=200)


@api.post("/deployments/{slug}/feedback")
def add_feedback(slug: str, body: FeedbackIn, user: User = Depends(current_user), s: Session = Depends(get_session)):
    d = _own_dep(slug, user, s)
    labels = (s.get(Lab, d.lab_id).spec or {}).get("labels", [])
    if body.correct not in labels:
        raise HTTPException(422, "Correct label must be one of the model's labels.")
    s.add(Feedback(deployment_id=d.id, text=body.text, predicted=body.predicted, correct=body.correct)); s.commit()
    return {"ok": True}


@api.post("/deployments/{slug}/retrain")
async def retrain(slug: str, user: User = Depends(current_user), s: Session = Depends(get_session)):
    """Close the loop: corrections become real labeled data and the whole lab re-runs."""
    limiter.check(f"lab:{user.id}", 12, 3600)
    d = _own_dep(slug, user, s)
    old = s.get(Lab, d.lab_id)
    fb = s.exec(select(Feedback).where(Feedback.deployment_id == d.id)).all()
    seeds = list(old.seed_examples or []) + [{"text": f.text, "label": f.correct} for f in fb]
    lab = Lab(user_id=user.id, prompt=old.prompt, seed_examples=seeds)
    s.add(lab); s.commit(); s.refresh(lab)
    orchestrator.start(lab.id)
    return {"lab_id": lab.id, "added_examples": len(fb)}


class PredictIn(BaseModel):
    inputs: list[str] = Field(min_length=1, max_length=100)

    @field_validator("inputs")
    @classmethod
    def lens(cls, v: list[str]) -> list[str]:
        if any(len(x) > 4000 for x in v):
            raise ValueError("Each input must be under 4000 characters.")
        return v


async def _predict(slug: str, body: PredictIn, user: User, s: Session):
    limiter.check(f"predict:{user.id}", 600, 60)
    d = s.exec(select(Deployment).where(Deployment.slug == slug)).first()
    if not d or d.user_id != user.id or d.status != "live":
        raise HTTPException(404, "Model not found")
    c = s.get(Candidate, d.candidate_id)
    preds, ms = await predict(c, body.inputs)
    d.requests += len(body.inputs); d.total_latency_ms += ms
    s.add(d); s.commit()
    return {"model": d.slug, "predictions": preds, "latency_ms": round(ms, 3)}


@api.post("/deployments/{slug}/predict")
async def playground_predict(slug: str, body: PredictIn, user: User = Depends(current_user),
                             s: Session = Depends(get_session)):
    return await _predict(slug, body, user, s)


@public.post("/models/{slug}/predict")
async def public_predict(slug: str, body: PredictIn, user: User = Depends(api_key_user),
                         s: Session = Depends(get_session)):
    return await _predict(slug, body, user, s)


# ------------------------------------------------------------------ API keys

class KeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)


@api.get("/keys")
def list_keys(user: User = Depends(current_user), s: Session = Depends(get_session)):
    keys = s.exec(select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.revoked == False)).all()  # noqa: E712
    return [{"id": k.id, "name": k.name, "prefix": k.prefix, "created_at": k.created_at.isoformat(),
             "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None} for k in keys]


@api.post("/keys")
def create_key(body: KeyIn, user: User = Depends(current_user), s: Session = Depends(get_session)):
    raw, prefix, digest = new_api_key()
    k = ApiKey(user_id=user.id, name=body.name, prefix=prefix, key_hash=digest)
    s.add(k); s.commit()
    return {"id": k.id, "name": k.name, "prefix": prefix, "key": raw}


@api.delete("/keys/{key_id}")
def revoke_key(key_id: str, user: User = Depends(current_user), s: Session = Depends(get_session)):
    k = s.get(ApiKey, key_id)
    if not k or k.user_id != user.id:
        raise HTTPException(404, "Key not found")
    k.revoked = True
    s.add(k); s.commit()
    return {"ok": True}


# ------------------------------------------------------------------ chat

class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    attachment: str | None = Field(default=None, max_length=5_000_000)


@api.get("/chat")
def chat_history(user: User = Depends(current_user), s: Session = Depends(get_session)):
    msgs = s.exec(select(ChatMessage).where(ChatMessage.user_id == user.id).order_by(ChatMessage.id.desc()).limit(60)).all()
    return [{"role": m.role, "content": m.content, "actions": (m.data or {}).get("actions", []),
             "ts": m.created_at.isoformat()} for m in reversed(msgs)]


@api.delete("/chat")
def clear_chat(user: User = Depends(current_user), s: Session = Depends(get_session)):
    for m in s.exec(select(ChatMessage).where(ChatMessage.user_id == user.id)).all():
        s.delete(m)
    s.commit()
    return {"ok": True}


@api.post("/chat")
async def chat(body: ChatIn, user: User = Depends(current_user), s: Session = Depends(get_session)):
    limiter.check(f"chat:{user.id}", 30, 60)
    prior = s.exec(select(ChatMessage).where(ChatMessage.user_id == user.id).order_by(ChatMessage.id.desc()).limit(12)).all()
    history = [{"role": m.role, "content": m.content} for m in reversed(prior) if m.content]
    while history and history[0]["role"] != "user":
        history.pop(0)
    bot = Copilot(user, s, body.attachment)
    try:
        reply = await bot.respond(history, body.message)
    except Exception as e:
        reply = f"Something went wrong talking to the model ({e.__class__.__name__}). Try again."
    s.add(ChatMessage(user_id=user.id, role="user", content=body.message,
                      data={"attachment": bool(body.attachment)}))
    s.add(ChatMessage(user_id=user.id, role="assistant", content=reply, data={"actions": bot.actions}))
    s.commit()
    return {"reply": reply, "actions": bot.actions}
