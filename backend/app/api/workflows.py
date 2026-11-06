"""Workflows: a file or email in, every row classified by the best deployed model, results back out."""
from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from .. import labs as lab_service
from .. import workflows as wf
from ..config import get_settings
from ..db import Candidate, Deployment, Feedback, Lab, LabEvent, User, get_session
from ..events import bus, serialize
from ..export import to_csv, to_xlsx
from ..ingest import parse_upload
from ..security import client_ip, current_user, limiter

router = APIRouter(prefix="/api/workflows")


def _own(wf_id: str, user: User, s: Session) -> wf.Workflow:
    w = s.get(wf.Workflow, wf_id)
    if not w or w.user_id != user.id:
        raise HTTPException(404, "Workflow not found")
    return w


def _out(w: wf.Workflow, s: Session) -> dict[str, Any]:
    dep = s.get(Deployment, w.deployment_id) if w.deployment_id else None
    cand = s.get(Candidate, dep.candidate_id) if dep else None
    return {"id": w.id, "name": w.name, "prompt": w.prompt, "status": w.status, "error": w.error,
            "source_kind": w.source_kind, "source_name": w.source_name, "rows_total": w.rows_total,
            "rows_done": w.rows_done, "rows_review": w.rows_review, "confidence_floor": w.confidence_floor,
            "summary": w.summary or {}, "lab_id": w.lab_id, "created_at": w.created_at.isoformat(),
            "finished_at": w.finished_at.isoformat() if w.finished_at else None,
            "deployment": {"slug": dep.slug, "name": dep.name, "status": dep.status,
                           "model": cand.display_name if cand else None,
                           "labels": (s.get(Lab, dep.lab_id).spec or {}).get("labels", [])} if dep else None}


def _row_out(r: wf.WorkflowRow) -> dict[str, Any]:
    return {"id": r.id, "idx": r.idx, "text": r.text, "meta": r.meta, "label": r.label, "confidence": r.confidence,
            "route": r.route, "top": r.top, "needs_review": r.needs_review, "reviewed_label": r.reviewed_label,
            "truth": r.meta.get("label")}


@router.post("")
async def create_workflow(request: Request, file: UploadFile | None = File(None), prompt: str = Form(""),
                          deployment: str | None = Form(None), confidence_floor: float = Form(0.6),
                          text: str | None = Form(None), user: User = Depends(current_user),
                          s: Session = Depends(get_session)):
    limiter.check(f"wf:{user.id}", 20, 3600)
    settings = get_settings()
    if file is not None:
        raw = await file.read(settings.max_upload_bytes + 1)
        if len(raw) > settings.max_upload_bytes:
            raise HTTPException(413, "File too large (5 MB max).")
        parsed = parse_upload(file.filename or "upload", raw, file.content_type or "")
        source_name = file.filename or "upload"
    elif text and text.strip():
        parsed = parse_upload("pasted.txt", text.encode("utf-8"))
        source_name = "pasted text"
    else:
        raise HTTPException(422, "Attach a file or paste text.")
    if not parsed.rows:
        raise HTTPException(422, "Couldn't find rows to work on. " + " ".join(parsed.notes))
    dep = None
    if deployment:
        dep = s.exec(select(Deployment).where(Deployment.slug == deployment)).first()
        if not dep or dep.user_id != user.id:
            raise HTTPException(404, "Model not found")
    w = wf.Workflow(user_id=user.id, name=(source_name if file is not None else (prompt.strip()[:60] or "Pasted text")),
                    prompt=prompt.strip()[:2000], source_kind=parsed.kind, source_name=source_name,
                    deployment_id=dep.id if dep else None, confidence_floor=max(0.0, min(0.99, confidence_floor)),
                    rows_total=len(parsed.rows), summary={"parse": parsed.summary()})
    s.add(w)
    s.commit()
    s.refresh(w)
    s.add_all([wf.WorkflowRow(workflow_id=w.id, idx=i, text=r["text"], meta={**r.get("meta", {}), **({"label": r["label"]} if r.get("label") else {})})
               for i, r in enumerate(parsed.rows)])
    s.commit()
    wf.start(w.id)
    return _out(w, s)


@router.get("")
def list_workflows(user: User = Depends(current_user), s: Session = Depends(get_session)):
    rows = s.exec(select(wf.Workflow).where(wf.Workflow.user_id == user.id).order_by(wf.Workflow.created_at.desc()).limit(100)).all()
    return [_out(w, s) for w in rows]


@router.get("/{wf_id}")
def get_workflow(wf_id: str, user: User = Depends(current_user), s: Session = Depends(get_session)):
    return _out(_own(wf_id, user, s), s)


@router.get("/{wf_id}/rows")
def list_rows(wf_id: str, review: bool = False, offset: int = 0, limit: int = 100, user: User = Depends(current_user),
              s: Session = Depends(get_session)):
    _own(wf_id, user, s)
    q = select(wf.WorkflowRow).where(wf.WorkflowRow.workflow_id == wf_id)
    if review:
        q = q.where(wf.WorkflowRow.needs_review == True)  # noqa: E712
    rows = s.exec(q.order_by(wf.WorkflowRow.idx).offset(max(0, offset)).limit(max(1, min(limit, 500)))).all()
    return [_row_out(r) for r in rows]


class ReviewIn(BaseModel):
    label: str = Field(min_length=1, max_length=200)


@router.post("/{wf_id}/rows/{row_id}/review")
def review_row(wf_id: str, row_id: int, body: ReviewIn, user: User = Depends(current_user), s: Session = Depends(get_session)):
    w = _own(wf_id, user, s)
    r = s.get(wf.WorkflowRow, row_id)
    if not r or r.workflow_id != wf_id:
        raise HTTPException(404, "Row not found")
    dep = s.get(Deployment, w.deployment_id) if w.deployment_id else None
    labels = (s.get(Lab, dep.lab_id).spec or {}).get("labels", []) if dep else []
    if labels and body.label not in labels:
        raise HTTPException(422, "Label must be one of the model's labels.")
    r.reviewed_label = body.label
    s.add(r)
    # A human decision is real labeled data: it feeds the model's next retrain.
    if dep and r.label != body.label:
        s.add(Feedback(deployment_id=dep.id, text=r.text, predicted=r.label or "", correct=body.label))
    s.commit()
    return _row_out(r)


@router.post("/{wf_id}/teach")
def teach_from_workflow(wf_id: str, user: User = Depends(current_user), s: Session = Depends(get_session)):
    """Turn a finished workflow into training data: a person's decisions are labels, the file's own label column
    is a label, and everything else goes in unlabeled for the teacher model to label by agreement. The lab that
    starts keeps the deployed model's task so the new model is a drop-in successor."""
    w = _own(wf_id, user, s)
    if w.status not in ("completed", "failed"):
        raise HTTPException(409, "Wait for the workflow to finish first.")
    rows = s.exec(select(wf.WorkflowRow).where(wf.WorkflowRow.workflow_id == wf_id).order_by(wf.WorkflowRow.idx)).all()
    if not rows:
        raise HTTPException(422, "This workflow has no rows to learn from.")
    seeds = [{"text": r.text, "label": r.reviewed_label or r.meta.get("label") or None} for r in rows]
    dep = s.get(Deployment, w.deployment_id) if w.deployment_id else None
    old = s.get(Lab, dep.lab_id) if dep else None
    prompt = old.prompt if old else (w.prompt or "Classify these rows")
    if old:
        seeds = list(old.seed_examples or []) + seeds
    lab = lab_service.create_lab(s, user, prompt, seeds)
    return {"lab_id": lab.id, "labeled": sum(1 for x in seeds if x.get("label")),
            "unlabeled": sum(1 for x in seeds if not x.get("label")), "from_deployment": dep.slug if dep else None}


@router.get("/{wf_id}/export")
def export_workflow(wf_id: str, format: str = "xlsx", user: User = Depends(current_user), s: Session = Depends(get_session)):
    w = _own(wf_id, user, s)
    rows = s.exec(select(wf.WorkflowRow).where(wf.WorkflowRow.workflow_id == wf_id).order_by(wf.WorkflowRow.idx)).all()
    data = [{"text": r.text, "meta": {k: v for k, v in r.meta.items() if k != "label"}} for r in rows]
    preds = [{"label": r.reviewed_label or r.label or "", "confidence": r.confidence, "needs_review": r.needs_review and not r.reviewed_label,
              "top": r.top} for r in rows]
    stem = quote((w.name.rsplit(".", 1)[0] or "results")[:60])
    if format == "csv":
        return Response(to_csv(data, preds), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{stem}-results.csv"'})
    return Response(to_xlsx(data, preds, title=w.name[:31]), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{stem}-results.xlsx"'})


@router.delete("/{wf_id}")
def delete_workflow(wf_id: str, user: User = Depends(current_user), s: Session = Depends(get_session)):
    w = _own(wf_id, user, s)
    if w.status not in ("completed", "failed"):
        raise HTTPException(409, "Wait for the workflow to finish first.")
    for r in s.exec(select(wf.WorkflowRow).where(wf.WorkflowRow.workflow_id == wf_id)).all():
        s.delete(r)
    for e in s.exec(select(LabEvent).where(LabEvent.lab_id == wf_id)).all():
        s.delete(e)
    s.delete(w)
    s.commit()
    return {"ok": True}


@router.get("/{wf_id}/events")
async def workflow_events(wf_id: str, request: Request, after: int = 0, user: User = Depends(current_user),
                          s: Session = Depends(get_session)):
    w = _own(wf_id, user, s)
    terminal = w.status in ("completed", "failed")
    try:
        after = max(after, int(request.headers.get("last-event-id") or 0))
    except ValueError:
        pass
    q = bus.subscribe(wf_id)
    past = [serialize(e) for e in s.exec(select(LabEvent).where(LabEvent.lab_id == wf_id, LabEvent.seq > after).order_by(LabEvent.seq)).all()]
    s.close()

    def is_terminal(ev: dict) -> bool:
        return ev["kind"] == "error" or (ev.get("data") or {}).get("stage") in ("done", "failed")

    async def gen():
        try:
            last = after
            for ev in past:
                last = ev["seq"]
                yield {"event": "lab", "id": str(ev["seq"]), "data": json.dumps(ev)}
            if terminal:
                return
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
                if is_terminal(ev):
                    break
        finally:
            bus.unsubscribe(wf_id, q)

    return EventSourceResponse(gen(), headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache, no-transform"})
