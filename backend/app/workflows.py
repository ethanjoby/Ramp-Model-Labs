"""Workflow data model and runner (filled in next to app/api/workflows.py)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from .db import now, uid


class Workflow(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uid("wf"), primary_key=True)
    user_id: str = Field(index=True, foreign_key="user.id")
    name: str
    prompt: str = ""
    source_kind: str = "text"
    source_name: str = ""
    deployment_id: Optional[str] = None
    lab_id: Optional[str] = None
    status: str = "queued"  # queued|choosing|building|classifying|completed|failed
    rows_total: int = 0
    rows_done: int = 0
    rows_review: int = 0
    confidence_floor: float = 0.6
    summary: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=now)
    finished_at: Optional[datetime] = None


class WorkflowRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_id: str = Field(index=True, foreign_key="workflow.id")
    idx: int
    text: str
    meta: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    label: Optional[str] = None
    confidence: Optional[float] = None
    route: Optional[str] = None
    top: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    needs_review: bool = False
    reviewed_label: Optional[str] = None


# ------------------------------------------------------------------ runner

import asyncio  # noqa: E402
import re  # noqa: E402
import traceback  # noqa: E402

from sqlmodel import Session, select  # noqa: E402

from . import labs  # noqa: E402
from .datasets.templates import match_template  # noqa: E402
from .db import Candidate, Deployment, Lab, User, engine  # noqa: E402
from .events import bus  # noqa: E402
from .inference import predict  # noqa: E402

BATCH = 100
BUILD_TIMEOUT_S = 25 * 60
_running: dict[str, asyncio.Task] = {}
_WORD = re.compile(r"[a-z][a-z0-9]{2,}")


def _update(wf_id: str, **fields: Any) -> None:
    with Session(engine) as s:
        wf = s.get(Workflow, wf_id)
        for k, v in fields.items():
            setattr(wf, k, v)
        s.add(wf)
        s.commit()


def start(wf_id: str) -> None:
    _running[wf_id] = asyncio.create_task(run(wf_id))


def _words(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def pick_deployment(s: Session, user_id: str, prompt: str, row_labels: set[str]) -> tuple[Deployment | None, str]:
    """The model whose task best matches the request; None when nothing clearly fits.

    A labeled upload defines the task: a model whose labels do not overlap it is never a match, however
    well the prompt words line up, and a draft (below-bar) model whose labels match beats a live one that
    speaks a different label set.
    """
    deps = s.exec(select(Deployment).where(Deployment.user_id == user_id, Deployment.status.in_(["live", "draft"]))
                  .order_by(Deployment.created_at.desc())).all()
    if not deps:
        return None, "no deployed models"
    want = _words(prompt)
    scored = []
    for d in deps:
        lab = s.get(Lab, d.lab_id)
        spec = (lab.spec or {}) if lab else {}
        labels = set(spec.get("labels", []))
        if row_labels and not (row_labels & labels):
            continue
        have = _words(f"{d.name} {lab.prompt if lab else ''} {spec.get('description', '')} {' '.join(labels)}")
        score = len(want & have) * 2 + (10 if row_labels and row_labels <= labels else 0) + (len(row_labels & labels) if row_labels else 0)
        if d.status == "draft":
            score -= 1  # a published model wins a tie, but a draft that knows the labels beats a stranger
        scored.append((score, d))
    if not scored:
        return None, "no model knows these labels"
    score, best = max(scored, key=lambda x: x[0])
    note = " (below its accuracy bar, not published)" if best.status == "draft" else ""
    if score > 0:
        return best, f"matched {best.name}{note}"
    live = [d for _, d in scored if d.status == "live"]
    if len(live) == 1 and not want and not row_labels:
        return live[0], f"only one model, {live[0].name}"
    return None, "no model matched the request"


async def _wait_for_lab(lab_id: str, emit) -> Lab:
    deadline = asyncio.get_running_loop().time() + BUILD_TIMEOUT_S
    last_stage = None
    while True:
        with Session(engine) as s:
            lab = s.get(Lab, lab_id)
            status, stage = lab.status, (lab.spec or {}).get("name")
            s.expunge(lab)
        if status in ("completed", "failed"):
            return lab
        if stage != last_stage and stage:
            last_stage = stage
            await emit("builder", "log", f"lab is building {stage}")
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("Building a model took too long.")
        await asyncio.sleep(2)


async def run(wf_id: str) -> None:
    async def emit(agent: str, kind: str, message: str = "", data: dict | None = None):
        await bus.emit(wf_id, agent, kind, message, data)

    try:
        with Session(engine) as s:
            wf = s.get(Workflow, wf_id)
            user = s.get(User, wf.user_id)
            rows = s.exec(select(WorkflowRow).where(WorkflowRow.workflow_id == wf_id).order_by(WorkflowRow.idx)).all()
            row_labels = {r.meta.get("label") for r in rows if r.meta.get("label")}
            prompt, floor, dep_id, kind, name = wf.prompt, wf.confidence_floor, wf.deployment_id, wf.source_kind, wf.source_name
            s.expunge_all()
        await emit("router", "stage", f"{len(rows)} rows parsed from {kind}{' ' + name if name else ''}", {"stage": "parse", "rows": len(rows)})

        # 1. pick a model, or build one
        _update(wf_id, status="choosing")
        await emit("router", "stage", "Choosing a model", {"stage": "model"})
        with Session(engine) as s:
            dep = s.get(Deployment, dep_id) if dep_id else None
            why = "chosen by you" if dep else ""
            if not dep:
                dep, why = pick_deployment(s, wf.user_id, prompt, row_labels)
            if dep:
                s.expunge(dep)
        if not dep:
            can_build = bool(row_labels and len(row_labels) >= 2) or match_template(prompt) is not None
            if not can_build:
                raise RuntimeError("No deployed model fits this request, and there isn't enough to build one: "
                                   "upload labeled rows or describe one of the built-in workflows.")
            _update(wf_id, status="building")
            await emit("builder", "log", f"{why}; building a new model" + (" from your labeled rows" if row_labels else " for this task"))
            with Session(engine) as s:
                user = s.get(User, wf.user_id)
                seeds = [{"text": r.text, "label": r.meta.get("label")} for r in rows if r.meta.get("label")]
                lab = labs.create_lab(s, user, prompt or "Classify these rows", seeds)
                lab_id = lab.id
            _update(wf_id, lab_id=lab_id)
            await emit("builder", "stage", "Building a model", {"stage": "build", "lab_id": lab_id})
            lab = await _wait_for_lab(lab_id, emit)
            if lab.status != "completed":
                raise RuntimeError(f"The model build failed: {lab.error}")
            with Session(engine) as s:
                dep = s.get(Deployment, (lab.result or {}).get("deployment", {}).get("id"))
                s.expunge(dep)
            why = f"built {dep.name}"
        _update(wf_id, deployment_id=dep.id)
        with Session(engine) as s:
            cand = s.get(Candidate, dep.candidate_id)
            s.expunge(cand)
        await emit("router", "log", f"model: {dep.name} ({cand.display_name}), {why}", {"deployment": dep.slug, "model": cand.display_name})

        # 2. classify every row
        _update(wf_id, status="classifying")
        await emit("classifier", "stage", "Classifying rows", {"stage": "classify"})
        total_ms, done, review, dist, confs, correct, labeled = 0.0, 0, 0, {}, [], 0, 0
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            preds, ms = await predict(cand, [r.text for r in batch])
            total_ms += ms
            with Session(engine) as s:
                for r, p in zip(batch, preds):
                    db_row = s.get(WorkflowRow, r.id)
                    label, conf, route = p.get("label"), p.get("confidence"), p.get("route")
                    flag = label is None or route == "review" or (conf is not None and conf < floor)
                    db_row.label, db_row.confidence, db_row.route, db_row.top, db_row.needs_review = label, conf, route, p.get("top") or [], flag
                    s.add(db_row)
                    done += 1
                    review += int(flag)
                    if label:
                        dist[label] = dist.get(label, 0) + 1
                    if conf is not None:
                        confs.append(conf)
                    truth = r.meta.get("label")
                    if truth:
                        labeled += 1
                        correct += int(truth == label)
                s.commit()
            _update(wf_id, rows_done=done, rows_review=review)
            await emit("classifier", "log", f"{done}/{len(rows)} rows classified, {review} sent to review",
                       {"stage": "classify", "done": done, "total": len(rows), "review": review})

        # 3. summary
        metrics = cand.metrics or {}
        with Session(engine) as s:
            lab_for_dep = s.get(Lab, dep.lab_id)
            savings = (lab_for_dep.result or {}).get("savings") if lab_for_dep else None
        cost_1k = metrics.get("cost_per_1k") or 0.0
        summary = {
            "model": {"slug": dep.slug, "name": dep.name, "candidate": cand.display_name, "status": dep.status,
                      "accuracy": metrics.get("accuracy"), "eval_source": None},
            "distribution": dict(sorted(dist.items(), key=lambda kv: -kv[1])),
            "rows": len(rows), "review": review, "auto": len(rows) - review,
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "latency_ms": round(total_ms, 1),
            "cost_usd": round(cost_1k * len(rows) / 1000, 6),
            "frontier_cost_usd": round((savings or {}).get("frontier_cost_per_1k", 0) * len(rows) / 1000, 4) if savings else None,
            "frontier_measured": bool((savings or {}).get("frontier_measured")) if savings else False,
            "accuracy_on_labeled": round(correct / labeled, 4) if labeled else None,
            "labeled_rows": labeled,
        }
        _update(wf_id, status="completed", summary={**(wf.summary or {}), **summary}, finished_at=now())
        await emit("router", "result", f"done: {len(rows) - review} rows handled automatically, {review} for review",
                   {"stage": "done", "summary": summary})
    except Exception as e:
        traceback.print_exc()
        msg = str(e)[:500] or e.__class__.__name__
        _update(wf_id, status="failed", error=msg, finished_at=now())
        await emit("router", "error", msg, {"stage": "failed"})
    finally:
        _running.pop(wf_id, None)
