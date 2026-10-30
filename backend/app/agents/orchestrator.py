"""The lab director: plan → curate → train in parallel → evaluate → select → deploy."""
from __future__ import annotations

import asyncio
import os
import re
import secrets
import traceback

from sqlmodel import Session, select

from .. import llm
from ..config import DATA_DIR
from ..db import Candidate, Deployment, Lab, engine, now
from ..events import bus
from ..trainers.base import LabContext, TaskSpec, score
from ..trainers.registry import strategies
from . import data as data_agent
from . import planner, selector

_running: dict[str, asyncio.Task] = {}


def _slug(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "model"
    return f"{base}-{secrets.token_hex(2)}"


def _update_lab(lab_id: str, **fields) -> None:
    with Session(engine) as s:
        lab = s.get(Lab, lab_id)
        for k, v in fields.items():
            setattr(lab, k, v)
        s.add(lab)
        s.commit()


def start(lab_id: str) -> None:
    _running[lab_id] = asyncio.create_task(run_lab(lab_id))


async def run_lab(lab_id: str) -> None:
    async def emit(agent: str, kind: str, message: str = "", data: dict | None = None):
        await bus.emit(lab_id, agent, kind, message, data)

    usage = llm.Usage()
    with Session(engine) as s:
        lab = s.get(Lab, lab_id)
        prompt, seeds, user_id = lab.prompt, list(lab.seed_examples or []), lab.user_id
    _update_lab(lab_id, status="running")
    mode = "Claude-powered agents" if llm.available() else "offline mode (procedural data + local training)"
    await emit("director", "log", f"lab started · {mode}")

    try:
        # 1. plan
        await emit("planner", "stage", "Analyzing the task", {"stage": "plan"})
        spec: TaskSpec = await planner.plan(prompt, seeds, usage)
        _update_lab(lab_id, spec=spec.to_dict(), name=spec.name)
        await emit("planner", "metric",
                   f"spec · {spec.name} · {len(spec.labels)} labels · target ≥{spec.target_accuracy:.0%} · optimize for {spec.priority}",
                   {"spec": spec.to_dict()})

        # 2. data
        await emit("data", "stage", "Curating training data", {"stage": "data"})
        ds = await data_agent.curate(spec, seeds, emit, usage)
        _update_lab(lab_id, dataset_stats=ds["stats"])

        # 3. train + evaluate every viable strategy on the same held-out set
        await emit("trainer", "stage", "Training candidates", {"stage": "train"})
        art_dir = str(DATA_DIR / "models" / lab_id)
        os.makedirs(art_dir, exist_ok=True)
        ctx = LabContext(lab_id, spec, ds["train"], ds["dev"], ds["test"], emit, art_dir)

        strats = strategies()
        cand_ids: dict[str, str] = {}
        with Session(engine) as s:
            for st in strats:
                ok, reason = st.available()
                c = Candidate(lab_id=lab_id, strategy=st.key, display_name=st.display_name, family=st.family,
                              status="queued" if ok else "skipped",
                              metrics={} if ok else {"skip_reason": reason})
                s.add(c)
                s.commit()
                cand_ids[st.key] = c.id
        await emit("trainer", "log", "candidate roster", {"roster": [
            {"key": st.key, "name": st.display_name, "family": st.family, "blurb": st.blurb,
             "available": st.available()[0], "reason": st.available()[1]} for st in strats]})

        results: list[dict] = []
        sem_cpu, sem_llm = asyncio.Semaphore(2), asyncio.Semaphore(3)

        async def run_one(st):
            sem = sem_cpu if st.family == "classical" else sem_llm
            async with sem:
                cid = cand_ids[st.key]
                await emit("trainer", "candidate", f"training {st.display_name}", {"key": st.key, "status": "running"})
                with Session(engine) as s:
                    c = s.get(Candidate, cid); c.status = "running"; s.add(c); s.commit()
                try:
                    res = await st.run(ctx)
                    m = score(spec, ds["test"], res)
                    usage.cost += res.train_cost if st.family != "classical" else 0
                    with Session(engine) as s:
                        c = s.get(Candidate, cid)
                        c.status, c.metrics, c.artifact_path = "done", m, res.artifact_path
                        s.add(c); s.commit()
                    results.append({"key": st.key, "id": cid, "name": st.display_name, "family": st.family,
                                    "metrics": m})
                    await emit("evaluator", "candidate",
                               f"{st.display_name} · acc {m['accuracy']:.1%} · f1 {m['macro_f1']:.3f} · "
                               f"${m['cost_per_1k']:.5f}/1k · p50 {m['latency_p50_ms']:.1f}ms",
                               {"key": st.key, "status": "done", "metrics": m})
                except Exception as e:
                    traceback.print_exc()
                    with Session(engine) as s:
                        c = s.get(Candidate, cid); c.status = "failed"; c.metrics = {"error": str(e)[:300]}
                        s.add(c); s.commit()
                    await emit("trainer", "candidate", f"{st.display_name} failed: {str(e)[:120]}",
                               {"key": st.key, "status": "failed"})

        await asyncio.gather(*(run_one(st) for st in strats if st.available()[0]))

        # 4. select (the frontier reference is a yardstick, never deployed)
        await emit("selector", "stage", "Selecting the winner", {"stage": "select"})
        eligible = [r for r in results if r["family"] != "reference"]
        winner, front, why = selector.choose(spec, eligible)
        if not winner:
            raise RuntimeError("Every candidate failed to train.")
        ref = next((r for r in results if r["family"] == "reference"), None)
        frontier = ref["metrics"]["cost_per_1k"] if ref else selector.frontier_cost_per_1k(spec, ds["test"])
        wcost = max(winner["metrics"]["cost_per_1k"], 1e-7)
        savings = {
            "frontier_cost_per_1k": frontier, "frontier_measured": bool(ref),
            "frontier_accuracy": ref["metrics"]["accuracy"] if ref else None,
            "winner_cost_per_1k": winner["metrics"]["cost_per_1k"],
            "multiple": round(frontier / wcost, 1),
            "monthly_at_1m": {"winner": round(wcost * 1000, 2), "frontier": round(frontier * 1000, 2)},
        }
        rationale = await selector.explain(spec, winner, results, why, savings, usage)
        with Session(engine) as s:
            for r in results:
                c = s.get(Candidate, r["id"]); c.is_winner = r["id"] == winner["id"]; s.add(c)
            s.commit()
        await emit("selector", "result", rationale, {"winner": winner["key"], "pareto": front, "savings": savings})

        # 5. deploy
        await emit("deployer", "stage", "Deploying endpoint", {"stage": "deploy"})
        with Session(engine) as s:
            dep = Deployment(user_id=user_id, lab_id=lab_id, candidate_id=winner["id"], slug=_slug(spec.name),
                             name=spec.name)
            s.add(dep); s.commit(); s.refresh(dep)
            slug, dep_id = dep.slug, dep.id
        await emit("deployer", "log", f"live at POST /v1/models/{slug}/predict", {"slug": slug})

        result = {"winner": winner["key"], "winner_name": winner["name"], "winner_metrics": winner["metrics"],
                  "pareto": front, "rationale": rationale, "savings": savings, "deployment": {"slug": slug, "id": dep_id},
                  "agent_spend": round(usage.cost, 4)}
        _update_lab(lab_id, status="completed", result=result, finished_at=now())
        await emit("director", "result", "lab complete", {"stage": "done", "result": result})
    except Exception as e:
        traceback.print_exc()
        msg = str(e) if isinstance(e, (planner.PlanningError, ValueError)) else f"{e.__class__.__name__}: {e}"
        _update_lab(lab_id, status="failed", error=msg[:500], finished_at=now())
        await emit("director", "error", msg[:500], {"stage": "failed"})
    finally:
        _running.pop(lab_id, None)


def recover_interrupted() -> None:
    """Labs left 'running' by a restart are marked failed rather than hanging forever."""
    with Session(engine) as s:
        for lab in s.exec(select(Lab).where(Lab.status.in_(["queued", "running"]))).all():
            lab.status, lab.error = "failed", "Server restarted during the run. Start it again."
            s.add(lab)
        s.commit()
