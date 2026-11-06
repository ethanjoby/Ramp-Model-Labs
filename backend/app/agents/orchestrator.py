"""The lab director: plan → curate → train in parallel → benchmark serially → judge → refit → deploy."""
from __future__ import annotations

import asyncio
import os
import re
import secrets
import traceback

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from .. import llm
from ..config import DATA_DIR
from ..db import Candidate, Deployment, Lab, LabEvent, engine, now
from ..events import bus
from ..trainers.base import LabContext, TaskSpec, score
from ..trainers.registry import strategies
from . import data as data_agent
from . import planner, selector

# Process-wide, so ten labs at once share the machine instead of each assuming it is alone.
_CPU = asyncio.Semaphore(max(2, (os.cpu_count() or 4) // 2))
_LLM = asyncio.Semaphore(6)
_GPU = asyncio.Semaphore(1)  # one local-GPU trainer at a time: three encoders at once would thrash 16 GB of unified memory
_LAB_SLOTS = asyncio.Semaphore(3)
CANDIDATE_TIMEOUT_S = {"finetuned": 25 * 60}
DEFAULT_TIMEOUT_S = 20 * 60
_running: dict[str, asyncio.Task] = {}


def _slug(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "model"
    return f"{base}-{secrets.token_hex(4)}"


def _update_lab(lab_id: str, **fields) -> None:
    with Session(engine) as s:
        lab = s.get(Lab, lab_id)
        for k, v in fields.items():
            setattr(lab, k, v)
        s.add(lab)
        s.commit()


async def _mark_failed(lab_id: str, msg: str) -> None:
    """The failure write must not itself be the thing that leaves a lab 'running' forever."""
    for attempt in range(4):
        try:
            _update_lab(lab_id, status="failed", error=msg[:500], finished_at=now())
            with Session(engine) as s:
                for c in s.exec(select(Candidate).where(Candidate.lab_id == lab_id, Candidate.status.in_(["queued", "running"]))).all():
                    c.status, c.metrics = "failed", {"error": "lab failed before this candidate finished"}
                    s.add(c)
                s.commit()
            return
        except Exception:
            traceback.print_exc()
            await asyncio.sleep(0.5 * (attempt + 1))


def start(lab_id: str) -> None:
    _running[lab_id] = asyncio.create_task(run_lab(lab_id))


async def run_lab(lab_id: str) -> None:
    async def emit(agent: str, kind: str, message: str = "", data: dict | None = None):
        await bus.emit(lab_id, agent, kind, message, data)

    usage = llm.Usage()
    try:
        async with _LAB_SLOTS:
            with Session(engine) as s:
                lab = s.get(Lab, lab_id)
                prompt, seeds, user_id = lab.prompt, list(lab.seed_examples or []), lab.user_id
            _update_lab(lab_id, status="running")
            mode = "Claude-powered agents" if llm.available() else "offline mode (procedural data + local training)"
            await emit("director", "log", f"lab started · {mode}")

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

            # 3. train every viable strategy on the same splits
            await emit("trainer", "stage", "Training candidates", {"stage": "train"})
            art_dir = str(DATA_DIR / "models" / lab_id)
            os.makedirs(art_dir, exist_ok=True)
            ctx = LabContext(lab_id, spec, ds["train"], ds["dev"], ds["test"], emit, art_dir)
            strats = strategies()
            by_key = {st.key: st for st in strats}
            cand_ids: dict[str, str] = {}
            with Session(engine) as s:
                for st in strats:
                    ok, reason = st.available()
                    c = Candidate(lab_id=lab_id, strategy=st.key, display_name=st.display_name, family=st.family,
                                  status="queued" if ok else "skipped", metrics={} if ok else {"skip_reason": reason})
                    s.add(c)
                    s.commit()
                    cand_ids[st.key] = c.id
            await emit("trainer", "log", "candidate roster", {"roster": [
                {"key": st.key, "name": st.display_name, "family": st.family, "blurb": st.blurb,
                 "available": st.available()[0], "reason": st.available()[1]} for st in strats]})

            results: list[dict] = []
            spend = 0.0

            async def run_one(st):
                nonlocal spend
                sem = _CPU if st.family == "classical" else _GPU if getattr(st, "uses_gpu", None) == "local" else _LLM
                cid = cand_ids[st.key]
                async with sem:
                    await emit("trainer", "candidate", f"training {st.display_name}", {"key": st.key, "status": "running"})
                    with Session(engine) as s:
                        c = s.get(Candidate, cid); c.status = "running"; s.add(c); s.commit()
                    try:
                        res = await asyncio.wait_for(st.run(ctx), timeout=CANDIDATE_TIMEOUT_S.get(st.family, DEFAULT_TIMEOUT_S))
                        m = score(spec, ds["test"], res)
                        if st.family != "classical":
                            spend += res.train_cost + res.eval_cost
                        with Session(engine) as s:
                            c = s.get(Candidate, cid)
                            c.status, c.metrics, c.artifact_path = "done", m, res.artifact_path
                            s.add(c); s.commit()
                        results.append({"key": st.key, "id": cid, "name": st.display_name, "family": st.family,
                                        "metrics": m, "artifact_path": res.artifact_path})
                        await emit("evaluator", "candidate",
                                   f"{st.display_name} · acc {m['accuracy']:.1%} · f1 {m['macro_f1']:.3f} · "
                                   f"${m['cost_per_1k']:.5f}/1k · p50 {m['latency_p50_ms']:.1f}ms",
                                   {"key": st.key, "status": "done", "metrics": m})
                    except Exception as e:
                        if isinstance(e, asyncio.TimeoutError):
                            e = TimeoutError(f"gave up after {CANDIDATE_TIMEOUT_S.get(st.family, DEFAULT_TIMEOUT_S) // 60} minutes")
                        traceback.print_exc()
                        with Session(engine) as s:
                            c = s.get(Candidate, cid); c.status = "failed"; c.metrics = {"error": str(e)[:300]}
                            s.add(c); s.commit()
                        await emit("trainer", "candidate", f"{st.display_name} failed: {str(e)[:120]}",
                                   {"key": st.key, "status": "failed"})

            await asyncio.gather(*(run_one(st) for st in strats if st.available()[0]))

            # 3b. one serial benchmark pass: latency sampled while siblings trained is noise, not a measurement
            for r in results:
                if not r.get("artifact_path"):
                    continue
                try:
                    bm = await by_key[r["key"]].benchmark(ctx, r["artifact_path"])
                except Exception as e:
                    await emit("evaluator", "log", f"{r['name']} · benchmark skipped ({e.__class__.__name__})")
                    bm = None
                if not bm:
                    continue
                lat, cost_1k = bm
                lat = sorted(lat)
                r["metrics"].update(latency_p50_ms=round(lat[len(lat) // 2], 3),
                                    latency_p95_ms=round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 3), cost_per_1k=cost_1k)
                with Session(engine) as s:
                    c = s.get(Candidate, r["id"]); c.metrics = r["metrics"]; s.add(c); s.commit()
                await emit("evaluator", "candidate", f"{r['name']} · re-measured on an idle machine · p50 {r['metrics']['latency_p50_ms']:.1f}ms · ${cost_1k:.5f}/1k",
                           {"key": r["key"], "status": "done", "metrics": r["metrics"]})

            # 4. select (the frontier reference is a yardstick, never deployed)
            await emit("selector", "stage", "Selecting the winner", {"stage": "select"})
            eligible = [r for r in results if r["family"] != "reference"]
            winner, front, why = selector.choose(spec, eligible, imbalance=ds["stats"].get("imbalance"))
            if not winner:
                raise RuntimeError("Every candidate failed to train.")
            meets_bar = bool(winner.get("meets_bar", True))
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
            await emit("selector", "result", rationale, {"winner": winner["key"], "pareto": front, "savings": savings, "meets_bar": meets_bar})

            # 4b. the deployed model gets every row; the reported score stays the held-out one
            all_rows = ds["train"] + ds["dev"] + ds["test"]
            try:
                if winner.get("artifact_path") and await by_key[winner["key"]].refit(ctx, all_rows, winner["artifact_path"]):
                    await emit("trainer", "log", f"{winner['name']} refit on all {len(all_rows)} rows for deployment; the scores above are from the held-out fit",
                               {"refit_rows": len(all_rows)})
            except Exception as e:
                await emit("trainer", "log", f"refit skipped ({e.__class__.__name__}); deploying the held-out fit")

            # 5. deploy
            await emit("deployer", "stage", "Deploying endpoint", {"stage": "deploy"})
            status = "live" if meets_bar else "draft"
            with Session(engine) as s:
                for attempt in range(5):
                    dep = Deployment(user_id=user_id, lab_id=lab_id, candidate_id=winner["id"], slug=_slug(spec.name),
                                     name=spec.name, status=status)
                    s.add(dep)
                    try:
                        s.commit()
                        break
                    except IntegrityError:
                        s.rollback()
                        if attempt == 4:
                            raise
                s.refresh(dep)
                slug, dep_id = dep.slug, dep.id
            if meets_bar:
                await emit("deployer", "log", f"live at POST /v1/models/{slug}/predict", {"slug": slug})
            else:
                await emit("deployer", "log", f"trained but not published: it missed the {spec.target_accuracy:.0%} bar. The playground works; publish it from the model page if you accept the trade-off.", {"slug": slug, "draft": True})

            result = {"winner": winner["key"], "winner_name": winner["name"], "winner_metrics": winner["metrics"],
                      "pareto": front, "rationale": rationale, "savings": savings, "deployment": {"slug": slug, "id": dep_id},
                      "meets_bar": meets_bar, "agent_spend": round(usage.cost + spend, 4)}
            _update_lab(lab_id, status="completed", result=result, finished_at=now())
            await emit("director", "result", "lab complete", {"stage": "done", "result": result})
    except Exception as e:
        traceback.print_exc()
        msg = str(e) if isinstance(e, (planner.PlanningError, ValueError, RuntimeError)) else f"{e.__class__.__name__}: {e}"
        await _mark_failed(lab_id, msg)
        try:
            await emit("director", "error", msg[:500], {"stage": "failed"})
        except Exception:
            traceback.print_exc()
    finally:
        _running.pop(lab_id, None)


def recover_interrupted() -> None:
    """Labs left queued/running by a restart are closed out, candidates included, so nothing shows as building forever."""
    with Session(engine) as s:
        for lab in s.exec(select(Lab).where(Lab.status.in_(["queued", "running"]))).all():
            lab.status, lab.error, lab.finished_at = "failed", "Server restarted during the run. Start it again.", now()
            s.add(lab)
            for c in s.exec(select(Candidate).where(Candidate.lab_id == lab.id, Candidate.status.in_(["queued", "running"]))).all():
                c.status, c.metrics = "failed", {"error": "interrupted by server restart"}
                s.add(c)
            last = s.exec(select(LabEvent.seq).where(LabEvent.lab_id == lab.id).order_by(LabEvent.seq.desc())).first() or 0
            s.add(LabEvent(lab_id=lab.id, seq=last + 1, agent="director", kind="error",
                           message="Server restarted during the run. Start it again.", data={"stage": "failed"}))
        s.commit()
