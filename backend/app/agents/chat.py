"""Conversational front door. Say what you need; the copilot builds, queries, or runs models."""
from __future__ import annotations

import json
import re
from typing import Any

from sqlmodel import Session, select

from .. import llm
from ..config import get_settings
from ..datasets.templates import match_template
from ..db import Candidate, Deployment, Lab, User
from ..inference import predict
from . import orchestrator
from .data import parse_examples

SYSTEM = """You are the copilot for Ramp Model Labs, a platform that turns one plain-English prompt into a cheap, \
specialized model for a financial workflow (expense categorization, fraud screening, invoice triage, support routing, \
or any custom classification). Behind you, a team of agents plans the task, synthesizes training data, trains and \
fine-tunes several candidates, evaluates them on held-out data, and deploys the cheapest one that meets the bar.

Use tools to act. When the user describes a task they want automated, call build_model right away with a precise \
task prompt (include labels and constraints if they gave any). When they want to classify something, call \
run_model. Keep replies short (1-3 sentences), confident, concrete. No markdown headers."""

TOOLS = [
    {"name": "build_model", "description": "Start a lab that builds and deploys a specialized model for a task.",
     "input_schema": {"type": "object", "properties": {
         "task_prompt": {"type": "string", "description": "Plain-English description of the task, labels, constraints."},
         "use_attachment": {"type": "boolean", "description": "Use the user's attached examples as seed data."}},
         "required": ["task_prompt"]}},
    {"name": "list_models", "description": "List the user's deployed models with accuracy and cost.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_model", "description": "Run a deployed model on one or more inputs.",
     "input_schema": {"type": "object", "properties": {
         "slug": {"type": "string", "description": "Deployment slug; omit to use the most recent model."},
         "inputs": {"type": "array", "items": {"type": "string"}}}, "required": ["inputs"]}},
    {"name": "lab_status", "description": "Get the status and results of the most recent lab or a given lab id.",
     "input_schema": {"type": "object", "properties": {"lab_id": {"type": "string"}}}},
]


class Copilot:
    def __init__(self, user: User, session: Session, attachment: str | None):
        self.user, self.s, self.attachment = user, session, attachment
        self.actions: list[dict[str, Any]] = []

    def _deployments(self) -> list[Deployment]:
        return self.s.exec(select(Deployment).where(Deployment.user_id == self.user.id)
                           .order_by(Deployment.created_at.desc())).all()

    async def build_model(self, task_prompt: str, use_attachment: bool = True) -> dict:
        seeds = parse_examples(self.attachment) if (self.attachment and use_attachment) else []
        lab = Lab(user_id=self.user.id, prompt=task_prompt[:4000], seed_examples=seeds)
        self.s.add(lab); self.s.commit(); self.s.refresh(lab)
        orchestrator.start(lab.id)
        self.actions.append({"type": "lab_started", "lab_id": lab.id, "prompt": task_prompt})
        return {"lab_id": lab.id, "seed_examples": len(seeds), "status": "started"}

    async def list_models(self) -> dict:
        out = []
        for d in self._deployments()[:20]:
            c = self.s.get(Candidate, d.candidate_id)
            out.append({"slug": d.slug, "name": d.name, "model": c.display_name if c else None,
                        "accuracy": (c.metrics or {}).get("accuracy") if c else None,
                        "cost_per_1k": (c.metrics or {}).get("cost_per_1k") if c else None})
        self.actions.append({"type": "models", "models": out})
        return {"models": out}

    async def run_model(self, inputs: list[str], slug: str | None = None) -> dict:
        deps = self._deployments()
        dep = next((d for d in deps if d.slug == slug), None) if slug else (deps[0] if deps else None)
        if not dep:
            return {"error": "No deployed model yet. Build one first."}
        c = self.s.get(Candidate, dep.candidate_id)
        preds, ms = await predict(c, [i[:2000] for i in inputs[:50]])
        dep.requests += len(inputs); dep.total_latency_ms += ms; self.s.add(dep); self.s.commit()
        rows = [{"input": i, **p} for i, p in zip(inputs, preds)]
        self.actions.append({"type": "prediction", "slug": dep.slug, "name": dep.name, "rows": rows,
                             "latency_ms": round(ms, 2)})
        return {"model": dep.name, "predictions": rows}

    async def lab_status(self, lab_id: str | None = None) -> dict:
        q = select(Lab).where(Lab.user_id == self.user.id)
        lab = self.s.exec(q.where(Lab.id == lab_id)).first() if lab_id else \
            self.s.exec(q.order_by(Lab.created_at.desc())).first()
        if not lab:
            return {"error": "No labs yet."}
        self.actions.append({"type": "lab_ref", "lab_id": lab.id})
        r = lab.result or {}
        return {"lab_id": lab.id, "name": lab.name, "status": lab.status, "error": lab.error,
                "winner": r.get("winner_name"), "accuracy": (r.get("winner_metrics") or {}).get("accuracy"),
                "savings_multiple": (r.get("savings") or {}).get("multiple")}

    async def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, name, None)
        if name not in {t["name"] for t in TOOLS} or not fn:
            return {"error": f"unknown tool {name}"}
        try:
            return await fn(**args)
        except Exception as e:
            return {"error": str(e)[:300]}

    async def respond(self, history: list[dict], message: str) -> str:
        if llm.available():
            return await self._llm(history, message)
        return await self._offline(message)

    async def _llm(self, history: list[dict], message: str) -> str:
        content = message
        if self.attachment:
            preview = "\n".join(self.attachment.splitlines()[:8])
            content += f"\n\n[User attached {len(parse_examples(self.attachment))} examples. Preview:\n{preview}]"
        msgs = [*history[-12:], {"role": "user", "content": content}]
        model = get_settings().agent_model
        for _ in range(5):
            resp = await llm.client().messages.create(model=model, max_tokens=1024, system=SYSTEM, tools=TOOLS,
                                                      messages=msgs)
            msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in resp.content]})
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text").strip()
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    out = await self.call(b.name, b.input or {})
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out)[:6000]})
            msgs.append({"role": "user", "content": results})
        return "Done."

    async def _offline(self, message: str) -> str:
        m = message.strip()
        low = m.lower()
        run = re.match(r"^(classify|categorize|run|predict|score|route|check)\b[:\s]+(.+)$", m, re.I | re.S)
        if run:
            inputs = [x.strip(" -•") for x in re.split(r"\n|;(?=\s*[A-Z])", run.group(2)) if x.strip(" -•")]
            out = await self.run_model(inputs)
            if "error" in out:
                return out["error"]
            p = out["predictions"]
            return f"{out['model']} says: " + ", ".join(f"**{r['label']}**" for r in p[:5])
        if any(w in low for w in ("my models", "list models", "what models", "deployed")):
            out = await self.list_models()
            return f"You have {len(out['models'])} live model(s)." if out["models"] else "No models yet. Describe a workflow and I'll build one."
        if any(w in low for w in ("status", "how's it going", "progress", "results")):
            out = await self.lab_status()
            return out.get("error") or f"{out['name']} is {out['status']}."
        if match_template(m) or self.attachment or any(w in low for w in ("build", "train", "fine-tune", "finetune", "model", "classif")):
            await self.build_model(m)
            return "On it. The agent team is planning the task and generating data. Watch it train below."
        return ("I build cheap, specialized models for financial workflows. Try: \"Build a model that categorizes card "
                "transactions into GL codes\" or \"classify: UBER *TRIP SF $23.10\".")
