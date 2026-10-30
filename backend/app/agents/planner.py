"""Task analyst: turns one plain-English prompt into a concrete, testable spec."""
from __future__ import annotations

import re

from .. import llm
from ..config import get_settings
from ..datasets.templates import TEMPLATES, match_template
from ..trainers.base import TaskSpec


class PlanningError(Exception):
    pass


def _constraints(prompt: str) -> dict:
    p = prompt.lower()
    out: dict = {}
    m = re.search(r"(\d{2,3})\s*%", p)
    if m and 50 <= int(m.group(1)) <= 100:
        out["target_accuracy"] = int(m.group(1)) / 100
    m = re.search(r"(\d+)\s*ms", p)
    if m:
        out["max_latency_ms"] = float(m.group(1))
    if any(w in p for w in ("cheap", "cost", "budget", "inexpensive", "low cost")):
        out["priority"] = "cost"
    elif any(w in p for w in ("fast", "real-time", "realtime", "latency", "instant")):
        out["priority"] = "latency"
    elif any(w in p for w in ("accurate", "accuracy", "precise", "best quality")):
        out["priority"] = "accuracy"
    return out


async def plan(prompt: str, seed_examples: list[dict], usage: llm.Usage) -> TaskSpec:
    seed_labels = sorted({r["label"] for r in seed_examples if r.get("label")})
    tmpl = match_template(prompt)
    cons = _constraints(prompt)

    if llm.available():
        catalog = "\n".join(f"- {t.key}: {t.description} Labels: {t.labels}" for t in TEMPLATES.values())
        out = await llm.complete_json(
            "You are the task-analysis agent of an AutoML platform that builds specialized models for financial "
            "workflows. Convert the request into a single-label classification spec.\n\n"
            f"User request: \"\"\"{prompt}\"\"\"\n\n"
            f"Labels present in the user's uploaded examples: {seed_labels or 'none'}\n\n"
            f"Built-in procedural data generators (reuse one ONLY if it truly fits, and then keep its labels):\n{catalog}\n\n"
            "Return JSON: {\"name\": short product-style name (2-4 words), \"description\": one sentence, "
            "\"labels\": [3-15 snake_case or Title Case labels; MUST equal the uploaded labels if any], "
            "\"label_descriptions\": {label: short decision rule}, \"input_description\": what one input looks like, "
            "\"template\": generator key or null, \"target_accuracy\": 0.5-0.99, \"max_latency_ms\": number, "
            "\"priority\": one of cost|accuracy|latency|balanced}",
            model=get_settings().agent_model, max_tokens=1500, usage=usage)
        template = out.get("template") if out.get("template") in TEMPLATES else None
        labels = seed_labels or (TEMPLATES[template].labels if template else out.get("labels") or [])
        if len(labels) < 2:
            raise PlanningError("Couldn't infer at least two output labels from the request.")
        spec = TaskSpec(
            name=out.get("name") or "Custom classifier", task_type="classification",
            description=out.get("description") or prompt[:200], labels=list(labels),
            label_descriptions={k: v for k, v in (out.get("label_descriptions") or {}).items() if k in labels},
            input_description=out.get("input_description", ""), template=template,
            target_accuracy=float(out.get("target_accuracy", 0.85)), max_latency_ms=float(out.get("max_latency_ms", 1500)),
            priority=out.get("priority", "balanced") if out.get("priority") in ("cost", "accuracy", "latency", "balanced") else "balanced",
        )
    elif seed_labels and len(seed_labels) >= 2:
        spec = TaskSpec(name="Custom classifier", task_type="classification", description=prompt[:200],
                        labels=seed_labels, input_description="user-provided examples")
        if tmpl and set(seed_labels) <= set(tmpl.labels):
            spec.template, spec.name = tmpl.key, tmpl.name
    elif tmpl:
        spec = TaskSpec(name=tmpl.name, task_type="classification", description=tmpl.description,
                        labels=tmpl.labels, input_description=tmpl.input_hint, template=tmpl.key)
    else:
        raise PlanningError(
            "Offline mode can build expense categorization, fraud screening, support routing and invoice triage "
            "models, or anything you upload labeled examples for. Add an ANTHROPIC_API_KEY for open-ended tasks.")

    for k, v in cons.items():
        setattr(spec, k, v)
    return spec
