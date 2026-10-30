"""Judge: picks the cheapest model that actually meets the bar, and explains why."""
from __future__ import annotations

from typing import Any

from .. import llm
from ..trainers.base import TaskSpec

TOLERANCE = {"cost": 0.02, "balanced": 0.015, "latency": 0.02, "accuracy": 0.0}


def pareto(cands: list[dict]) -> list[str]:
    front = []
    for c in cands:
        dominated = any(
            o is not c and o["metrics"]["accuracy"] >= c["metrics"]["accuracy"]
            and o["metrics"]["cost_per_1k"] <= c["metrics"]["cost_per_1k"]
            and (o["metrics"]["accuracy"] > c["metrics"]["accuracy"] or o["metrics"]["cost_per_1k"] < c["metrics"]["cost_per_1k"])
            for o in cands
        )
        if not dominated:
            front.append(c["key"])
    return front


def frontier_cost_per_1k(spec: TaskSpec, test: list[dict]) -> float:
    """What a frontier model (Sonnet 5.5) would cost per 1k calls on this traffic."""
    avg_chars = sum(len(r["text"]) for r in test) / max(1, len(test))
    prompt_chars = 400 + sum(len(l) + 40 for l in spec.labels)
    in_tok = (avg_chars + prompt_chars) / 3.6
    return round(llm.cost_of("claude-sonnet-5-5", int(in_tok * 1000), 8 * 1000), 4)


def choose(spec: TaskSpec, cands: list[dict]) -> tuple[dict | None, list[str], str]:
    ok = [c for c in cands if c.get("metrics")]
    if not ok:
        return None, [], "No candidate finished."
    meets = [c for c in ok if c["metrics"]["accuracy"] >= spec.target_accuracy
             and c["metrics"]["latency_p50_ms"] <= spec.max_latency_ms]
    pool = meets or ok
    best_acc = max(c["metrics"]["accuracy"] for c in pool)
    tol = TOLERANCE.get(spec.priority, 0.015)
    near = [c for c in pool if c["metrics"]["accuracy"] >= best_acc - tol]
    if spec.priority == "latency":
        key = lambda c: (c["metrics"]["latency_p50_ms"], -c["metrics"]["accuracy"])  # noqa: E731
    elif spec.priority == "accuracy":
        key = lambda c: (-c["metrics"]["accuracy"], c["metrics"]["cost_per_1k"])  # noqa: E731
    else:
        # Costs within 2x of the cheapest are effectively tied: take the most accurate, then the smallest.
        floor = min(c["metrics"]["cost_per_1k"] for c in near)
        near = [c for c in near if c["metrics"]["cost_per_1k"] <= max(floor * 2, floor + 1e-6)]
        key = lambda c: (-c["metrics"]["accuracy"], c["metrics"].get("size_kb") or 0)  # noqa: E731
    winner = sorted(near, key=key)[0]
    why = (f"{len(meets)} of {len(ok)} candidates met the bar (≥{spec.target_accuracy:.0%} accuracy, "
           f"≤{spec.max_latency_ms:.0f}ms p50)." if meets else
           f"No candidate hit the {spec.target_accuracy:.0%} target, so the agent picked the strongest available.")
    return winner, pareto(ok), why


async def explain(spec: TaskSpec, winner: dict, cands: list[dict], why: str, savings: dict, usage: llm.Usage) -> str:
    m = winner["metrics"]
    fallback = (f"{why} {winner['name']} scored {m['accuracy']:.1%} on held-out data at "
                f"${m['cost_per_1k']:.5f} per 1k predictions and {m['latency_p50_ms']:.1f}ms p50, "
                f"{savings['multiple']:,.0f}x cheaper than running a frontier model on every request.")
    if not llm.available():
        return fallback
    rows = "\n".join(f"- {c['name']} ({c['family']}): acc {c['metrics']['accuracy']:.3f}, f1 {c['metrics']['macro_f1']:.3f}, "
                     f"${c['metrics']['cost_per_1k']}/1k, p50 {c['metrics']['latency_p50_ms']}ms"
                     for c in cands if c.get("metrics"))
    try:
        return (await llm.complete(
            f"Task: {spec.description}\nPriority: {spec.priority}. Target accuracy {spec.target_accuracy}.\n"
            f"Candidates:\n{rows}\n\nSelected: {winner['name']}. Context: {why} Frontier-model cost per 1k: "
            f"${savings['frontier_cost_per_1k']}.\n\nIn 2-3 crisp sentences for a finance-ops engineer, explain why this "
            "model was chosen and the trade-off versus the runner-up. No preamble, no markdown.",
            max_tokens=300, usage=usage)).strip() or fallback
    except Exception:
        return fallback
