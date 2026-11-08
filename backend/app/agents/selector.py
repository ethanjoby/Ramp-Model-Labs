"""Judge: picks the cheapest model that actually meets the bar, says whether anyone did, and explains why."""
from __future__ import annotations

import math
from typing import Any

from .. import llm
from ..trainers.base import TaskSpec

SONNET_CACHE_READ_PER_MTOK = 0.20


def wilson(acc: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; on 18 rows a 61% score really means 'somewhere between 39% and 80%'."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


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
    """What the frontier yardstick would cost per 1k calls on this traffic, with the instruction served from the prompt cache."""
    model = llm.reference_model()
    avg_chars = sum(len(r["text"]) for r in test) / max(1, len(test))
    prompt_chars = 400 + sum(len(l) + 40 for l in spec.labels)
    fresh_tok, cached_tok = avg_chars / 3.6, prompt_chars / 3.6
    per_call = llm.cost_of(model, int(fresh_tok), 8) + cached_tok * llm.cached_input_price(model) / 1_000_000
    return round(per_call * 1000, 4)


def _interval(c: dict) -> tuple[float, float]:
    m = c["metrics"]
    ci = m.get("accuracy_ci")
    return tuple(ci) if ci else wilson(m["accuracy"], m.get("n_eval", 0))


def choose(spec: TaskSpec, cands: list[dict], imbalance: float | None = None) -> tuple[dict | None, list[str], str]:
    """Returns (winner, pareto_front, why). winner["meets_bar"] says whether the winner actually cleared the bar;
    with it False the orchestrator deploys as a draft rather than publishing."""
    ok = [c for c in cands if c.get("metrics")]
    if not ok:
        return None, [], "No candidate finished."
    strict = (imbalance or 1) > 3 or any(
        v == 0 for c in ok for v in (c["metrics"].get("per_label_f1") or {}).values() if v is not None)

    def meets_bar(c: dict) -> bool:
        m = c["metrics"]
        if m["accuracy"] < spec.target_accuracy or m["latency_p50_ms"] > spec.max_latency_ms:
            return False
        # Skewed labels let a majority-class model clear raw accuracy; demand macro F1 near the bar too.
        return not strict or m.get("macro_f1", 0) >= spec.target_accuracy - 0.05

    meets = [c for c in ok if meets_bar(c)]
    pool = meets or ok
    best = max(pool, key=lambda c: c["metrics"]["accuracy"])
    best_lo, _ = _interval(best)
    # Scores whose intervals overlap the best one's are ties on a test set this size; fixed tolerances pretend otherwise.
    near = [c for c in pool if _interval(c)[1] >= best_lo] or [best]
    if spec.priority == "latency":
        key = lambda c: (c["metrics"]["latency_p50_ms"], -c["metrics"]["accuracy"])  # noqa: E731
    elif spec.priority == "accuracy":
        key = lambda c: (-c["metrics"]["accuracy"], c["metrics"]["cost_per_1k"])  # noqa: E731
    else:
        # Costs within 2x of the cheapest are effectively tied: take the most accurate, then the smallest.
        floor = min(c["metrics"]["cost_per_1k"] for c in near)
        near = [c for c in near if c["metrics"]["cost_per_1k"] <= max(floor * 2, floor + 1e-6)]
        # Then the one that keeps its answers under noisy copies of the same rows, then the smallest.
        key = lambda c: (-c["metrics"]["accuracy"], -(c["metrics"].get("robust_accuracy") or 0), c["metrics"].get("size_kb") or 0)  # noqa: E731
    winner = sorted(near, key=key)[0]
    winner["meets_bar"] = winner in meets
    n = winner["metrics"].get("n_eval", 0)
    lo, hi = _interval(winner)
    bar = f"≥{spec.target_accuracy:.0%} accuracy" + (f", macro F1 ≥{spec.target_accuracy - 0.05:.0%}" if strict else "") + f", ≤{spec.max_latency_ms:.0f}ms p50"
    why = (f"{len(meets)} of {len(ok)} candidates met the bar ({bar}) on {n} held-out rows (95% interval {lo:.0%}-{hi:.0%})."
           if meets else
           f"No candidate hit the bar ({bar}) on {n} held-out rows, so the strongest available is trained but not published.")
    return winner, pareto(ok), why


async def explain(spec: TaskSpec, winner: dict, cands: list[dict], why: str, savings: dict, usage: llm.Usage) -> str:
    m = winner["metrics"]
    est = "" if savings.get("frontier_measured") else " (est. from list prices; add an Anthropic key to measure it)"
    fallback = (f"{why} {winner['name']} scored {m['accuracy']:.1%} on held-out data at "
                f"${m['cost_per_1k']:.5f} per 1k predictions and {m['latency_p50_ms']:.1f}ms p50, "
                f"{savings['multiple']:,.0f}x cheaper than running a frontier model on every request{est}.")
    if not llm.available():
        return fallback
    rows = "\n".join(f"- {c['name']} ({c['family']}): acc {c['metrics']['accuracy']:.3f}, f1 {c['metrics']['macro_f1']:.3f}, "
                     f"${c['metrics']['cost_per_1k']}/1k, p50 {c['metrics']['latency_p50_ms']}ms"
                     for c in cands if c.get("metrics"))
    try:
        return (await llm.complete(
            f"Task: {spec.description}\nPriority: {spec.priority}. Target accuracy {spec.target_accuracy}.\n"
            f"Candidates:\n{rows}\n\nSelected: {winner['name']}. Context: {why} Frontier-model cost per 1k: "
            f"${savings['frontier_cost_per_1k']} ({'measured' if savings.get('frontier_measured') else 'estimated from list prices'}).\n\n"
            "In 2-3 crisp sentences for a finance-ops engineer, explain why this model was chosen and the trade-off versus the "
            "runner-up. If the frontier cost is an estimate, say 'est.' once. No preamble, no markdown.",
            max_tokens=600, usage=usage, purpose="plan")).strip() or fallback
    except Exception:
        return fallback
