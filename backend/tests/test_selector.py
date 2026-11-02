from app.agents import selector
from app.trainers.base import TaskSpec


def cand(key, acc, cost, lat=1.0, size=100, n=200):
    return {"key": key, "id": key, "name": key, "family": "classical",
            "metrics": {"accuracy": acc, "macro_f1": acc, "per_label_f1": {}, "n_eval": n, "cost_per_1k": cost,
                        "latency_p50_ms": lat, "size_kb": size}}


def spec(priority="balanced", target=0.85, max_latency=1500):
    return TaskSpec(name="t", task_type="classification", description="d", labels=["a", "b"],
                    priority=priority, target_accuracy=target, max_latency_ms=max_latency)


def test_pareto_front_excludes_dominated():
    front = selector.pareto([cand("cheap_ok", 0.90, 0.001), cand("pricey_same", 0.90, 0.01), cand("best", 0.95, 0.5)])
    assert set(front) == {"cheap_ok", "best"}


def test_balanced_picks_cheapest_within_tolerance():
    w, _, why = selector.choose(spec(), [cand("a", 0.96, 1.0), cand("b", 0.95, 0.001), cand("c", 0.80, 0.0001)])
    assert w["key"] == "b" and "2 of 3" in why


def test_accuracy_priority_ignores_cost():
    w, _, _ = selector.choose(spec("accuracy"), [cand("a", 0.96, 1.0), cand("b", 0.95, 0.001)])
    assert w["key"] == "a"


def test_latency_priority_and_budget():
    w, _, _ = selector.choose(spec("latency", max_latency=10), [cand("slow", 0.99, 0.001, lat=50), cand("fast", 0.90, 0.01, lat=2)])
    assert w["key"] == "fast"


def test_nothing_meets_bar_still_returns_strongest():
    w, front, why = selector.choose(spec(target=0.99), [cand("a", 0.90, 1.0), cand("b", 0.78, 0.001)])
    assert w["key"] == "a" and "No candidate hit" in why and set(front) == {"a", "b"}
    assert w["meets_bar"] is False


def test_overlapping_intervals_tie_and_tiny_test_sets_cannot_separate_candidates():
    # 0.90 vs 0.78 on 200 rows are different; on 18 rows the intervals overlap and the cheaper one wins.
    w, _, _ = selector.choose(spec(), [cand("a", 0.95, 1.0, n=18), cand("b", 0.86, 0.001, n=18)])
    assert w["key"] == "b"
    w, _, _ = selector.choose(spec(), [cand("a", 0.95, 1.0, n=200), cand("b", 0.86, 0.001, n=200)])
    assert w["key"] == "a" and w["meets_bar"] is True


def test_imbalance_demands_macro_f1_too():
    majority = cand("maj", 0.92, 0.001)
    majority["metrics"]["macro_f1"] = 0.48
    fair = cand("fair", 0.90, 0.01)
    w, _, why = selector.choose(spec(), [majority, fair], imbalance=6.0)
    assert w["key"] == "fair" and "macro F1" in why
