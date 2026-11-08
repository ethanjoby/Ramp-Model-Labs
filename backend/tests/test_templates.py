"""The procedural generators keep two pools apart: what a model trains on and what it is scored on."""
import re

import pytest

from app.datasets import templates as T
from app.datasets.templates import TEMPLATES, generate

PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def _patterns(entities: set[str]) -> list[re.Pattern]:
    """One boundary-aware, case-insensitive pattern per entity. A phrasing with placeholders is matched by its
    longest literal fragment (short fragments are too generic to mean anything)."""
    out = []
    for e in entities:
        lit = max((f.strip(" #:;,.-?!'") for f in PLACEHOLDER.split(e)), key=len) if "{" in e else e
        if len(lit) >= 8 or "{" not in e:
            out.append(re.compile(rf"(?<![A-Za-z0-9]){re.escape(lit)}(?![A-Za-z0-9])", re.I))
    return out


@pytest.mark.parametrize("key", list(TEMPLATES))
def test_pools_are_disjoint_and_both_reach_every_label(key):
    t = TEMPLATES[key]
    train, ev = t.entities("train"), t.entities("eval")
    assert train and ev and not (train & ev), key
    for src in t.entity_sources:  # every list contributes to both pools, so no label is reachable in one only
        assert T._pool(src, "train") and T._pool(src, "eval")
    for pool in T.POOLS:
        assert {r["label"] for r in generate(t, 600, seed=3, pool=pool)} == set(t.labels), (key, pool)


@pytest.mark.parametrize("key", list(TEMPLATES))
def test_no_train_entity_is_hidden_inside_an_eval_entity(key):
    t = TEMPLATES[key]
    pats = _patterns(t.entities("train"))
    plain_eval = [e for e in t.entities("eval") if "{" not in e]
    hits = [(p.pattern, e) for e in plain_eval for p in pats if p.search(e)]
    assert not hits, hits


@pytest.mark.parametrize("key", list(TEMPLATES))
def test_eval_rows_never_contain_a_train_entity(key):
    t = TEMPLATES[key]
    pats = _patterns(t.entities("train"))
    rows = generate(t, 600, seed=1, pool="eval") + generate(t, 600, seed=2, pool="eval", noise=False)
    leaks = [(r["text"], p.pattern) for r in rows for p in pats if p.search(r["text"])]
    assert not leaks, leaks[:5]
    # and the check is not vacuous: eval entities do show up in eval rows
    ev = _patterns({e for e in t.entities("eval") if "{" not in e})
    assert sum(1 for r in rows if any(p.search(r["text"]) for p in ev)) > len(rows) // 2


@pytest.mark.parametrize("key", list(TEMPLATES))
@pytest.mark.parametrize("pool", T.POOLS)
def test_noise_only_flips_labels_in_both_pools(key, pool):
    t = TEMPLATES[key]
    noisy = generate(t, 800, seed=11, noise=True, pool=pool)
    clean = generate(t, 800, seed=11, noise=False, pool=pool)
    assert [r["text"] for r in noisy] == [r["text"] for r in clean]
    rate = sum(a["label"] != b["label"] for a, b in zip(noisy, clean)) / len(noisy)
    assert 0.01 < rate < 0.08, (key, pool, rate)
    assert {r["label"] for r in noisy} <= set(t.labels)


@pytest.mark.parametrize("key", list(TEMPLATES))
@pytest.mark.parametrize("pool", T.POOLS)
def test_each_pool_yields_enough_unique_rows(key, pool):
    rows = generate(TEMPLATES[key], 300, seed=1, pool=pool)
    assert len(rows) == 300 and len({r["text"] for r in rows}) == 300


def test_pool_split_rule_and_bad_pool():
    assert T._pool(list("abcdefghij"), "eval") == ["e", "j"]
    assert T._pool(list("abcdefghij"), "train") == list("abcdfghi")
    assert T._pool(["x", "y", "z"], "eval") == ["z"] and T._pool(["x", "y", "z"], "train") == ["x", "y"]
    with pytest.raises(ValueError):
        generate(TEMPLATES["fraud_detection"], 5, pool="test")


def test_hard_negatives_are_coded_the_way_a_controller_would():
    m = T.EXPENSE_MERCHANTS
    assert "UBER *EATS" in m["Meals & Entertainment"][0] and "DD *DOORDASH" in m["Meals & Entertainment"][0]
    assert not any("EATS" in x for x in m["Ground Transportation"][0]) and "UBER *TRIP" in m["Ground Transportation"][0]
    assert "APPLE.COM/BILL" in m["Software & SaaS"][0] and "APPLE STORE" in m["Office Supplies"][0]
    assert "AMZN MKTP US" in m["Office Supplies"][0] and "AMAZON KINDLE" in m["Training & Education"][0]
    assert all(len(ms) >= 15 and len(T._pool(ms, "eval")) >= 3 for ms, _ in m.values())
    assert all(len(ps) == 10 and all("{" in ps[i] for i in (4, 9)) for ps in T.SUPPORT_INTENTS.values())
