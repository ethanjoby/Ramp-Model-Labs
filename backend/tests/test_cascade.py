import random

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.trainers import cascade
from app.trainers.classical import _features


def rows():
    r = random.Random(0)
    out = []
    for i in range(120):
        lab = ["software", "travel", "meals"][i % 3]
        word = {"software": "github", "travel": "delta", "meals": "chipotle"}[lab]
        out.append({"text": f"{word} charge {r.randint(10, 999)} ref{i}", "label": lab})
    return out


def test_rules_are_pure_and_supported():
    rules = cascade.learn_rules(rows())
    assert rules.get("github") == "software" and rules.get("delta") == "travel"
    assert "charge" not in rules  # appears under every label, so never a rule


def test_route_prefers_rules_then_model_then_review():
    train = rows()
    pipe = Pipeline([("f", _features(len(train))), ("clf", LogisticRegression(max_iter=1000))]).fit(
        [r["text"] for r in train], [r["label"] for r in train])
    model = cascade.CascadeModel(cascade.learn_rules(train), pipe, threshold=0.99, llm_system=None, llm_model=None,
                                 labels=["software", "travel", "meals"])
    out = model.route(["github charge 12 ref999", "zzz unknown thing"])
    assert out[0]["route"] == "rule" and out[0]["label"] == "software" and out[0]["confidence"] == 1.0
    assert out[1]["route"] in ("model", "review") and out[1]["label"] in model.labels
    assert out[1]["route"] == "review"  # a 0.99 floor sends an unknown string to a person


def test_threshold_is_lowest_that_clears_target():
    train = rows()
    pipe = Pipeline([("f", _features(len(train))), ("clf", LogisticRegression(max_iter=1000))]).fit(
        [r["text"] for r in train], [r["label"] for r in train])
    th = cascade._pick_threshold(pipe, train[:30], target=0.85)
    assert th in cascade.THRESHOLDS and th == cascade.THRESHOLDS[0]
