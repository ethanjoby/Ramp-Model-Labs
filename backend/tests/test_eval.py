"""Scoring: slices, robustness, calibration and the automation rate; the perturbations that feed them."""
import random

from app.datasets.perturb import SAFE_OPS, perturb, perturb_rows
from app.trainers.base import Result, TaskSpec, automation_rate, score

SPEC = TaskSpec(name="t", task_type="classification", description="d", labels=["a", "b"])


def _rows():
    return [{"text": "UBER *TRIP SAN FRANCISCO CA | $23.10 | memo: to airport", "label": "a"},
            {"text": "SLACK TECHNOLOGIES | $1,200.00 | memo: annual plan", "label": "b"},
            {"text": "LUFTHANSA | $980.00", "label": "a", "slice": "unseen"},
            {"text": "DATADOG | $2,400.00 | memo: monthly bill", "label": "b", "slice": "unseen"}]


def test_perturb_changes_text_keeps_amount_magnitude_and_is_deterministic():
    rng1, rng2 = random.Random(7), random.Random(7)
    t = "SLACK TECHNOLOGIES #4411 | $1,200.00 | memo: annual plan"
    a, b = perturb(t, rng1, strength=2), perturb(t, rng2, strength=2)
    assert a == b and a != t
    for op in SAFE_OPS:
        out = op(t, random.Random(1))
        assert "1200" in out.replace(",", "") or "1,200" in out, (op.__name__, out)


def test_perturb_rows_tags_slice_and_origin_and_never_duplicates():
    rows = [{"text": f"VENDOR {i} INV-{1000 + i} ${i * 3}.50", "label": "a"} for i in range(30)]
    out = perturb_rows(rows, random.Random(3), 20)
    assert len(out) == 20
    assert all(o["slice"] == "perturbed" and o["label"] == "a" and o["origin"] in {r["text"] for r in rows} for o in out)
    assert len({o["text"] for o in out} | {r["text"] for r in rows}) == 50


def test_score_separates_headline_from_perturbed_copies():
    test = _rows() + perturb_rows(_rows()[:2], random.Random(1), 2)
    assert len(test) == 6
    # right on every original row, wrong on both perturbed copies: headline stays 100%, robustness shows the fragility
    preds = [r["label"] for r in test[:4]] + [("b" if r["label"] == "a" else "a") for r in test[4:]]
    m = score(SPEC, test, Result(preds, [1.0] * 6, 0.001, confidences=[0.9, 0.8, 0.7, 0.95, 0.6, 0.6]))
    assert m["accuracy"] == 1.0 and m["n_eval"] == 4
    assert m["slices"]["core"] == {"accuracy": 1.0, "n": 2} and m["slices"]["unseen"] == {"accuracy": 1.0, "n": 2}
    assert m["slices"]["perturbed"]["n"] == 2 and m["robust_accuracy"] == 0.0 and m["unseen_accuracy"] == 1.0
    assert m["ece"] is not None and m["auto_rate_at_95"] == 1.0 and m["confidence_floor_95"] == 0.7
    assert m["errors"] == []  # perturbed misses are a slice, not headline errors


def test_score_without_confidences_leaves_calibration_null():
    m = score(SPEC, _rows(), Result(["a", "b", "b", "b"], [1.0] * 4, 0.001))
    assert m["accuracy"] == 0.75 and m["unseen_accuracy"] == 0.5 and m["robust_accuracy"] is None
    assert m["ece"] is None and m["auto_rate_at_95"] is None
    assert m["errors"][0]["slice"] == "unseen"


def test_automation_rate_is_the_largest_confident_prefix_at_target_accuracy():
    conf = [0.99, 0.95, 0.9, 0.85, 0.5, 0.4]
    correct = [True, True, True, True, False, True]
    rate, floor = automation_rate(conf, correct, 0.95)
    assert rate == 4 / 6 and floor == 0.85
    assert automation_rate(conf, [False] * 6, 0.95) == (0.0, None)
