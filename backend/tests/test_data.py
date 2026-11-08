import asyncio
import random

from app.agents import data
from app.datasets.templates import TEMPLATES, generate, match_template
from app.trainers.base import TaskSpec


def test_stratified_keeps_label_proportions():
    rows = [{"text": f"t{i}", "label": "a" if i % 3 else "b"} for i in range(90)]
    take, rest = data._stratified(rows, 30, random.Random(1))
    assert len(take) + len(rest) == 90
    assert sum(r["label"] == "b" for r in take) == 10
    assert sum(r["label"] == "a" for r in take) == 20


def test_augment_preserves_labels_and_never_duplicates():
    rows = [{"text": f"VENDOR {i} #{1000 + i} ${i}.50", "label": "x" if i % 2 else "y", "source": "user"} for i in range(20)]
    extra = data.augment(rows, random.Random(3), target=40)
    assert 0 < len(extra) <= 40
    originals = {r["text"] for r in rows}
    assert all(e["text"] not in originals for e in extra)
    assert len({e["text"] for e in extra}) == len(extra)
    assert all(e["label"] in ("x", "y") and e["source"] == "augmented" for e in extra)


def _spec(labels, template=None):
    return TaskSpec(name="t", task_type="classification", description="d", labels=labels, template=template)


def test_curate_small_upload_tests_on_real_rows_only_and_augments_training():
    merchants = {"software": ["GITHUB", "FIGMA", "NOTION", "ZOOM"], "travel": ["DELTA", "UNITED", "MARRIOTT", "HERTZ"],
                 "meals": ["CHIPOTLE", "SWEETGREEN", "STARBUCKS", "DOORDASH"]}
    seeds = [{"text": f"{m} #{i} ${i * 7 % 300}.00", "label": lab} for lab, ms in merchants.items() for i, m in enumerate(ms * 5)]
    logs = []

    async def emit(agent, kind, message="", payload=None):
        logs.append(message)

    out = asyncio.run(data.curate(_spec(list(merchants)), seeds, emit, None))
    stats = out["stats"]
    assert stats["eval_source"].startswith("your real examples") and "perturbed copies" in stats["eval_source"]
    assert stats["test"] == int(len(seeds) * 0.3)
    headline = [r for r in out["test"] if r.get("slice", "core") != "perturbed"]
    assert len(headline) == stats["test"] and all(r["source"] == "user" for r in headline)
    assert stats["slices"]["perturbed"] > 0 and all(r["source"] == "user" for r in out["dev"])
    test_texts = {r["text"] for r in out["test"]}
    assert not test_texts & {r["text"] for r in out["train"]}, "a test row leaked into training"
    assert stats["sources"].get("augmented", 0) > 0 and any("augmenter" in m for m in logs)


def test_curate_template_task_offline():
    async def emit(*a, **k):
        pass

    tmpl = TEMPLATES["fraud_detection"]
    out = asyncio.run(data.curate(_spec(tmpl.labels, template=tmpl.key), [], emit, None))
    st = out["stats"]
    assert st["train"] > 1500 and st["test"] == 200 and st["test_total"] == 260
    assert st["slices"] == {"core": 120, "unseen": 80, "perturbed": 60}
    assert set(r["label"] for r in out["train"]) == set(tmpl.labels)
    # the unseen slice is built from the eval pool only, and nothing from that pool trains
    eval_entities = {e.upper() for e in tmpl.entities("eval")}
    train_text = " || ".join(r["text"].upper() for r in out["train"])
    assert not [e for e in eval_entities if e in train_text]
    assert all(r["source"] == "procedural-unseen" for r in out["test"] if r.get("slice") == "unseen")
    assert all(r["origin"] in {x["text"] for x in out["test"]} for r in out["test"] if r.get("slice") == "perturbed")


def test_generators_stay_within_their_labels_and_templates_match_prompts():
    for t in TEMPLATES.values():
        rows = generate(t, 300, seed=1)
        assert len(rows) == 300 and {r["label"] for r in rows} <= set(t.labels)
        assert len({r["text"] for r in rows}) == 300
    assert match_template("Categorize card transactions into GL codes").key == "expense_categorization"
    assert match_template("Reconcile bank lines against the ledger").key == "transaction_reconciliation"
    assert match_template("Flag suspicious card activity as fraud").key == "fraud_detection"
    assert match_template("Triage incoming vendor invoices").key == "invoice_triage"
    assert match_template("Route support tickets to the right queue").key == "support_routing"
    assert match_template("write me a poem") is None


def test_template_eval_rows_are_clean_and_never_mirror_training_rows():
    async def emit(*a, **k):
        pass

    tmpl = TEMPLATES["fraud_detection"]
    out = asyncio.run(data.curate(_spec(tmpl.labels, template=tmpl.key), [], emit, None))
    st = out["stats"]
    assert st["noise_rate"] is not None and 0.02 < st["noise_rate"] < 0.08
    assert st["eval_source"].startswith("held-out synthetic, no planted noise")
    held = {data._fold(r["text"]) for r in out["test"] + out["dev"]}
    assert not any(data._fold(r["text"]) in held for r in out["train"])


def test_user_rows_are_weighted_up_when_a_template_supplies_the_bulk():
    async def emit(*a, **k):
        pass

    tmpl = TEMPLATES["fraud_detection"]
    seeds = [{"text": r["text"], "label": r["label"]} for r in generate(tmpl, 40, seed=77, noise=False)]
    out = asyncio.run(data.curate(_spec(tmpl.labels, template=tmpl.key), seeds, emit, None))
    st = out["stats"]
    assert st["eval_source"].startswith("your real examples")
    assert st["sources"].get("user-variant", 0) > 0, "rows are weighted up with rewritten variants, not plain copies"
    user_share = sum(1 for r in out["train"] if r["source"].startswith("user")) / st["train"]
    assert user_share >= 0.24


def test_unlabeled_and_off_spec_rows_are_counted_not_swallowed():
    async def emit(*a, **k):
        pass

    tmpl = TEMPLATES["fraud_detection"]
    seeds = [{"text": "x1", "label": None}, {"text": "x2", "label": "banana"}, {"text": "", "label": "fraud"}]
    out = asyncio.run(data.curate(_spec(tmpl.labels, template=tmpl.key), seeds, emit, None))
    assert out["stats"]["ignored"] == {"unlabeled": 1, "off_spec": 1, "empty": 1}
