"""Teacher labels: majority vote across framings, disagreements dropped, train-only."""
import asyncio

from app import llm
from app.agents import data, teacher
from app.trainers.base import TaskSpec


def spec():
    return TaskSpec(name="Vendor types", task_type="classification", description="Classify card merchants.",
                    labels=["software", "travel", "meals"], label_descriptions={"software": "SaaS and tooling"},
                    input_description="a card merchant string")


def test_three_framings_vote_and_disagreements_are_dropped(monkeypatch):
    calls = []

    async def fake_batch(system, texts, labels, model, usage=None):
        calls.append(system)
        # framing 1 and 2 agree on every row; framing 3 disagrees on the second row, so only agreement decides
        third = "examples" in system.lower() or "closest" in system.lower()
        out = []
        for t in texts:
            if "GITHUB" in t:
                out.append("software")
            elif "DELTA" in t:
                out.append("meals" if third else "travel")
            else:
                out.append(None)  # the model named no allowed label for this one
        return out

    monkeypatch.setattr(teacher, "classify_batch", fake_batch)
    kept, stats = asyncio.run(teacher.label_rows(spec(), ["GITHUB.COM $4", "DELTA AIR $400", "???", "github.com $4"], llm.Usage()))
    assert len(calls) == 3 and len({c for c in calls}) == 3, "three distinct framings, one request each for a small batch"
    assert [k["label"] for k in kept] == ["software", "travel"] and all(k["source"] == "teacher" for k in kept)
    assert kept[0]["agreement"] == 1.0 and kept[1]["agreement"] == round(2 / 3, 2)
    assert stats == {**stats, "asked": 3, "kept": 2, "dropped": 1, "framings": 3} and stats["skipped"] == 0


def test_curator_trains_on_teacher_rows_but_never_tests_on_them(monkeypatch):
    async def emit(*a, **k):
        pass

    async def fake_label(spec_, texts, usage, examples=None, max_rows=None):
        return [{"text": t, "label": "travel", "source": "teacher", "agreement": 1.0} for t in texts], \
               {"asked": len(texts), "skipped": 0, "kept": len(texts), "dropped": 0, "cost": 0.01, "model": "x", "framings": 3}

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(teacher, "label_rows", fake_label)

    async def no_synth(*a, **k):
        return []

    monkeypatch.setattr(data, "_synth_label", no_synth)
    labeled = [{"text": f"GITHUB.COM #{i} $5.00", "label": "software"} for i in range(40)] + \
              [{"text": f"SWEETGREEN #{i} $14.00", "label": "meals"} for i in range(40)] + \
              [{"text": f"DELTA AIR #{i} $400.00", "label": "travel"} for i in range(40)]
    unlabeled = [{"text": f"UNITED AIRLINES #{i} $350.00", "label": None} for i in range(30)]
    out = asyncio.run(data.curate(spec(), labeled + unlabeled, emit, llm.Usage()))
    st = out["stats"]
    assert st["teacher"]["kept"] == 30 and st["ignored"]["unlabeled"] == 0
    assert sum(1 for r in out["train"] if r["source"] == "teacher") == 30
    assert not any(r["source"] == "teacher" for r in out["test"] + out["dev"])
    assert st["eval_source"].startswith("your real examples")
