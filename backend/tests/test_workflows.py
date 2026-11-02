"""A file in, every row classified, results back out: the workflow runner through the API."""
import io
import time

import pytest
from openpyxl import load_workbook

from app.api.workflows import router
from app.datasets.templates import TEMPLATES, generate
from app.main import app
from conftest import make_id_token, wait_for_lab

if not any(getattr(r, "path", "").startswith("/api/workflows") for r in app.routes):
    app.include_router(router)

UNLABELED = "text\n" + "\n".join(r["text"].replace("\n", " ") for r in generate(TEMPLATES["fraud_detection"], 12, seed=4))


def wait_wf(client, wf_id, timeout=400):
    deadline = time.time() + timeout
    while time.time() < deadline:
        w = client.get(f"/api/workflows/{wf_id}").json()
        if w["status"] in ("completed", "failed"):
            return w
        time.sleep(1)
    raise AssertionError(f"workflow {wf_id} did not finish")


@pytest.fixture(scope="module")
def fraud(signed_in):
    r = signed_in.post("/api/labs", json={"prompt": "Screen transactions for fraud and flag anything suspicious"})
    assert r.status_code == 200, r.text
    lab = wait_for_lab(signed_in, r.json()["id"])
    assert lab["status"] == "completed", lab["error"]
    return lab


def test_spreadsheet_rows_run_through_the_chosen_model(signed_in, fraud):
    slug = fraud["result"]["deployment"]["slug"]
    r = signed_in.post("/api/workflows", files={"file": ("october.csv", UNLABELED.encode(), "text/csv")},
                       data={"deployment": slug, "confidence_floor": "0.9", "prompt": "flag the risky ones"})
    assert r.status_code == 200, r.text
    w = wait_wf(signed_in, r.json()["id"])
    assert w["status"] == "completed", w["error"]
    assert w["rows_total"] == 12 and w["rows_done"] == 12 and w["deployment"]["slug"] == slug
    assert w["summary"]["model"]["slug"] == slug and set(w["summary"]["distribution"]) <= {"fraud", "legitimate"}
    assert w["summary"]["cost_usd"] >= 0 and w["summary"]["auto"] + w["summary"]["review"] == 12
    rows = signed_in.get(f"/api/workflows/{w['id']}/rows").json()
    assert len(rows) == 12 and all(x["label"] in ("fraud", "legitimate") for x in rows)
    review = signed_in.get(f"/api/workflows/{w['id']}/rows", params={"review": "true"}).json()
    assert len(review) == w["rows_review"] and all(x["needs_review"] for x in review)

    xlsx = signed_in.get(f"/api/workflows/{w['id']}/export")
    assert xlsx.status_code == 200 and "spreadsheetml" in xlsx.headers["content-type"]
    ws = load_workbook(io.BytesIO(xlsx.content)).active
    assert [c.value for c in ws[1]][-4:] == ["prediction", "confidence", "needs_review", "runner_up"] and ws.max_row == 13
    csv = signed_in.get(f"/api/workflows/{w['id']}/export", params={"format": "csv"})
    assert csv.status_code == 200 and csv.text.splitlines()[0].endswith("prediction,confidence,needs_review,runner_up")

    with signed_in.stream("GET", f"/api/workflows/{w['id']}/events") as s:
        body = "".join(s.iter_text())
    assert '"stage": "done"' in body and '"stage": "parse"' in body


def test_labeled_upload_picks_the_model_from_the_prompt_and_scores_it(signed_in, fraud):
    rows = generate(TEMPLATES["fraud_detection"], 30, seed=11)
    csv = "text,label\n" + "\n".join(f"\"{r['text']}\",{r['label']}" for r in rows)
    r = signed_in.post("/api/workflows", files={"file": ("labeled.csv", csv.encode(), "text/csv")},
                       data={"prompt": "screen these card transactions for fraud"})
    assert r.status_code == 200, r.text
    w = wait_wf(signed_in, r.json()["id"])
    assert w["status"] == "completed", w["error"]
    assert w["deployment"]["slug"] == fraud["result"]["deployment"]["slug"], "should reuse the fraud model, not build a new one"
    assert w["lab_id"] is None and w["summary"]["labeled_rows"] == 30 and w["summary"]["accuracy_on_labeled"] > 0.5
    first = signed_in.get(f"/api/workflows/{w['id']}/rows", params={"limit": "1"}).json()[0]
    assert first["truth"] in ("fraud", "legitimate")


def test_review_decision_becomes_feedback(signed_in, fraud):
    w = signed_in.get("/api/workflows").json()[0]
    slug = w["deployment"]["slug"]
    row = signed_in.get(f"/api/workflows/{w['id']}/rows", params={"limit": "1"}).json()[0]
    other = "fraud" if row["label"] == "legitimate" else "legitimate"
    before = signed_in.get(f"/api/deployments/{slug}").json()["feedback_count"]
    assert signed_in.post(f"/api/workflows/{w['id']}/rows/{row['id']}/review", json={"label": "nope"}).status_code == 422
    r = signed_in.post(f"/api/workflows/{w['id']}/rows/{row['id']}/review", json={"label": other})
    assert r.status_code == 200 and r.json()["reviewed_label"] == other
    assert signed_in.get(f"/api/deployments/{slug}").json()["feedback_count"] == before + 1
    exported = signed_in.get(f"/api/workflows/{w['id']}/export", params={"format": "csv"}).text.splitlines()
    assert any(other in line for line in exported[1:2])


def test_other_users_cannot_see_workflows_and_hopeless_requests_fail_clearly(client):
    assert client.post("/api/auth/firebase", json={"id_token": make_id_token(sub="g-wf2", email="wf2@example.com", name="WF Two")}).status_code == 200
    assert client.get("/api/workflows").json() == []
    r = client.post("/api/workflows", data={"text": "hello there\nanother line", "prompt": "do magic"})
    assert r.status_code == 200
    w = wait_wf(client, r.json()["id"], timeout=60)
    assert w["status"] == "failed" and "No deployed model" in w["error"]
    assert client.post("/api/workflows", data={"prompt": "nothing attached"}).status_code == 422
    assert client.post("/api/auth/firebase", json={"id_token": make_id_token()}).status_code == 200


def test_template_prompt_builds_a_model_when_none_fits(client):
    assert client.post("/api/auth/firebase", json={"id_token": make_id_token(sub="g-wf3", email="wf3@example.com", name="WF Three")}).status_code == 200
    invoices = "text\n" + "\n".join(r["text"] for r in generate(TEMPLATES["invoice_triage"], 8, seed=2))
    r = client.post("/api/workflows", files={"file": ("invoices.csv", invoices.encode(), "text/csv")},
                    data={"prompt": "Triage these incoming vendor invoices"})
    assert r.status_code == 200, r.text
    w = wait_wf(client, r.json()["id"])
    assert w["status"] == "completed", w["error"]
    assert w["lab_id"] and w["deployment"]["name"] == "AP Invoice Triage" and w["rows_done"] == 8
    assert client.post("/api/auth/firebase", json={"id_token": make_id_token()}).status_code == 200


def test_model_choice_respects_the_labels_in_the_file(signed_in, fraud):
    from sqlmodel import Session

    from app.db import Candidate, Deployment, Lab, engine
    from app.workflows import pick_deployment

    me = signed_in.get("/api/auth/me").json()["id"]
    with Session(engine) as s:
        lab = Lab(user_id=me, prompt="vendor types", name="Vendor types", status="completed",
                  spec={"labels": ["software", "travel", "meals"], "description": "merchant categories"})
        s.add(lab); s.commit(); s.refresh(lab)
        cand = Candidate(lab_id=lab.id, strategy="naive_bayes", display_name="NB", family="classical", status="done")
        s.add(cand); s.commit(); s.refresh(cand)
        draft = Deployment(user_id=me, lab_id=lab.id, candidate_id=cand.id, slug="vendor-types-test", name="Vendor types", status="draft")
        s.add(draft); s.commit()

        chosen, why = pick_deployment(s, me, "flag risky transactions", set())
        assert chosen and chosen.name == "Fraud Screener", why
        chosen, why = pick_deployment(s, me, "classify these transactions", {"software", "travel"})
        assert chosen and chosen.slug == "vendor-types-test" and "not published" in why
        chosen, why = pick_deployment(s, me, "flag fraud in these transactions", {"approve", "reject"})
        assert chosen is None and "labels" in why

        s.delete(draft); s.delete(cand); s.delete(lab); s.commit()
