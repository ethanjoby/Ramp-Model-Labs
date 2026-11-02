"""End to end through the FastAPI app: sign in, build a lab offline, call the model, close the feedback loop."""
import shutil

import pytest
from fastapi.testclient import TestClient

from app.main import app

from app.config import DATA_DIR
from conftest import make_id_token, wait_for_lab


def test_unauthenticated_requests_are_rejected():
    with TestClient(app) as anon:  # the shared client may already carry a session from another module
        assert anon.get("/api/labs").status_code == 401
        assert anon.post("/api/labs", json={"prompt": "Screen transactions for fraud"}).status_code == 401
        assert anon.post("/v1/models/nope/predict", json={"inputs": ["x"]}).status_code == 401


def test_firebase_login_rules(client):
    assert client.post("/api/auth/firebase", json={"id_token": make_id_token(verified=False, sub="g-unv")}).status_code == 403
    assert client.post("/api/auth/firebase", json={"id_token": make_id_token(provider="password", sub="g-pw")}).status_code == 403
    assert client.post("/api/auth/firebase", json={"id_token": "x" * 40}).status_code == 401
    r = client.post("/api/auth/firebase", json={"id_token": make_id_token(sub="g-2", email="two@example.com", name="Two")})
    assert r.status_code == 200 and r.json()["provider"] == "google" and "rml_session" in r.cookies
    again = client.post("/api/auth/firebase", json={"id_token": make_id_token(sub="g-2", email="two@example.com", name="Two Renamed")})
    assert again.json()["id"] == r.json()["id"] and again.json()["name"] == "Two Renamed"


@pytest.fixture(scope="module")
def lab(signed_in):
    r = signed_in.post("/api/labs", json={"prompt": "Screen transactions for fraud and flag anything suspicious"})
    assert r.status_code == 200, r.text
    lab = wait_for_lab(signed_in, r.json()["id"])
    yield lab
    shutil.rmtree(DATA_DIR / "models" / lab["id"], ignore_errors=True)


def test_offline_lab_trains_scores_and_deploys(lab):
    assert lab["status"] == "completed", lab["error"]
    assert lab["spec"]["labels"] == ["fraud", "legitimate"]
    done = [c for c in lab["candidates"] if c["status"] == "done"]
    skipped = [c for c in lab["candidates"] if c["status"] == "skipped"]
    assert len(done) >= 5 and len(done) + len(skipped) == len(lab["candidates"]), [c["status"] for c in lab["candidates"]]
    assert not [c for c in lab["candidates"] if c["status"] == "failed"], [c["metrics"] for c in lab["candidates"] if c["status"] == "failed"]
    assert all(c["metrics"]["n_eval"] == 200 for c in done)
    winner = [c for c in done if c["is_winner"]]
    assert len(winner) == 1 and winner[0]["metrics"]["accuracy"] >= 0.85
    assert sum(c["metrics"]["train_seconds"] for c in done) > 0.3, "models did not actually train"
    assert all(c["metrics"]["latency_p50_ms"] > 0 and c["metrics"]["cost_per_1k"] > 0 for c in done)
    assert lab["result"]["savings"]["frontier_measured"] is False
    assert lab["dataset"]["eval_source"].startswith("held-out synthetic")


def test_events_replay_in_order(signed_in, lab):
    with signed_in.stream("GET", f"/api/labs/{lab['id']}/events") as s:
        seqs = []
        for line in s.iter_lines():
            if line.startswith("id: "):
                seqs.append(int(line[4:]))
            if len(seqs) >= 5:
                break
    assert seqs == sorted(seqs) and seqs[0] == 1


def test_playground_public_api_feedback_and_retrain(signed_in, lab):
    slug = lab["result"]["deployment"]["slug"]
    r = signed_in.post(f"/api/deployments/{slug}/predict", json={"inputs": ["card-not-present $2,499.00 at GIFTCARDMALL 03:12; new device; 4 declines"]})
    assert r.status_code == 200 and r.json()["predictions"][0]["label"] in ("fraud", "legitimate")
    assert signed_in.post(f"/api/deployments/{slug}/predict", json={"inputs": []}).status_code == 422

    key = signed_in.post("/api/keys", json={"name": "ci"}).json()
    pub = signed_in.post(f"/v1/models/{slug}/predict", json={"inputs": ["UBER *TRIP"]}, headers={"Authorization": f"Bearer {key['key']}"})
    assert pub.status_code == 200
    assert signed_in.delete(f"/api/keys/{key['id']}").status_code == 200
    revoked = signed_in.post(f"/v1/models/{slug}/predict", json={"inputs": ["UBER *TRIP"]}, headers={"Authorization": f"Bearer {key['key']}"})
    assert revoked.status_code == 401

    assert signed_in.post(f"/api/deployments/{slug}/feedback", json={"text": "x", "predicted": "fraud", "correct": "nope"}).status_code == 422
    assert signed_in.post(f"/api/deployments/{slug}/feedback", json={"text": "contactless $12 at STAPLES", "predicted": "fraud", "correct": "legitimate"}).status_code == 200
    dep = signed_in.get(f"/api/deployments/{slug}").json()
    assert dep["feedback_count"] == 1 and dep["requests"] >= 2


def test_other_users_cannot_see_or_delete_a_lab(client, lab):
    other = make_id_token(sub="g-intruder", email="intruder@example.com", name="Intruder")
    assert client.post("/api/auth/firebase", json={"id_token": other}).status_code == 200
    assert client.get(f"/api/labs/{lab['id']}").status_code == 404
    assert client.delete(f"/api/labs/{lab['id']}").status_code == 404
    assert client.get(f"/api/deployments/{lab['result']['deployment']['slug']}").status_code == 404
    # sign the owner back in for later modules
    assert client.post("/api/auth/firebase", json={"id_token": make_id_token()}).status_code == 200


def test_chat_offline_round_trip(signed_in, lab):
    r = signed_in.post("/api/chat", json={"message": "my models"})
    assert r.status_code == 200 and r.json()["actions"][0]["type"] == "models"
    r = signed_in.post("/api/chat", json={"message": "classify: card-not-present $900 at LUXE WATCHES 03:00; new device"})
    assert r.json()["actions"][0]["type"] == "prediction"
    assert len(signed_in.get("/api/chat").json()) >= 4
    assert signed_in.delete("/api/chat").status_code == 200 and signed_in.get("/api/chat").json() == []


def test_csrf_guard_blocks_foreign_origins_without_the_marker(signed_in):
    body = {"message": "my models"}
    no_marker = {"X-Requested-With": ""}
    assert signed_in.post("/api/chat", json=body, headers={**no_marker, "Origin": "https://evil.example"}).status_code == 403
    assert signed_in.post("/api/chat", json=body, headers={**no_marker}).status_code == 403
    assert signed_in.post("/api/chat", json=body, headers={"Origin": "https://evil.example", "X-Requested-With": "rml"}).status_code == 200
    assert signed_in.post("/api/chat", json=body, headers={**no_marker, "Origin": "http://localhost:3000"}).status_code == 200
    # the public API authenticates with a bearer key, never a cookie, so the guard is not involved
    assert signed_in.post("/v1/models/nope/predict", json={"inputs": ["x"]}, headers={**no_marker, "Authorization": "Bearer bad", "Origin": "https://evil.example"}).status_code == 401


def test_finished_lab_stream_ends_by_itself_and_resumes_from_last_event_id(signed_in, lab):
    with signed_in.stream("GET", f"/api/labs/{lab['id']}/events") as st:
        body = "".join(st.iter_text())
    assert '"stage": "done"' in body
    with signed_in.stream("GET", f"/api/labs/{lab['id']}/events", headers={"Last-Event-ID": "99999"}) as st:
        assert "event: lab" not in "".join(st.iter_text())


def test_restart_recovery_closes_out_running_labs(signed_in):
    from sqlmodel import Session, select

    from app.agents.orchestrator import recover_interrupted
    from app.db import Candidate, Lab, LabEvent, engine

    me = signed_in.get("/api/auth/me").json()["id"]
    with Session(engine) as s:
        lab = Lab(user_id=me, prompt="interrupted", status="running")
        s.add(lab); s.commit(); s.refresh(lab)
        cand = Candidate(lab_id=lab.id, strategy="naive_bayes", display_name="NB", family="classical", status="running")
        s.add(cand); s.commit()
        lab_id, cand_id = lab.id, cand.id
    recover_interrupted()
    with Session(engine) as s:
        assert s.get(Lab, lab_id).status == "failed" and s.get(Candidate, cand_id).status == "failed"
        events = s.exec(select(LabEvent).where(LabEvent.lab_id == lab_id)).all()
        assert events and events[-1].kind == "error" and events[-1].data["stage"] == "failed"
    assert signed_in.delete(f"/api/labs/{lab_id}").status_code == 200


def test_delete_lab_removes_its_artifacts(signed_in, lab):
    folder = DATA_DIR / "models" / lab["id"]
    assert folder.exists() and any(folder.iterdir())
    assert signed_in.delete(f"/api/labs/{lab['id']}").status_code == 200
    assert not folder.exists()
    assert signed_in.get(f"/api/labs/{lab['id']}").status_code == 404
