"""The offline copilot and planner: intent detection without training anything."""
import pytest
from fastapi import HTTPException

from app.agents import chat as chat_mod
from app.agents import planner
from app.datasets.templates import match_template
from conftest import make_id_token

STARTERS = [
    "Categorize card transactions into GL codes, cheap enough to run on every swipe",
    "Screen transactions for fraud and flag anything suspicious",
    "Triage incoming invoices by type and urgency",
    "Route financial support tickets to the right team",
    "Reconcile bank lines against the ledger and flag anything that doesn't match",
]
HERO = [
    "Categorize corporate card transactions into our 12 GL expense codes. Keep it cheap.",
    "Flag card transactions that look like fraud from the risk signals. Needs 90% accuracy.",
    "Route inbound support tickets to card, reimbursement, bill pay, or accounting queues.",
    "Triage AP invoices: approve, needs review, likely duplicate, or reject.",
    "Reconcile bank lines against the ledger and flag amount mismatches, timing gaps, and duplicates.",
]


@pytest.fixture
def no_training(monkeypatch):
    created = []

    def fake_create(s, user, prompt, seeds=None):
        created.append((prompt, seeds or []))
        return type("FakeLab", (), {"id": f"lab_fake{len(created)}"})()

    monkeypatch.setattr(chat_mod.lab_service, "create_lab", fake_create)
    monkeypatch.setattr(chat_mod.wf, "start", lambda wf_id: None)
    return created


@pytest.fixture
def agent_user(client):
    assert client.post("/api/auth/firebase", json={"id_token": make_id_token(sub="g-agents", email="agents@example.com", name="Agent Tester")}).status_code == 200
    yield client
    assert client.post("/api/auth/firebase", json={"id_token": make_id_token()}).status_code == 200


def test_every_starter_and_hero_prompt_is_a_build(agent_user, no_training):
    for prompt in STARTERS + HERO:
        r = agent_user.post("/api/chat", json={"message": prompt})
        assert r.status_code == 200, r.text
        assert [a["type"] for a in r.json()["actions"]] == ["lab_started"], (prompt, r.json())
    assert len(no_training) == 10


def test_only_the_colon_form_is_a_prediction_request(agent_user, no_training):
    r = agent_user.post("/api/chat", json={"message": "classify: UBER *TRIP SF $23.10"})
    assert r.status_code == 200 and "No deployed model" in r.json()["reply"] and r.json()["actions"] == []
    assert no_training == []


def test_labeled_attachment_builds_and_unlabeled_attachment_runs_a_workflow(agent_user, no_training):
    labeled = "text,label\n" + "\n".join(f"GITHUB.COM #{i},software\nDELTA AIR #{i},travel" for i in range(6))
    r = agent_user.post("/api/chat", json={"message": "use these", "attachment": labeled})
    assert [a["type"] for a in r.json()["actions"]] == ["lab_started"] and len(no_training[-1][1]) == 12
    unlabeled = "text\n" + "\n".join(f"card-not-present ${i * 7}.00 at STAPLES 0{i % 9}:12 local" for i in range(8))
    r = agent_user.post("/api/chat", json={"message": "categorize these", "attachment": unlabeled})
    assert [a["type"] for a in r.json()["actions"]] == ["workflow_started"]
    wfs = agent_user.get("/api/workflows").json()
    assert wfs and wfs[0]["rows_total"] == 8 and wfs[0]["source_kind"] in ("csv", "text")


def test_list_status_and_help(agent_user, no_training):
    assert agent_user.post("/api/chat", json={"message": "what models do I have"}).json()["actions"][0]["type"] == "models"
    assert agent_user.post("/api/chat", json={"message": "status"}).json()["reply"] == "No labs yet."
    assert "Try" in agent_user.post("/api/chat", json={"message": "hello there"}).json()["reply"]
    assert no_training == []


def test_rate_limit_message_reaches_the_user(agent_user, monkeypatch):
    def boom(*a, **k):
        raise HTTPException(429, "Too many requests. Slow down a moment.")
    monkeypatch.setattr(chat_mod.lab_service, "create_lab", boom)
    r = agent_user.post("/api/chat", json={"message": "Build a fraud model"})
    assert r.status_code == 200 and "Slow down" in r.json()["reply"] and r.json()["actions"] == []


def test_constraint_parser_needs_the_number_next_to_the_word():
    assert planner._constraints("needs 90% accuracy") == {"target_accuracy": 0.9, "priority": "accuracy"}
    assert planner._constraints("about 20% of our volume is refunds") == {}
    assert planner._constraints("must respond under 50ms") == {"max_latency_ms": 50.0}
    assert planner._constraints("keep it cheap") == {"priority": "cost"}
    assert planner._constraints("route 100 tickets a day") == {}


def test_template_matching_weights_specific_words_and_refuses_ties():
    assert match_template("Screen transactions for fraud").key == "fraud_detection"
    assert match_template("Categorize card transactions into GL codes").key == "expense_categorization"
    assert match_template("transaction") is None
    assert match_template("write me a poem") is None


def test_system_prompt_only_claims_what_is_enabled():
    text = chat_mod.system_prompt()
    assert "small specialist models trained locally" in text
    assert "estimated from list prices" in text  # no Anthropic key in tests
    assert "prompted Claude candidates" not in text
