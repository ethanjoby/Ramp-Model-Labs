"""Shared fixtures. Environment is pinned before any app import so settings, engine and data paths are test-local."""
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone

_TMP = tempfile.mkdtemp(prefix="rml-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/labs.db"
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["FIREBASE_PROJECT_ID"] = "rml-test"
os.environ["FIREBASE_AUTH_EMULATOR_HOST"] = "127.0.0.1:1"  # unsigned-token path; nothing is contacted
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["MODAL_ENABLED"] = "false"

import jwt  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

PROJECT = "rml-test"


def make_id_token(sub="g-1", email="tester@example.com", verified=True, name="Test User", provider="google.com",
                  picture="https://lh3.googleusercontent.com/a/x", aud=PROJECT, issuer=None, exp_delta=3600, alg="none"):
    now = datetime.now(timezone.utc)
    payload = {"sub": sub, "aud": aud, "iss": issuer or f"https://securetoken.google.com/{PROJECT}",
               "iat": now, "exp": now + timedelta(seconds=exp_delta), "auth_time": int(now.timestamp()),
               "email": email, "email_verified": verified, "name": name, "picture": picture,
               "firebase": {"sign_in_provider": provider}}
    return jwt.encode(payload, None, algorithm=alg)


@pytest.fixture(scope="session")
def client():
    with TestClient(app, headers={"X-Requested-With": "rml"}) as c:  # the marker the real frontend sends
        yield c


@pytest.fixture(scope="session")
def signed_in(client):
    r = client.post("/api/auth/firebase", json={"id_token": make_id_token()})
    assert r.status_code == 200, r.text
    return client


def wait_for_lab(client, lab_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        lab = client.get(f"/api/labs/{lab_id}").json()
        if lab["status"] in ("completed", "failed"):
            return lab
        time.sleep(1)
    raise AssertionError(f"lab {lab_id} did not finish in {timeout}s")
