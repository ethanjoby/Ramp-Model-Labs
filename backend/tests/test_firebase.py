import asyncio

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from app import firebase
from app.config import get_settings
from conftest import PROJECT, make_id_token


def verify(token):
    return asyncio.run(firebase.verify_id_token(token))


def test_emulator_token_is_accepted_and_mapped():
    ident = verify(make_id_token(sub="abc", email="A@Example.com", name="Ada"))
    assert ident.uid == "abc" and ident.email == "a@example.com" and ident.email_verified and ident.provider == "google.com"


@pytest.mark.parametrize("kwargs", [
    {"aud": "someone-else"},
    {"issuer": "https://securetoken.google.com/other"},
    {"exp_delta": -60},
])
def test_bad_claims_are_rejected_even_unsigned(kwargs):
    with pytest.raises(HTTPException) as e:
        verify(make_id_token(**kwargs))
    assert e.value.status_code == 401


def test_rs256_path_checks_signature_alg_and_kid(monkeypatch):
    settings = get_settings().model_copy(update={"firebase_auth_emulator_host": ""})
    monkeypatch.setattr(firebase, "get_settings", lambda: settings)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    async def keys(force=False):
        return {"k1": key.public_key()}
    monkeypatch.setattr(firebase, "_public_keys", keys)
    import time
    now = int(time.time())
    claims = {"sub": "u1", "aud": PROJECT, "iss": f"https://securetoken.google.com/{PROJECT}", "iat": now, "exp": now + 600,
              "auth_time": now, "email": "u@example.com", "email_verified": True, "firebase": {"sign_in_provider": "google.com"}}
    good = jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})
    assert verify(good).uid == "u1"
    for bad in (jwt.encode(claims, other, algorithm="RS256", headers={"kid": "k1"}),
                jwt.encode(claims, key, algorithm="RS256", headers={"kid": "unknown"}),
                jwt.encode(claims, "secret", algorithm="HS256", headers={"kid": "k1"}),
                make_id_token()):
        with pytest.raises(HTTPException):
            verify(bad)
