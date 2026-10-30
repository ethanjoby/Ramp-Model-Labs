"""Verifies Firebase ID tokens from Google sign-in with Google's public certs and PyJWT, no Admin SDK needed."""
from __future__ import annotations

from dataclasses import dataclass
import re
import time

from cryptography import x509
from fastapi import HTTPException
import httpx
import jwt

from .config import get_settings

CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
_certs: dict = {"keys": {}, "expires": 0.0}


@dataclass
class Identity:
    uid: str
    email: str
    email_verified: bool
    name: str
    picture: str | None
    provider: str


async def _public_keys(force: bool = False) -> dict[str, object]:
    if not force and _certs["keys"] and time.time() < _certs["expires"]:
        return _certs["keys"]
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(CERTS_URL)
        r.raise_for_status()
    keys = {kid: x509.load_pem_x509_certificate(pem.encode()).public_key() for kid, pem in r.json().items()}
    m = re.search(r"max-age=(\d+)", r.headers.get("cache-control", ""))
    _certs.update(keys=keys, expires=time.time() + (int(m.group(1)) if m else 3600))
    return keys


async def verify_id_token(token: str) -> Identity:
    s = get_settings()
    if not s.firebase_project_id:
        raise HTTPException(503, "Google sign-in isn't configured on this server yet.")
    issuer = f"https://securetoken.google.com/{s.firebase_project_id}"
    try:
        if s.firebase_auth_emulator_host:
            # The Auth emulator signs nothing; everything else about the token is still checked.
            payload = jwt.decode(token, options={"verify_signature": False, "verify_exp": True, "verify_aud": True,
                                                 "verify_iss": True}, audience=s.firebase_project_id, issuer=issuer)
        else:
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256":
                raise jwt.InvalidAlgorithmError("expected RS256")
            keys = await _public_keys()
            key = keys.get(header.get("kid")) or (await _public_keys(force=True)).get(header.get("kid"))
            if key is None:
                raise jwt.InvalidKeyError("unknown signing key")
            payload = jwt.decode(token, key=key, algorithms=["RS256"], audience=s.firebase_project_id, issuer=issuer,
                                 options={"require": ["exp", "iat", "sub"]})
    except jwt.PyJWTError:
        raise HTTPException(401, "Google sign-in didn't go through. Try again.")
    sub = payload.get("sub")
    if not sub or payload.get("auth_time", 0) > time.time() + 300:
        raise HTTPException(401, "Google sign-in didn't go through. Try again.")
    return Identity(
        uid=sub, email=(payload.get("email") or "").lower(), email_verified=bool(payload.get("email_verified")),
        name=(payload.get("name") or "").strip(), picture=payload.get("picture"),
        provider=(payload.get("firebase") or {}).get("sign_in_provider", ""),
    )
