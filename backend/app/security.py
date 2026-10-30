"""Auth primitives: signed session cookies (issued after Google sign-in), hashed API keys, rate limits."""
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
import time

from fastapi import Depends, HTTPException, Request, status
import jwt
from sqlmodel import Session, select

from .config import get_settings
from .db import ApiKey, User, get_session

SESSION_COOKIE = "rml_session"


def create_session_token(user: User) -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.id,
        "tv": user.token_version,
        "iat": now,
        "exp": now + timedelta(hours=s.session_ttl_hours),
    }
    return jwt.encode(payload, s.secret_key, algorithm="HS256")


def current_user(request: Request, session: Session = Depends(get_session)) -> User:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not signed in")
    try:
        payload = jwt.decode(token, get_settings().secret_key, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired")
    user = session.get(User, payload.get("sub"))
    if not user or user.token_version != payload.get("tv"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session revoked")
    return user


# ---- API keys (for calling deployed models) ----

def _digest(raw: str) -> str:
    return hmac.new(get_settings().secret_key.encode(), raw.encode(), hashlib.sha256).hexdigest()


def new_api_key() -> tuple[str, str, str]:
    raw = "rml_live_" + secrets.token_urlsafe(32)
    return raw, raw[:16], _digest(raw)


def api_key_user(request: Request, session: Session = Depends(get_session)) -> User:
    """Accept either a Bearer API key (programmatic) or a session cookie (playground)."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        key = session.exec(
            select(ApiKey).where(ApiKey.key_hash == _digest(auth[7:].strip()), ApiKey.revoked == False)  # noqa: E712
        ).first()
        if not key:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
        key.last_used_at = datetime.now(timezone.utc)
        session.add(key)
        session.commit()
        user = session.get(User, key.user_id)
        if not user:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
        return user
    return current_user(request, session)


# ---- rate limiting (in-process sliding window; swap for Redis when multi-instance) ----

class RateLimiter:
    def __init__(self) -> None:
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: int, window_s: float) -> None:
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > window_s:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Too many requests — slow down a moment.",
                headers={"Retry-After": str(int(window_s))},
            )
        q.append(now)


limiter = RateLimiter()


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")
