"""Auth primitives: signed session cookies (issued after Google sign-in), hashed API keys, rate limits."""
from collections import deque
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
import time

from fastapi import Depends, HTTPException, Request, status
import jwt
from sqlmodel import Session, select

from .config import Settings, get_settings
from .db import ApiKey, User, get_session

SESSION_COOKIE = "rml_session"


# ---- sessions ----

def cookie_secure() -> bool:
    """Secure cookies whenever the site is served over https; COOKIE_SECURE, when set, is the last word."""
    s = get_settings()
    if "cookie_secure" in s.model_fields_set:
        return s.cookie_secure
    return s.frontend_origin.startswith("https://")


def check_cookie_policy(s: Settings | None = None) -> None:
    """An https site with COOKIE_SECURE=false would let the session cookie travel over plain http.
    Called from the app lifespan so a misconfigured deployment refuses to start instead of leaking sessions."""
    s = s if s is not None else get_settings()
    if s.frontend_origin.startswith("https://") and "cookie_secure" in s.model_fields_set and not s.cookie_secure:
        raise RuntimeError("FRONTEND_ORIGIN is https but COOKIE_SECURE=false. Unset COOKIE_SECURE or set it to true.")


def create_session_token(user: User) -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.id,
        "tv": user.token_version,
        "jti": secrets.token_urlsafe(16),  # names this one cookie, so a future denylist can revoke it alone
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
    # The raw key already carries 256 bits of entropy, so a plain hash is enough, and the stored
    # digests survive a SECRET_KEY rotation.
    return hashlib.sha256(raw.encode()).hexdigest()


def _legacy_digest(raw: str) -> str:
    # Keys issued before the switch were HMAC-ed with SECRET_KEY; they stay valid on lookup.
    return hmac.new(get_settings().secret_key.encode(), raw.encode(), hashlib.sha256).hexdigest()


def new_api_key() -> tuple[str, str, str]:
    raw = "rml_live_" + secrets.token_urlsafe(32)
    return raw, raw[:16], _digest(raw)


def find_api_key(session: Session, raw: str) -> ApiKey | None:
    digests = [_digest(raw), _legacy_digest(raw)]
    return session.exec(
        select(ApiKey).where(ApiKey.key_hash.in_(digests), ApiKey.revoked == False)  # noqa: E712
    ).first()


def api_key_user(request: Request, session: Session = Depends(get_session)) -> User:
    """Accept either a Bearer API key (programmatic) or a session cookie (playground)."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        key = find_api_key(session, auth[7:].strip())
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
    SWEEP_EVERY_S = 60.0

    def __init__(self) -> None:
        self.hits: dict[str, deque[float]] = {}
        self.windows: dict[str, float] = {}
        self._last_sweep = time.monotonic()

    def _sweep(self, now: float) -> None:
        # Every key is one IP or user; without this, every visitor ever seen stays in memory.
        if now - self._last_sweep < self.SWEEP_EVERY_S:
            return
        self._last_sweep = now
        for key in list(self.hits):
            q, window = self.hits[key], self.windows.get(key, 0.0)
            while q and now - q[0] > window:
                q.popleft()
            if not q:
                del self.hits[key]
                self.windows.pop(key, None)

    def check(self, key: str, limit: int, window_s: float) -> None:
        now = time.monotonic()
        self._sweep(now)
        q = self.hits.get(key)
        if q is None:
            q = self.hits[key] = deque()
        self.windows[key] = window_s
        while q and now - q[0] > window_s:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Too many requests. Slow down a moment.",
                headers={"Retry-After": str(int(window_s))},
            )
        q.append(now)

    def reset(self, key: str) -> None:
        self.hits.pop(key, None)
        self.windows.pop(key, None)


limiter = RateLimiter()


def _trusted_proxies() -> set[str]:
    return {p.strip() for p in get_settings().trusted_proxies.split(",") if p.strip()}


def client_ip(request: Request) -> str:
    """The address rate limits key on. X-Forwarded-For is attacker-controlled unless a proxy we run appended
    it, so it counts only when the connection itself comes from a trusted proxy, and only the hop that proxy
    added (the last one); anything earlier in the list was supplied by the client."""
    peer = request.client.host if request.client else "?"
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd and peer in _trusted_proxies():
        hops = [h.strip() for h in fwd.split(",") if h.strip()]
        if hops:
            return hops[-1]
    return peer
