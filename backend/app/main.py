import asyncio
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import events, firebase
from .agents.orchestrator import recover_interrupted
from .api.routes import api, public
from .api.workflows import router as workflows_router
from .config import get_settings
from .db import init_db
from .security import check_cookie_policy


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_cookie_policy()  # raises: an https site must not run with COOKIE_SECURE=false
    firebase.log_startup()
    init_db()
    recover_interrupted()
    events.bind_loop(asyncio.get_running_loop())  # sync endpoints start labs from worker threads via this loop
    yield


app = FastAPI(title="Ramp Model Labs", version="1.0.0", lifespan=lifespan,
              docs_url="/v1/docs", openapi_url="/v1/openapi.json", redoc_url=None)
settings = get_settings()
app.add_middleware(CORSMiddleware, allow_origins=[settings.frontend_origin], allow_credentials=True,
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "Authorization", "X-Requested-With"])

UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


@app.middleware("http")
async def guard(request: Request, call_next):
    # CSRF defense for cookie-authenticated routes. Our frontend marks every request with a custom header,
    # which a cross-site page cannot add without a CORS preflight we never grant; a request without it
    # must at least come from a configured or same-host origin.
    if request.method in UNSAFE and request.url.path.startswith("/api/") and "rml_session" in request.cookies:
        if request.headers.get("x-requested-with") != "rml":
            origin = request.headers.get("origin") or request.headers.get("referer")
            allowed = {urlparse(settings.frontend_origin).netloc, request.headers.get("host"),
                       request.headers.get("x-forwarded-host")}
            if not origin or urlparse(origin).netloc not in allowed:
                return JSONResponse({"detail": "Cross-site request blocked"}, status_code=403)
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return resp


@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}


# Only /v1 is a public contract; the cookie-authenticated /api routes serve our own frontend.
app.include_router(api, include_in_schema=False)
app.include_router(workflows_router, include_in_schema=False)
app.include_router(public)
