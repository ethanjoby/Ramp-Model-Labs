from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .agents.orchestrator import recover_interrupted
from .api.routes import api, public
from .config import get_settings
from .db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    recover_interrupted()
    yield


app = FastAPI(title="Ramp Model Labs", version="1.0.0", lifespan=lifespan,
              docs_url="/v1/docs", openapi_url="/v1/openapi.json", redoc_url=None)
settings = get_settings()
app.add_middleware(CORSMiddleware, allow_origins=[settings.frontend_origin], allow_credentials=True,
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "Authorization"])

UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


@app.middleware("http")
async def guard(request: Request, call_next):
    # CSRF defense for cookie-authenticated routes: unsafe requests must come from our own origin.
    if request.method in UNSAFE and request.url.path.startswith("/api/") and "rml_session" in request.cookies:
        origin = request.headers.get("origin") or request.headers.get("referer")
        allowed = {urlparse(settings.frontend_origin).netloc, request.headers.get("host")}
        if origin and urlparse(origin).netloc not in allowed:
            return JSONResponse({"detail": "Cross-site request blocked"}, status_code=403)
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return resp


@app.get("/health")
def health():
    return {"ok": True}


app.include_router(api)
app.include_router(public)
