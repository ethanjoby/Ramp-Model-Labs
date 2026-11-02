from functools import lru_cache
from pathlib import Path
import secrets

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "models").mkdir(exist_ok=True)


def _persisted_secret() -> str:
    """Stable dev secret so sessions survive restarts; override with SECRET_KEY in prod."""
    path = DATA_DIR / ".secret_key"
    if not path.exists():
        path.write_text(secrets.token_urlsafe(48))
        path.chmod(0o600)
    return path.read_text().strip()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    secret_key: str = ""
    database_url: str = f"sqlite:///{DATA_DIR / 'labs.db'}"
    frontend_origin: str = "http://localhost:3000"
    cookie_secure: bool = False
    session_ttl_hours: int = 24 * 7

    firebase_project_id: str = ""  # Google sign-in; the matching web config lives in frontend/.env.local
    firebase_auth_emulator_host: str = ""  # set when testing against `firebase emulators:start --only auth`

    anthropic_api_key: str = ""
    agent_model: str = "claude-sonnet-5-5"
    fast_model: str = "claude-haiku-5-5"
    # X-Forwarded-For is trusted only when the connection comes from one of these (the Next.js proxy runs locally).
    trusted_proxies: str = "127.0.0.1,::1"

    local_lm_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # fine-tuned locally when torch is installed

    modal_enabled: bool = False  # set true after `modal token new` + `modal deploy modal_app.py`
    modal_app_name: str = "ramp-model-labs"

    max_upload_bytes: int = 5 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if not s.secret_key:
        s.secret_key = _persisted_secret()
    return s
