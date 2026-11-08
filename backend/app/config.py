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

    # Together AI (OpenAI-compatible) unlocks open-weight frontier models as prompted candidates, and stands in for
    # the Claude agents (planner, synthesizer, teacher, yardstick) when no Anthropic key is set.
    together_api_key: str = ""
    together_models: str = ("deepseek-ai/DeepSeek-V4.1-Flash-1017,moonshotai/Kimi-K3.1-Pro,MiniMaxAI/MiniMax-M3.1-Flash,"
                            "zai-org/GLM-5.3-Flash,Qwen/Qwen3.5-9B")  # zero-shot candidates, comma-separated API strings
    together_agent_model: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017"  # planning, synthesis, teacher labels, prompt search
    together_reference_model: str = "moonshotai/Kimi-K3.1-Pro"  # the frontier yardstick when Claude is not configured

    local_lm_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # fine-tuned locally when torch is installed
    local_embed_model: str = "BAAI/bge-small-en-v1.5"  # frozen sentence embeddings + linear head (33M, CPU-served)
    local_llm_model: str = "Qwen/Qwen3-0.6B"  # a small decoder LLM fine-tuned as a classifier on this machine; "" disables

    modal_enabled: bool = False  # set true after `modal token new` + `modal deploy modal_app.py`
    modal_app_name: str = "ramp-model-labs"
    modal_base_model: str = "Qwen/Qwen3-1.7B"  # the LoRA candidate's base; any HF causal LM with a chat template
    modal_heavy_model: str = ""  # e.g. "Qwen/Qwen3-4B-Instruct-2507" adds a second, slower LoRA candidate
    # The GPU roster: `org/model[@GPU][:4bit]`, comma-separated; the GPU defaults from the model's size and 4-bit (QLoRA)
    # switches on when bf16 weights would not fit the card. Replaces the two fields above when set.
    modal_models: str = ("Qwen/Qwen3-1.7B,deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,"
                         "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,moonshotai/Moonlight-16B-A3B-Instruct")

    teacher_max_rows: int = 1500  # unlabeled uploaded rows the frontier model will label per lab (about $1 per 1k)
    distill: bool = True  # after training, teach the cheapest local model from the strongest expensive one when it lags
    distill_rows: int = 1500  # unlabeled rows the teacher labels for the student

    max_upload_bytes: int = 5 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if not s.secret_key:
        s.secret_key = _persisted_secret()
    return s
