from datetime import datetime, timezone
from typing import Any, Optional
import uuid

from sqlalchemy import JSON, Column, event
from sqlmodel import Field, Session, SQLModel, create_engine

from .config import get_settings


def now() -> datetime:
    return datetime.now(timezone.utc)


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class User(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uid("usr"), primary_key=True)
    email: str = Field(index=True, unique=True)
    name: str
    firebase_uid: Optional[str] = Field(default=None, index=True)
    provider: str = "google"
    avatar_url: Optional[str] = None
    token_version: int = 0  # bump to revoke all sessions
    created_at: datetime = Field(default_factory=now)


class Lab(SQLModel, table=True):
    """One specialized-model project: a task described in plain English."""
    id: str = Field(default_factory=lambda: uid("lab"), primary_key=True)
    user_id: str = Field(index=True, foreign_key="user.id")
    prompt: str
    name: str = "Untitled lab"
    status: str = "queued"  # queued|running|completed|failed
    spec: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    dataset_stats: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    seed_examples: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=now)
    finished_at: Optional[datetime] = None


class LabEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    lab_id: str = Field(index=True, foreign_key="lab.id")
    seq: int
    agent: str
    kind: str  # log|stage|candidate|metric|result|error
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)


class Candidate(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uid("cand"), primary_key=True)
    lab_id: str = Field(index=True, foreign_key="lab.id")
    strategy: str
    display_name: str
    family: str  # classical|prompted|finetuned|reference
    status: str = "pending"
    metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    artifact_path: Optional[str] = None
    is_winner: bool = False


class Deployment(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uid("dep"), primary_key=True)
    user_id: str = Field(index=True, foreign_key="user.id")
    lab_id: str = Field(index=True, foreign_key="lab.id")
    candidate_id: str = Field(foreign_key="candidate.id")
    slug: str = Field(index=True, unique=True)
    name: str
    status: str = "live"
    requests: int = 0
    total_latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=now)


class ApiKey(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uid("key"), primary_key=True)
    user_id: str = Field(index=True, foreign_key="user.id")
    name: str
    prefix: str
    key_hash: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=now)
    last_used_at: Optional[datetime] = None
    revoked: bool = False


class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    deployment_id: str = Field(index=True, foreign_key="deployment.id")
    text: str
    predicted: str
    correct: str
    created_at: datetime = Field(default_factory=now)


class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True, foreign_key="user.id")
    role: str
    content: str
    data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)


engine = create_engine(
    get_settings().database_url, connect_args={"check_same_thread": False, "timeout": 5}
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record):
    # Readers (SSE replays, the UI polling) must not block the orchestrator's writes, and a busy file is a wait, not an error.
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.close()


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    # create_all never alters existing tables; add the Google sign-in columns to databases made before them.
    with engine.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql('PRAGMA table_info("user")')}
        for name, ddl in (("firebase_uid", "VARCHAR"), ("provider", "VARCHAR NOT NULL DEFAULT 'password'"),
                          ("avatar_url", "VARCHAR")):
            if name not in cols:
                conn.exec_driver_sql(f'ALTER TABLE "user" ADD COLUMN {name} {ddl}')
        if "password_hash" in cols:  # left over from email + password sign-in
            conn.exec_driver_sql('ALTER TABLE "user" DROP COLUMN password_hash')
        conn.exec_driver_sql('CREATE INDEX IF NOT EXISTS ix_user_firebase_uid ON "user" (firebase_uid)')


def get_session():
    with Session(engine) as session:
        yield session
