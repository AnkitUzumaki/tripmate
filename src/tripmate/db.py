"""Persistence. SQLite by default, PostgreSQL via DATABASE_URL, no code change."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import (
    Float, ForeignKey, Integer, JSON, LargeBinary, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from tripmate.config import Settings, get_settings
from tripmate.models import TraceEvent

POOL_SIZE = 5
MAX_OVERFLOW = 10


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    last_active_at: Mapped[float] = mapped_column(Float, default=time.time)


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class TraceRow(Base):
    __tablename__ = "trace_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[int] = mapped_column(Integer, ForeignKey("turns.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[float] = mapped_column(Float)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class CacheRow(Base):
    __tablename__ = "query_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    answer: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        kwargs: dict[str, Any] = {"future": True}
        if not url.startswith("sqlite"):
            kwargs |= {
                "pool_size": POOL_SIZE,
                "max_overflow": MAX_OVERFLOW,
                "pool_pre_ping": True,
            }
        self.engine = create_engine(url, **kwargs)
        self._factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


class SessionStore:
    """Conversation and trace persistence."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def ensure_session(self, session_id: str) -> None:
        with self.db.session() as session:
            if session.get(SessionRow, session_id) is None:
                session.add(SessionRow(id=session_id))

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
    ) -> int:
        self.ensure_session(session_id)
        with self.db.session() as session:
            row = TurnRow(
                session_id=session_id, role=role, content=content,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                cost_usd=cost_usd, latency_ms=latency_ms,
            )
            session.add(row)
            session.flush()
            return row.id

    def add_trace(self, turn_id: int, events: list[TraceEvent]) -> None:
        with self.db.session() as session:
            session.add_all([
                TraceRow(
                    turn_id=turn_id, seq=event.seq, event_type=event.event_type,
                    timestamp=event.timestamp, duration_ms=event.duration_ms,
                    payload=event.payload,
                )
                for event in events
            ])

    def history(self, session_id: str, limit: int = 20) -> list[dict[str, str]]:
        """The most recent `limit` turns, oldest first, as LLM messages."""
        with self.db.session() as session:
            rows = (
                session.query(TurnRow)
                .filter_by(session_id=session_id)
                .order_by(TurnRow.id.desc())
                .limit(limit)
                .all()
            )
        return [{"role": row.role, "content": row.content} for row in reversed(rows)]


def build_database(settings: Settings | None = None) -> Database:
    settings = settings or get_settings()
    db = Database(url=settings.database_url)
    db.create_all()
    return db
