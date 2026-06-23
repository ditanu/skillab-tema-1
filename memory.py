from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


Message = dict[str, str]


class ConversationMemory(Protocol):
    def load_messages(self, session_id: str) -> list[Message]:
        """Returneaza istoricul relevant pentru sesiunea curenta."""

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """Persistă un mesaj nou in memoria conversatiei."""


@dataclass
class InMemoryConversationMemory:
    window: int = 10
    _messages: dict[str, list[Message]] = field(default_factory=lambda: defaultdict(list))

    def load_messages(self, session_id: str) -> list[Message]:
        return list(self._messages[session_id][-self.window :])

    def save_message(self, session_id: str, role: str, content: str) -> None:
        self._messages[session_id].append({"role": role, "content": content})


class PersistentConversationMemory:
    def __init__(
        self,
        database_url: str,
        window: int = 10,
        create_tables: bool = True,
    ):
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
        except ImportError as error:
            raise RuntimeError(
                "PersistentConversationMemory necesita pachetele sqlalchemy si psycopg. "
                "Instaleaza dependintele din requirements.txt."
            ) from error

        self.window = window
        self.engine = create_engine(database_url)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

        if create_tables:
            Base.metadata.create_all(self.engine)

    def load_messages(self, session_id: str) -> list[Message]:
        with unit_of_work(self.session_factory) as db:
            rows = ChatMessageRepository(db).latest(session_id, self.window)
            return [{"role": row.role, "content": row.content} for row in rows]

    def save_message(self, session_id: str, role: str, content: str) -> None:
        with unit_of_work(self.session_factory) as db:
            SessionRepository(db).get_or_create(session_id=session_id)
            ChatMessageRepository(db).add(session_id=session_id, role=role, content=content)


def build_conversation_memory(
    database_url: str | None = None,
    window: int = 10,
) -> ConversationMemory:
    if database_url:
        return PersistentConversationMemory(database_url=database_url, window=window)

    return InMemoryConversationMemory(window=window)


try:
    from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, func, select
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session as DBSession
    from sqlalchemy.orm import mapped_column, relationship, sessionmaker

    class Base(DeclarativeBase):
        pass

    class ConversationSession(Base):
        __tablename__ = "sessions"

        id: Mapped[str] = mapped_column(String(64), primary_key=True)
        user_id: Mapped[str] = mapped_column(String(64), index=True, default="local-user")
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
        meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

        messages: Mapped[list["ChatMessage"]] = relationship(
            back_populates="session",
            cascade="all, delete-orphan",
        )

    class ChatMessage(Base):
        __tablename__ = "chat_messages"

        id: Mapped[int] = mapped_column(primary_key=True)
        session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
        role: Mapped[str] = mapped_column(String(16))
        content: Mapped[str] = mapped_column(Text)
        timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())

        session: Mapped["ConversationSession"] = relationship(back_populates="messages")

    @contextmanager
    def unit_of_work(session_factory: sessionmaker):
        db: DBSession = session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    class SessionRepository:
        def __init__(self, db: DBSession):
            self.db = db

        def get_or_create(self, session_id: str, user_id: str = "local-user") -> ConversationSession:
            existing = self.db.get(ConversationSession, session_id)

            if existing:
                return existing

            session = ConversationSession(id=session_id, user_id=user_id, meta={})
            self.db.add(session)
            return session

    class ChatMessageRepository:
        def __init__(self, db: DBSession):
            self.db = db

        def add(self, session_id: str, role: str, content: str) -> ChatMessage:
            message = ChatMessage(session_id=session_id, role=role, content=content)
            self.db.add(message)
            return message

        def latest(self, session_id: str, limit: int) -> list[ChatMessage]:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.timestamp.desc(), ChatMessage.id.desc())
                .limit(limit)
            )
            rows = self.db.execute(stmt).scalars().all()
            return list(reversed(rows))

except ImportError:
    Base = None
