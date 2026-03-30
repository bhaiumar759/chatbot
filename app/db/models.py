from __future__ import annotations

import datetime as dt
from enum import Enum

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import JSON as SA_JSON


class Base(DeclarativeBase):
    pass


class UserRole(str, Enum):
    admin = "admin"
    standard = "standard"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    # Use 191 to stay within older MySQL/MariaDB utf8mb4 index limits.
    email: Mapped[str] = mapped_column(String(191), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default=UserRole.standard.value)
    api_key: Mapped[str] = mapped_column(String(191), nullable=False, unique=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=dt.datetime.now(dt.timezone.utc))

    bots: Mapped[list["Bot"]] = relationship("Bot", back_populates="owner", cascade="all, delete-orphan")


class Bot(Base):
    __tablename__ = "bots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    api_key: Mapped[str] = mapped_column(String(191), nullable=False, unique=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=dt.datetime.now(dt.timezone.utc))

    owner: Mapped["User"] = relationship("User", back_populates="bots")
    pdfs: Mapped[list["PDF"]] = relationship("PDF", back_populates="bot", cascade="all, delete-orphan")
    chat_sessions: Mapped[list["ChatSession"]] = relationship("ChatSession", back_populates="bot", cascade="all, delete-orphan")


class PDF(Base):
    __tablename__ = "pdfs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id", ondelete="CASCADE"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(Text, nullable=False)
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=dt.datetime.now(dt.timezone.utc))

    bot: Mapped["Bot"] = relationship("Bot", back_populates="pdfs")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(191), nullable=False)
    messages: Mapped[object] = mapped_column(SA_JSON, nullable=False, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=dt.datetime.now(dt.timezone.utc))

    __table_args__ = (UniqueConstraint("bot_id", "session_id", name="uq_bot_session"),)

    bot: Mapped["Bot"] = relationship("Bot", back_populates="chat_sessions")


class VectorStore(Base):
    __tablename__ = "vectorstores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    faiss_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    bm25_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    parent_chunks_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=dt.datetime.now(dt.timezone.utc))

