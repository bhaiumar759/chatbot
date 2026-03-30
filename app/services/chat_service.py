from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatSession


async def get_or_create_chat_session(
    db: AsyncSession,
    bot_id: int,
    session_id: str,
) -> ChatSession:
    session = await db.scalar(
        select(ChatSession).where(
            ChatSession.bot_id == bot_id,
            ChatSession.session_id == session_id,
        )
    )
    if session:
        return session

    session = ChatSession(bot_id=bot_id, session_id=session_id, messages=[])
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def append_chat_message(
    db: AsyncSession,
    chat_session: ChatSession,
    user_message: str,
    assistant_message: str,
) -> None:
    messages: list[dict[str, Any]] = list(chat_session.messages or [])
    messages.append({"user": user_message, "assistant": assistant_message})
    chat_session.messages = messages
    db.add(chat_session)
    await db.commit()

