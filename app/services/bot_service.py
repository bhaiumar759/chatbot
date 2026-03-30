from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Bot, User


async def get_bot_for_user(db: AsyncSession, bot_id: int, current_user: User) -> Bot:
    bot = await db.scalar(select(Bot).where(Bot.id == bot_id))
    if not bot:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot not found.")
    if current_user.role != "admin" and bot.owner_user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden.")
    return bot

