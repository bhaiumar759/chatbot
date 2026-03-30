from __future__ import annotations

import secrets
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.schemas import BotCreateRequest, BotResponse
from app.core.security import create_api_key
from app.db.database import get_db
from app.db.models import Bot, User


router = APIRouter(tags=["bots"])


def _require_bot_access(user: User, bot: Bot) -> None:
    if user.role != "admin" and bot.owner_user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden.")


@router.post("/bots", response_model=BotResponse)
async def create_bot(
    body: BotCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> BotResponse:
    bot = Bot(
        owner_user_id=current_user.id,
        name=body.name,
        description=body.description,
        api_key=create_api_key(),
    )
    db.add(bot)
    await db.commit()
    await db.refresh(bot)
    return BotResponse(
        id=bot.id,
        owner_user_id=bot.owner_user_id,
        name=bot.name,
        description=bot.description,
        api_key=bot.api_key,
        created_at=bot.created_at,
    )


@router.get("/bots", response_model=list[BotResponse])
async def list_bots(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[BotResponse]:
    stmt = select(Bot)
    if current_user.role != "admin":
        stmt = stmt.where(Bot.owner_user_id == current_user.id)
    result = await db.scalars(stmt.order_by(Bot.created_at.desc()))
    bots = result.all()
    return [
        BotResponse(
            id=b.id,
            owner_user_id=b.owner_user_id,
            name=b.name,
            description=b.description,
            api_key=b.api_key,
            created_at=b.created_at,
        )
        for b in bots
    ]


@router.get("/bots/{bot_id}", response_model=BotResponse)
async def get_bot(
    bot_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> BotResponse:
    bot = await db.scalar(select(Bot).where(Bot.id == bot_id))
    if not bot:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot not found.")
    _require_bot_access(current_user, bot)
    return BotResponse(
        id=bot.id,
        owner_user_id=bot.owner_user_id,
        name=bot.name,
        description=bot.description,
        api_key=bot.api_key,
        created_at=bot.created_at,
    )


@router.delete("/bots/{bot_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bot(
    bot_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    bot = await db.scalar(select(Bot).where(Bot.id == bot_id))
    if not bot:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot not found.")
    _require_bot_access(current_user, bot)
    await db.delete(bot)
    await db.commit()

