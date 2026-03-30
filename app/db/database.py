from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.models import Base


engine: AsyncEngine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    # Create tables for dev environments. For production, prefer migrations (Alembic).
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

