from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AuthRegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str
    password: str = Field(min_length=8, max_length=200)


class AuthLoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class BotCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)


class BotResponse(BaseModel):
    id: int
    owner_user_id: int
    name: str
    description: str | None
    api_key: str
    created_at: datetime


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = None
    # Optional override knobs (kept minimal for now).
    top_k: int = Field(default=6, ge=1, le=20)


class ErrorResponse(BaseModel):
    detail: str

