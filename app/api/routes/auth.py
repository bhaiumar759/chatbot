from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.schemas import AuthLoginRequest, AuthRegisterRequest, TokenResponse
from app.core.security import create_access_token, create_api_key, hash_password, verify_password
from app.db.database import get_db
from app.db.models import User, UserRole
from app.core.config import settings


router = APIRouter(tags=["auth"])


@router.post("/auth/register", response_model=TokenResponse)
async def register(
    request: Request,
    body: AuthRegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    existing = await db.scalar(select(User).where(User.email == body.email))
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered.")

    user_count = await db.scalar(select(func.count()).select_from(User))

    invite_token = request.headers.get("X-Admin-Invite")
    bootstrap_is_admin = bool(settings.bootstrap_admin_email and body.email == settings.bootstrap_admin_email and user_count == 0)
    invite_is_admin = bool(settings.admin_invite_token and invite_token and invite_token == settings.admin_invite_token)

    role = UserRole.admin.value if (bootstrap_is_admin or invite_is_admin) else UserRole.standard.value

    user = User(
        name=body.name,
        email=body.email,
        password_hash=hash_password(body.password),
        role=role,
        api_key=create_api_key(),
    )
    db.add(user)
    await db.commit()

    token = create_access_token(subject=str(user.id), additional_claims={"role": user.role})
    return TokenResponse(access_token=token)


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    body: AuthLoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    user = await db.scalar(select(User).where(User.email == body.email))
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    token = create_access_token(subject=str(user.id), additional_claims={"role": user.role})
    return TokenResponse(access_token=token)

