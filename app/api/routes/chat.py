from __future__ import annotations

import hashlib
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.schemas import AskRequest
from app.db.database import get_db
from app.db.models import User, VectorStore
from app.rag.gemini_client import stream_answer
from app.rag.prompting import build_prompt
from app.rag.retrieval import hybrid_search
from app.rag.vectorstore_manager import get_vectorstore_state
from app.services.bot_service import get_bot_for_user
from app.services.chat_service import append_chat_message, get_or_create_chat_session


router = APIRouter(tags=["chat"])


def _client_ip(request: Request) -> str:
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _session_id_for_user(user_id: int, ip: str) -> str:
    raw = f"{user_id}:{ip}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@router.post("/bots/{bot_id}/ask")
async def ask(
    request: Request,
    bot_id: int,
    body: AskRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    bot = await get_bot_for_user(db, bot_id, current_user)

    # Determine session_id (deterministic by user + IP, unless client provides one).
    ip = _client_ip(request)
    session_id = body.session_id or _session_id_for_user(current_user.id, ip)

    chat_session = await get_or_create_chat_session(db, bot.id, session_id)

    # Load vectorstore for this bot.
    try:
        state = await get_vectorstore_state(bot.id)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No documents indexed yet — upload a PDF first.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Vector store error: {e}")

    contexts = await hybrid_search(body.query, state, top_k=body.top_k)
    if not contexts:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No relevant content found for that query.")

    messages = list(chat_session.messages or [])
    history = "\n".join(
        f"User: {m['user']}\nAssistant: {m['assistant']}"
        for m in messages[-5:]
        if isinstance(m, dict) and "user" in m and "assistant" in m
    )
    prompt = build_prompt(contexts, history, body.query)

    async def stream_and_persist() -> AsyncGenerator[str, None]:
        tokens: list[str] = []
        try:
            for t in stream_answer(prompt):
                tokens.append(t)
                yield t
        except Exception as e:
            err = f"\n[Stream error: {e}]"
            yield err
        finally:
            final_text = "".join(tokens).strip()
            if final_text:
                await append_chat_message(db, chat_session, body.query, final_text)

    return StreamingResponse(
        stream_and_persist(),
        media_type="text/plain",
        status_code=200,
        headers={"X-Session-Id": session_id},
    )

