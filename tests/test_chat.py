import asyncio
import uuid

from sqlalchemy import select


def test_ask_streaming_persists_chat_session(client, monkeypatch):
    import app.api.routes.chat as chat_route
    from app.db.database import AsyncSessionLocal
    from app.db.models import ChatSession

    email = f"user_{uuid.uuid4().hex[:8]}@example.com"
    r = client.post(
        "/auth/register",
        json={"name": "Test User", "email": email, "password": "password123"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    r = client.post("/bots", headers=headers, json={"name": "Bot Chat", "description": "x"})
    assert r.status_code == 200, r.text
    bot_id = r.json()["id"]

    async def fake_get_state(bot_id: int):
        return None

    async def fake_hybrid_search(query: str, state, top_k: int = 6):
        return ["CTX_A", "CTX_B"]

    def fake_stream_answer(prompt: str):
        yield "Answer "
        yield "from mock."

    monkeypatch.setattr(chat_route, "get_vectorstore_state", fake_get_state)
    monkeypatch.setattr(chat_route, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(chat_route, "stream_answer", fake_stream_answer)

    session_id = f"sess_{uuid.uuid4().hex[:10]}"
    r = client.post(
        f"/bots/{bot_id}/ask",
        headers=headers,
        json={"query": "What is this?", "session_id": session_id, "top_k": 2},
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("X-Session-Id") == session_id
    assert "Answer from mock." in r.text

    async def _check():
        async with AsyncSessionLocal() as db:
            cs = await db.scalar(
                select(ChatSession).where(ChatSession.bot_id == bot_id, ChatSession.session_id == session_id)
            )
            assert cs is not None
            msgs = cs.messages
            assert isinstance(msgs, list)
            assert len(msgs) == 1
            assert msgs[0]["assistant"] == "Answer from mock."

    asyncio.run(_check())

