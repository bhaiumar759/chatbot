import asyncio
import uuid

import pytest
from langchain_core.documents import Document
from sqlalchemy import func, select


def _make_pdf_bytes() -> bytes:
    # Content doesn't need to be a valid PDF since loader is mocked.
    return b"%PDF-1.4\n% Test\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


def test_pdf_upload_indexes_and_persists(client, monkeypatch):
    import app.api.routes.pdf_upload as pdf_route
    from app.db.database import AsyncSessionLocal
    from app.db.models import PDF, VectorStore
    from app.services.pdf_service import PdfIndexData

    email = f"user_{uuid.uuid4().hex[:8]}@example.com"
    r = client.post(
        "/auth/register",
        json={"name": "Test User", "email": email, "password": "password123"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    r = client.post("/bots", headers=headers, json={"name": "Bot PDF", "description": "x"})
    assert r.status_code == 200, r.text
    bot_id = r.json()["id"]

    async def fake_load_and_prepare(*, bot_id: int, pdf_path: str, upload_uuid: str):
        pid = f"{bot_id}:{upload_uuid}::p::0"
        parent_chunks = {pid: "PARENT CONTEXT"}
        child_docs = [Document(page_content="CHILD TEXT", metadata={"child_id": pid})]
        bm25_items = [{"id": pid, "text": "CHILD TEXT"}]
        return PdfIndexData(
            parent_chunks=parent_chunks,
            child_docs=child_docs,
            bm25_items=bm25_items,
            parent_count=1,
            child_count=1,
        )

    async def fake_upsert_vectorstore(*args, **kwargs):
        return None

    monkeypatch.setattr(pdf_route, "load_and_prepare_pdf_index_data", fake_load_and_prepare)
    monkeypatch.setattr(pdf_route, "upsert_vectorstore", fake_upsert_vectorstore)

    r = client.post(
        f"/bots/{bot_id}/upload",
        headers=headers,
        files=[("files", ("test.pdf", _make_pdf_bytes(), "application/pdf"))],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["bot_id"] == bot_id
    assert body["parents_indexed"] == 1
    assert body["children_indexed"] == 1

    async def _counts():
        async with AsyncSessionLocal() as db:
            pdf_count = await db.scalar(select(func.count()).select_from(PDF).where(PDF.bot_id == bot_id))
            vs_count = await db.scalar(select(func.count()).select_from(VectorStore).where(VectorStore.bot_id == bot_id))
            return pdf_count, vs_count

    pdf_count, vs_count = asyncio.run(_counts())
    assert pdf_count >= 1
    assert vs_count == 1

