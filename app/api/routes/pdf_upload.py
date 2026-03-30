from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.schemas import BotResponse
from app.core.config import settings
from app.db.database import get_db
from app.db.models import Bot, PDF, User, VectorStore
from app.rag.vectorstore_manager import get_bot_vector_paths, upsert_vectorstore
from app.services.bot_service import get_bot_for_user
from app.services.pdf_service import (
    generate_upload_uuid,
    load_and_prepare_pdf_index_data,
    save_upload_file_limited,
)

from app.services.pdf_service import _validate_upload_file  # internal helper


router = APIRouter(tags=["pdfs"])


@router.post("/bots/{bot_id}/upload")
async def upload_pdfs(
    request: Request,
    bot_id: int,
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    bot = await get_bot_for_user(db, bot_id, current_user)

    upload_dir = os.path.join(settings.uploads_root, str(bot_id))
    os.makedirs(upload_dir, exist_ok=True)

    all_parent_chunks: dict[str, str] = {}
    all_child_docs = []
    all_bm25_items: list[dict] = []
    total_parents = 0
    total_children = 0

    # Save and pre-process each PDF.
    for upload in files:
        _validate_upload_file(upload)
        upload_uuid = generate_upload_uuid()
        safe_name = os.path.basename(upload.filename or "document.pdf")
        saved_path = os.path.join(upload_dir, f"{upload_uuid}__{safe_name}")

        await save_upload_file_limited(upload, saved_path, settings.max_pdf_bytes)

        # Persist PDF metadata
        pdf_row = PDF(
            bot_id=bot.id,
            filename=safe_name,
            filepath=saved_path,
        )
        db.add(pdf_row)
        await db.flush()  # keep transaction consistent; id may be used later

        index_data = await load_and_prepare_pdf_index_data(
            bot_id=bot.id,
            pdf_path=saved_path,
            upload_uuid=upload_uuid,
        )

        all_parent_chunks.update(index_data.parent_chunks)
        all_child_docs.extend(index_data.child_docs)
        all_bm25_items.extend(index_data.bm25_items)
        total_parents += index_data.parent_count
        total_children += index_data.child_count

    # Update vector stores on disk for this bot.
    try:
        await upsert_vectorstore(
            bot_id=bot.id,
            child_lc_docs=all_child_docs,
            new_parent_chunks=all_parent_chunks,
            new_bm25_items=all_bm25_items,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector indexing failed: {e}")

    # Upsert VectorStores row.
    existing_vs = await db.scalar(select(VectorStore).where(VectorStore.bot_id == bot.id))
    paths = get_bot_vector_paths(bot.id)
    if existing_vs:
        existing_vs.faiss_file_path = paths["faiss_file_path"]
        existing_vs.bm25_file_path = paths["bm25_file_path"]
        existing_vs.parent_chunks_file_path = paths["parent_chunks_file_path"]
        existing_vs.updated_at = datetime.now(timezone.utc)
    else:
        db.add(
            VectorStore(
                bot_id=bot.id,
                faiss_file_path=paths["faiss_file_path"],
                bm25_file_path=paths["bm25_file_path"],
                parent_chunks_file_path=paths["parent_chunks_file_path"],
                updated_at=datetime.now(timezone.utc),
            )
        )

    await db.commit()

    return {
        "message": "Uploaded and indexed ✅",
        "bot_id": bot.id,
        "parents_indexed": total_parents,
        "children_indexed": total_children,
    }

