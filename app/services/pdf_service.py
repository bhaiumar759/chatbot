from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, UploadFile, status
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


@dataclass
class PdfIndexData:
    parent_chunks: dict[str, str]
    child_docs: list[Document]
    bm25_items: list[dict[str, Any]]
    parent_count: int
    child_count: int


def _ensure_pdf_filename(filename: str) -> None:
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are accepted.")


def _validate_upload_file(upload: UploadFile) -> None:
    _ensure_pdf_filename(upload.filename or "")
    # Starlette doesn't always provide content_length, so we rely on our size-limited save.
    if upload.content_type and upload.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid content type. Expected application/pdf.")


async def save_upload_file_limited(upload: UploadFile, destination_path: str, max_bytes: int) -> int:
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    total = 0
    with open(destination_path, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="PDF file too large.")
            f.write(chunk)
    return total


async def load_and_prepare_pdf_index_data(
    *,
    bot_id: int,
    pdf_path: str,
    upload_uuid: str,
) -> PdfIndexData:
    """
    Loads a PDF, splits into parent/context chunks (for the final prompt)
    and child chunks (for dense/sparse retrieval).
    """

    def _sync_build() -> PdfIndexData:
        docs = PyPDFLoader(pdf_path).load()

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)

        parent_docs = parent_splitter.split_documents(docs)

        parent_chunks: dict[str, str] = {}
        child_docs: list[Document] = []
        bm25_items: list[dict[str, Any]] = []

        for p_idx, p_doc in enumerate(parent_docs):
            p_id = f"{bot_id}:{upload_uuid}::p::{p_idx}"
            parent_chunks[p_id] = p_doc.page_content

            child_texts = child_splitter.split_text(p_doc.page_content)
            for child_text in child_texts:
                child_docs.append(Document(page_content=child_text, metadata={"child_id": p_id}))
                bm25_items.append({"id": p_id, "text": child_text})

        return PdfIndexData(
            parent_chunks=parent_chunks,
            child_docs=child_docs,
            bm25_items=bm25_items,
            parent_count=len(parent_docs),
            child_count=len(child_docs),
        )

    return await asyncio.to_thread(_sync_build)


def generate_upload_uuid() -> str:
    return uuid.uuid4().hex

