from __future__ import annotations

import asyncio
import os
import pickle
from dataclasses import dataclass
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.rag.embeddings import embedding_fn


@dataclass
class VectorStoreState:
    faiss: FAISS
    bm25_index: BM25Okapi
    bm25_corpus: list[dict[str, Any]]
    parent_chunks: dict[str, str]


def _save_pickle(path: str, obj: object) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: str) -> Any | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_bot_vector_dir(bot_id: int) -> str:
    return os.path.join(settings.vectorstore_root, str(bot_id))


def get_bot_vector_paths(bot_id: int) -> dict[str, str]:
    vector_dir = get_bot_vector_dir(bot_id)
    return {
        "faiss_file_path": os.path.join(vector_dir, "index.faiss"),
        "bm25_file_path": os.path.join(vector_dir, "bm25.pkl"),
        "parent_chunks_file_path": os.path.join(vector_dir, "parent_chunks.pkl"),
    }


def _faiss_index_exists(bot_vector_dir: str) -> bool:
    return os.path.exists(os.path.join(bot_vector_dir, "index.faiss"))


# Simple in-process cache (sufficient for a single FastAPI worker).
_state_cache: dict[int, VectorStoreState] = {}
_locks: dict[int, asyncio.Lock] = {}


async def get_vectorstore_state(bot_id: int) -> VectorStoreState:
    if bot_id in _state_cache:
        return _state_cache[bot_id]

    lock = _locks.setdefault(bot_id, asyncio.Lock())
    async with lock:
        if bot_id in _state_cache:
            return _state_cache[bot_id]
        state = await asyncio.to_thread(_load_vectorstore_state_sync, bot_id)
        _state_cache[bot_id] = state
        return state


def _load_vectorstore_state_sync(bot_id: int) -> VectorStoreState:
    vector_dir = get_bot_vector_dir(bot_id)
    paths = get_bot_vector_paths(bot_id)

    if not _faiss_index_exists(vector_dir):
        raise FileNotFoundError(f"FAISS index missing for bot_id={bot_id}")

    faiss = FAISS.load_local(
        vector_dir,
        embedding_fn,
        allow_dangerous_deserialization=True,
    )

    bm25_data = _load_pickle(paths["bm25_file_path"])
    if not bm25_data:
        # Should not happen if we created the directory via upload; still handle.
        raise FileNotFoundError(f"BM25 data missing for bot_id={bot_id}")
    bm25_index, bm25_corpus = bm25_data

    parent_chunks = _load_pickle(paths["parent_chunks_file_path"]) or {}

    return VectorStoreState(
        faiss=faiss,
        bm25_index=bm25_index,
        bm25_corpus=bm25_corpus,
        parent_chunks=parent_chunks,
    )


async def upsert_vectorstore(
    bot_id: int,
    child_lc_docs: list[Document],
    new_parent_chunks: dict[str, str],
    new_bm25_items: list[dict[str, Any]],
) -> VectorStoreState:
    """
    Updates (or creates) FAISS + BM25 + parent-chunks for a bot.
    """
    lock = _locks.setdefault(bot_id, asyncio.Lock())
    async with lock:
        state = await asyncio.to_thread(
            _upsert_vectorstore_sync,
            bot_id,
            child_lc_docs,
            new_parent_chunks,
            new_bm25_items,
        )
        _state_cache[bot_id] = state
        return state


def _upsert_vectorstore_sync(
    bot_id: int,
    child_lc_docs: list[Document],
    new_parent_chunks: dict[str, str],
    new_bm25_items: list[dict[str, Any]],
) -> VectorStoreState:
    vector_dir = get_bot_vector_dir(bot_id)
    os.makedirs(vector_dir, exist_ok=True)
    paths = get_bot_vector_paths(bot_id)

    faiss: FAISS
    if _faiss_index_exists(vector_dir):
        faiss = FAISS.load_local(
            vector_dir,
            embedding_fn,
            allow_dangerous_deserialization=True,
        )
        if child_lc_docs:
            faiss.add_documents(child_lc_docs)
    else:
        if not child_lc_docs:
            raise ValueError("Cannot create FAISS index without documents.")
        faiss = FAISS.from_documents(child_lc_docs, embedding_fn)

    # BM25
    bm25_data = _load_pickle(paths["bm25_file_path"])
    if bm25_data:
        bm25_index, bm25_corpus = bm25_data
    else:
        bm25_index, bm25_corpus = None, []

    bm25_corpus.extend(new_bm25_items)
    # Rebuild BM25 index from the full corpus.
    bm25_index = BM25Okapi([item["text"].lower().split() for item in bm25_corpus])

    # Parent chunks
    parent_chunks = _load_pickle(paths["parent_chunks_file_path"]) or {}
    parent_chunks.update(new_parent_chunks)

    # Persist
    faiss.save_local(vector_dir)
    _save_pickle(paths["bm25_file_path"], (bm25_index, bm25_corpus))
    _save_pickle(paths["parent_chunks_file_path"], parent_chunks)

    return VectorStoreState(
        faiss=faiss,
        bm25_index=bm25_index,
        bm25_corpus=bm25_corpus,
        parent_chunks=parent_chunks,
    )

