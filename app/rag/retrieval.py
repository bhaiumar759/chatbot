from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from app.rag.gemini_client import generate_hypothetical_answer
from app.rag.rrf import reciprocal_rank_fusion
from app.rag.vectorstore_manager import VectorStoreState


async def hybrid_search(
    query: str,
    state: VectorStoreState,
    top_k: int = 6,
) -> list[str]:
    """
    Hybrid retrieval (dense + sparse) with RRF, then maps back to parent contexts.
    """
    hyde_text = await asyncio.to_thread(generate_hypothetical_answer, query)

    dense_docs = state.faiss.similarity_search(hyde_text, k=top_k * 2)
    dense_ids = [
        d.metadata.get("child_id") if hasattr(d, "metadata") else None
        for d in dense_docs
    ]
    dense_ids = [x for x in dense_ids if isinstance(x, str)]

    sparse_ids: list[str] = []
    if state.bm25_index is not None and state.bm25_corpus:
        tokens = query.lower().split()
        bm25_scores = state.bm25_index.get_scores(tokens)
        top_idx = np.argsort(bm25_scores)[::-1][: top_k * 2]
        sparse_ids = [state.bm25_corpus[i]["id"] for i in top_idx]

    fused_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)[:top_k]

    # Map child chunks back to parent contexts.
    seen_texts: set[str] = set()
    contexts: list[str] = []
    for doc_id in fused_ids:
        text = state.parent_chunks.get(doc_id, "")
        if text and text not in seen_texts:
            seen_texts.add(text)
            contexts.append(text)
    return contexts

