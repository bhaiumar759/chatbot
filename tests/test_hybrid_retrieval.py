import types

import numpy as np
import pytest
from langchain_core.documents import Document

import app.rag.retrieval as retrieval


@pytest.mark.asyncio
async def test_hybrid_retrieval_rrf_order():
    # Patch HyDE so we don't call Gemini.
    retrieval.generate_hypothetical_answer = lambda q: "hyde"

    class DummyFAISS:
        def similarity_search(self, text: str, k: int):
            # Ensure dense_ids = [c1, c2, c3, c4] for k=4
            return [
                Document(page_content="CHILD1", metadata={"child_id": "c1"}),
                Document(page_content="CHILD2", metadata={"child_id": "c2"}),
                Document(page_content="CHILD3", metadata={"child_id": "c3"}),
                Document(page_content="CHILD4", metadata={"child_id": "c4"}),
            ]

    class DummyBM25:
        def get_scores(self, tokens):
            # bm25_corpus indices correspond to ids: [c3, c1, c2, c4]
            # Descending -> [c2, c4, c1, c3]
            return np.array([0.1, 0.3, 0.9, 0.6])

    state = types.SimpleNamespace(
        faiss=DummyFAISS(),
        bm25_index=DummyBM25(),
        bm25_corpus=[
            {"id": "c3", "text": "t3"},
            {"id": "c1", "text": "t1"},
            {"id": "c2", "text": "t2"},
            {"id": "c4", "text": "t4"},
        ],
        parent_chunks={"c1": "P1", "c2": "P2", "c3": "P3", "c4": "P4"},
    )

    contexts = await retrieval.hybrid_search(query="What?", state=state, top_k=2)
    assert contexts == ["P2", "P1"]

