from __future__ import annotations

from langchain_core.embeddings import Embeddings

from google.genai import types

from app.rag.gemini_client import get_gemini_client


class GeminiEmbedding(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        client = get_gemini_client()
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        if not result.embeddings:
            raise RuntimeError("Gemini embedding response was empty.")
        return list(result.embeddings[0].values)


embedding_fn = GeminiEmbedding()

