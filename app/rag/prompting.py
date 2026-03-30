from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a precise document assistant. Answer ONLY from the CONTEXT below.\n\n"
    "RULES:\n"
    "1. Never use outside knowledge.\n"
    "2. If context is insufficient say: "
    "'The document does not contain enough information to answer this.'\n"
    "3. Quote dates, numbers, and names directly from context.\n"
    "4. No padding or filler sentences.\n"
    "5. For summaries, cover every key point in the context.\n"
)


def build_prompt(context_chunks: list[str], history: str, query: str) -> str:
    context = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(context_chunks)
    )
    return (
        f"{SYSTEM_PROMPT}\n"
        f"CONTEXT:\n{context}\n\n"
        f"CONVERSATION HISTORY:\n{history}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"ANSWER:"
    )

