from __future__ import annotations

from typing import Iterable

from google import genai

from app.core.config import settings

_client: genai.Client | None = None


def _require_gemini_key() -> str:
    if not settings.gemini_api_key or not settings.gemini_api_key.get_secret_value():
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return settings.gemini_api_key.get_secret_value()


def get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = _require_gemini_key()
        _client = genai.Client(api_key=api_key)
    return _client


def generate_hypothetical_answer(query: str) -> str:
    client = get_gemini_client()
    resp = client.models.generate_content(
        model=settings.gemini_model,
        contents=(
            "Write one concise factual paragraph that directly answers: "
            f"'{query}'. Use specific details. For document retrieval only."
        ),
    )
    return (resp.text or "").strip()


def stream_answer(prompt: str) -> Iterable[str]:
    client = get_gemini_client()
    response = client.models.generate_content_stream(
        model=settings.gemini_model,
        contents=prompt,
    )
    for chunk in response:
        text = getattr(chunk, "text", None)
        if text:
            yield text

