"""Answer generation: DeepSeek LLM with graceful extractive fallback.

DeepSeek exposes an OpenAI-compatible API, so we use the `openai` SDK with a
custom base_url. If no API key is configured — or the provider call fails —
we degrade to an *extractive* answer assembled from the top retrieved passages,
so the demo never dies in front of an interviewer.
"""

import time

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)
from .retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are a precise technical assistant for an AI/ML knowledge base.
Answer the user's question using ONLY the provided context passages.
Rules:
- Cite sources inline using their bracketed numbers, e.g. [1], [2].
- If the context does not contain the answer, say "I don't know based on the provided documents." Do not invent facts.
- Be concise and technically accurate."""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] (source: {c.source}" + (f", section: {c.section})" if c.section else ")")
        lines.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(lines)


def _llm_client():
    from openai import OpenAI

    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=LLM_TIMEOUT_SECONDS,
    )


def _extractive_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Fallback: quote the most relevant passages instead of generating text."""
    if not chunks:
        return "I don't know based on the provided documents."
    parts = ["(LLM unavailable — showing the most relevant passages instead)\n"]
    for i, c in enumerate(chunks[:3], start=1):
        snippet = c.text.strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        parts.append(f"[{i}] From {c.source} — {c.section}:\n{snippet}")
    return "\n\n".join(parts)


def generate_answer(
    question: str, chunks: list[RetrievedChunk]
) -> tuple[str, str, float]:
    """Generate a grounded answer.

    Returns (answer, mode, seconds) where mode is "llm" or "extractive".
    """
    t0 = time.perf_counter()

    if not DEEPSEEK_API_KEY:
        return _extractive_answer(question, chunks), "extractive", time.perf_counter() - t0

    context = _format_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        client = _llm_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response.choices[0].message.content or ""
        return answer.strip(), "llm", time.perf_counter() - t0
    except Exception as exc:  # provider down / quota exhausted / network error
        fallback = _extractive_answer(question, chunks)
        fallback += f"\n\n(Provider error: {type(exc).__name__})"
        return fallback, "extractive", time.perf_counter() - t0
