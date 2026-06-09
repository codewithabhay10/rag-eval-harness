"""
generation/llm.py — LLM provider factory (local-first, API opt-in).

Returns a LangChain chat model selected by config.llm_provider so the rest of the
codebase — and RAGAS in Phase 2 — stays provider-agnostic. Default is Ollama (no
paid key). Groq and Gemini are opt-in and require an API key in .env; we fail loudly
with a clear message if the key is missing rather than silently degrading.

Swappable with: any LangChain BaseChatModel.
"""
from __future__ import annotations

from functools import lru_cache

from config import settings


@lru_cache(maxsize=1)
def get_llm():
    provider = settings.llm_provider

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
            num_ctx=settings.ollama_num_ctx,  # avoid silent prompt truncation (see config)
        )

    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError(
                "llm_provider='groq' but GROQ_API_KEY is not set in .env"
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if provider == "gemini":
        if not settings.google_api_key:
            raise RuntimeError(
                "llm_provider='gemini' but GOOGLE_API_KEY is not set in .env"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
        )

    raise ValueError(f"Unknown llm_provider: {provider!r}")
