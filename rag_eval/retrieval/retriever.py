"""
retrieval/retriever.py — the single retrieval entry point; honors the config knobs.

retrieve() dispatches on config.retrieval_strategy (dense | hybrid) and applies the
reranker when config.use_reranker is set. Keeping the switch here means callers
(generation, eval) never change as ablations flip strategy/reranker on and off.

Phase 1 implements dense only. Hybrid and reranking arrive in Phase 3; until then
they fail loudly rather than silently degrading and corrupting eval results.
"""
from __future__ import annotations

from config import settings
from rag_eval.retrieval.dense import Retrieved, dense_search


def retrieve(query: str, top_k: int | None = None) -> list[Retrieved]:
    k = top_k if top_k is not None else settings.top_k

    # When reranking is on, fetch a wider candidate pool first, then trim after rerank.
    fetch_k = settings.candidate_k if settings.use_reranker else k

    if settings.retrieval_strategy == "dense":
        results = dense_search(query, k=fetch_k)
    elif settings.retrieval_strategy == "hybrid":
        raise NotImplementedError(
            "retrieval_strategy='hybrid' arrives in Phase 3. Set it back to 'dense'."
        )
    else:
        raise ValueError(f"Unknown retrieval_strategy: {settings.retrieval_strategy!r}")

    if settings.use_reranker:
        raise NotImplementedError(
            "use_reranker=true arrives in Phase 3. Set it back to false."
        )

    return results[:k]
