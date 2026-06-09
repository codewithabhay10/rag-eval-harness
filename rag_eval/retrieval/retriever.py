"""
retrieval/retriever.py — the single retrieval entry point; honors the config knobs.

retrieve() dispatches on config.retrieval_strategy (dense | hybrid) and applies the
cross-encoder reranker when config.use_reranker is set. Keeping the switch here means
callers (generation, eval) never change as ablations flip strategy/reranker on and
off — which is exactly what the Phase-3 ablation does.

Pipeline: first-stage retrieve (a wider candidate pool if reranking) -> optional
rerank -> top_k.
"""
from __future__ import annotations

from config import settings
from rag_eval.retrieval.dense import Retrieved, dense_search
from rag_eval.retrieval.hybrid import hybrid_search


def retrieve(query: str, top_k: int | None = None) -> list[Retrieved]:
    k = top_k if top_k is not None else settings.top_k

    # When reranking, pull a wider candidate pool first, then trim to k after rerank.
    fetch_k = settings.candidate_k if settings.use_reranker else k

    if settings.retrieval_strategy == "dense":
        results = dense_search(query, k=fetch_k)
    elif settings.retrieval_strategy == "hybrid":
        results = hybrid_search(query, k=fetch_k, candidate_k=settings.candidate_k)
    else:
        raise ValueError(f"Unknown retrieval_strategy: {settings.retrieval_strategy!r}")

    if settings.use_reranker:
        from rag_eval.retrieval.rerank import rerank  # lazy: avoids loading reranker unless used

        results = rerank(query, results, top_k=k)

    return results[:k]
