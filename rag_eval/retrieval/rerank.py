"""
retrieval/rerank.py — BGE cross-encoder reranking.

First-stage retrieval (dense or hybrid) is fast but approximate: it scores the query
and a chunk independently. A cross-encoder reads the query and chunk TOGETHER, so it
judges relevance far more accurately — at higher cost, which is why we only rerank a
small candidate pool (config.candidate_k) and keep the top_k. This is the reranker
on/off arm of the ablation.

Swappable with: any reranker exposing a (query, passage) -> score function.
"""
from __future__ import annotations

from functools import lru_cache

from config import settings
from rag_eval.retrieval.dense import Retrieved


@lru_cache(maxsize=1)
def _reranker():
    from FlagEmbedding import FlagReranker  # heavy import, lazy

    use_fp16 = settings.embedding_device.lower().startswith("cuda")
    return FlagReranker(settings.reranker_model, use_fp16=use_fp16)


def rerank(query: str, candidates: list[Retrieved], top_k: int) -> list[Retrieved]:
    """Re-score candidates with the cross-encoder and return the best top_k."""
    if not candidates:
        return []
    pairs = [[query, c.text] for c in candidates]
    scores = _reranker().compute_score(pairs, normalize=True)
    # compute_score returns a float for a single pair; normalize to a list.
    if not isinstance(scores, list):
        scores = [scores]
    for cand, score in zip(candidates, scores):
        cand.score = float(score)  # replace first-stage score with rerank score
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]
