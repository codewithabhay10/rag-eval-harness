"""
eval/retrieval_metrics.py — retrieval-quality metrics (no LLM needed).

These score the retriever directly against the gold chunk id(s) for each question,
which is why we kept chunk ids in the test set. They complement RAGAS (which scores
the answer + contexts) by measuring retrieval in isolation.

- Recall@k: fraction of gold chunks found in the top-k retrieved.
- MRR: mean reciprocal rank of the first gold chunk across questions.
"""
from __future__ import annotations

from statistics import mean


def recall_at_k(gold_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    hits = gold & set(retrieved_ids[:k])
    return len(hits) / len(gold)


def reciprocal_rank(gold_ids: list[str], retrieved_ids: list[str]) -> float:
    gold = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def aggregate(
    per_question: list[tuple[list[str], list[str]]], k: int
) -> dict[str, float]:
    """per_question is a list of (gold_chunk_ids, retrieved_chunk_ids) tuples."""
    if not per_question:
        return {f"recall@{k}": 0.0, "mrr": 0.0}
    recalls = [recall_at_k(g, r, k) for g, r in per_question]
    rrs = [reciprocal_rank(g, r) for g, r in per_question]
    return {f"recall@{k}": mean(recalls), "mrr": mean(rrs)}
