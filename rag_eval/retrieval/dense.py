"""
retrieval/dense.py — dense vector search over Qdrant.

Embeds the query with BGE-M3 (dense) and runs cosine search against the
chunk-size-keyed collection. This is the Phase-1 baseline; hybrid (dense + sparse)
and reranking are layered on top in Phase 3 via retriever.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import settings
from rag_eval.ingestion.embed import embed_query
from rag_eval.ingestion.index import get_client


@dataclass
class Retrieved:
    chunk_id: str
    paper_id: str
    title: str
    text: str
    score: float


def dense_search(query: str, k: int) -> list[Retrieved]:
    client = get_client()
    vector = embed_query(query).tolist()
    response = client.query_points(
        collection_name=settings.collection_name,
        query=vector,
        limit=k,
        with_payload=True,
    )
    results: list[Retrieved] = []
    for point in response.points:
        payload = point.payload or {}
        results.append(
            Retrieved(
                chunk_id=payload.get("chunk_id", ""),
                paper_id=payload.get("paper_id", ""),
                title=payload.get("title", ""),
                text=payload.get("text", ""),
                score=float(point.score),
            )
        )
    return results
