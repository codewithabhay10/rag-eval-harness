"""
retrieval/hybrid.py — hybrid (dense + sparse) search with RRF fusion.

Dense vectors capture semantic similarity; sparse (BGE-M3 lexical) vectors capture
exact term overlap (think BM25). Hybrid runs both and fuses the two ranked lists with
Reciprocal Rank Fusion — Qdrant does this natively via a prefetch + FusionQuery, so
we don't hand-roll the fusion math. This is the "hybrid" arm of the retrieval ablation.

Why late fusion (RRF) rather than mixing scores: dense cosine scores and sparse dot
scores live on different scales, so combining raw scores is unreliable; RRF combines
by RANK, which is scale-free and robust.
"""
from __future__ import annotations

from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from config import settings
from rag_eval.ingestion.embed import embed_query, embed_query_sparse, sparse_to_indices_values
from rag_eval.ingestion.index import DENSE, SPARSE, get_client
from rag_eval.retrieval.dense import Retrieved, to_retrieved


def hybrid_search(query: str, k: int, candidate_k: int | None = None) -> list[Retrieved]:
    client = get_client()
    prefetch_limit = candidate_k or max(k, settings.candidate_k)

    dense_vec = embed_query(query).tolist()
    indices, values = sparse_to_indices_values(embed_query_sparse(query))
    sparse_vec = SparseVector(indices=indices, values=values)

    response = client.query_points(
        collection_name=settings.collection_name,
        prefetch=[
            Prefetch(query=dense_vec, using=DENSE, limit=prefetch_limit),
            Prefetch(query=sparse_vec, using=SPARSE, limit=prefetch_limit),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return to_retrieved(response.points)
