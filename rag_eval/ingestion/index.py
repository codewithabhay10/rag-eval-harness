"""
ingestion/index.py — Qdrant local-mode index: client, collection, upsert.

Local mode persists to a folder (config.qdrant_path) — no server to run. The
collection is keyed by chunk size (config.collection_name) so the chunk-size
ablation can keep separate indexes side by side. Payload stores the chunk text +
metadata so retrieval can return citable sources without a second lookup.

Each point carries TWO named vectors from BGE-M3: a "dense" vector (semantic search)
and a "sparse" vector (lexical/BM25-like). Phase-1 dense retrieval uses only "dense";
the Phase-3 hybrid ablation fuses both. Storing both at index time means hybrid needs
no re-indexing or second model.
"""
from __future__ import annotations

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FilterSelector,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from config import settings
from rag_eval.ingestion.chunk import Chunk
from rag_eval.ingestion.embed import embed_both, sparse_to_indices_values

DENSE = "dense"
SPARSE = "sparse"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Single local-mode client. NOTE: only one process may open the folder at a time."""
    settings.ensure_dirs()
    return QdrantClient(path=str(settings.qdrant_path))


def recreate_collection(name: str) -> None:
    """Drop and create the collection (named dense + sparse vectors) from scratch.

    Qdrant LOCAL mode can retain points across delete_collection + create_collection
    (the on-disk segment isn't dropped), which silently duplicates the corpus on a
    rebuild. So after recreating we explicitly clear any points that survived.
    """
    client = get_client()
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE: VectorParams(size=settings.dense_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={SPARSE: SparseVectorParams()},
    )
    if client.count(name).count:  # defensive: force-empty if local mode kept old points
        client.delete(collection_name=name, points_selector=FilterSelector(filter=Filter()))


def scroll_all_chunks(collection: str | None = None) -> list[dict]:
    """Read every stored chunk payload back out of the index.

    Used by the eval test-set builder so the 'gold' context is guaranteed to be a
    chunk that retrieval can actually return (same text, same chunk_id).
    """
    client = get_client()
    name = collection or settings.collection_name
    if not client.collection_exists(name):
        raise RuntimeError(
            f"Collection '{name}' does not exist. Run ingestion first: "
            f"python -m rag_eval.ingestion.run"
        )
    payloads: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        payloads.extend(p.payload for p in points if p.payload)
        if offset is None:
            break
    return payloads


def index_chunks(chunks: list[Chunk], batch_size: int = 64) -> int:
    """Embed (dense + sparse) and upsert all chunks. Returns the number indexed."""
    client = get_client()
    name = settings.collection_name
    recreate_collection(name)

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        dense_vecs, sparse_weights = embed_both([c.text for c in batch])
        points = []
        for i, c in enumerate(batch):
            indices, values = sparse_to_indices_values(sparse_weights[i])
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        DENSE: dense_vecs[i].tolist(),
                        SPARSE: SparseVector(indices=indices, values=values),
                    },
                    payload={
                        "chunk_id": c.chunk_id,
                        "paper_id": c.paper_id,
                        "title": c.title,
                        "index": c.index,
                        "text": c.text,
                    },
                )
            )
        client.upsert(collection_name=name, points=points)
        total += len(points)
        print(f"      indexed {total}/{len(chunks)} chunks", end="\r")
    print()

    # Fail loudly if the index isn't exactly what we put in (e.g. stale duplicates):
    # a corrupt index would silently corrupt every metric downstream.
    stored = client.count(name).count
    if stored != total:
        raise RuntimeError(
            f"Index sanity check failed for '{name}': upserted {total} but collection "
            f"holds {stored}. Delete {settings.qdrant_path} and re-ingest."
        )
    return total
