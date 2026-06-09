"""
ingestion/index.py — Qdrant local-mode index: client, collection, upsert.

Local mode persists to a folder (config.qdrant_path) — no server to run. The
collection is keyed by chunk size (config.collection_name) so the chunk-size
ablation can keep separate indexes side by side. Payload stores the chunk text +
metadata so retrieval can return citable sources without a second lookup.
"""
from __future__ import annotations

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import settings
from rag_eval.ingestion.chunk import Chunk
from rag_eval.ingestion.embed import embed_dense


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Single local-mode client. NOTE: only one process may open the folder at a time."""
    settings.ensure_dirs()
    return QdrantClient(path=str(settings.qdrant_path))


def recreate_collection(name: str) -> None:
    """Drop and create the collection so re-ingestion starts from a clean state."""
    client = get_client()
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=settings.dense_dim, distance=Distance.COSINE),
    )


def index_chunks(chunks: list[Chunk], batch_size: int = 64) -> int:
    """Embed and upsert all chunks into the chunk-size-keyed collection. Returns count."""
    client = get_client()
    name = settings.collection_name
    recreate_collection(name)

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_dense([c.text for c in batch])
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i].tolist(),
                payload={
                    "chunk_id": c.chunk_id,
                    "paper_id": c.paper_id,
                    "title": c.title,
                    "index": c.index,
                    "text": c.text,
                },
            )
            for i, c in enumerate(batch)
        ]
        client.upsert(collection_name=name, points=points)
        total += len(points)
        print(f"      indexed {total}/{len(chunks)} chunks", end="\r")
    print()
    return total
