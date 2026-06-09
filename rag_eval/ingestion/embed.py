"""
ingestion/embed.py — BGE-M3 embeddings (dense now; sparse ready for hybrid).

One model produces BOTH a dense vector and sparse lexical weights. Phase 1 uses
dense only; the Phase-3 hybrid ablation reuses embed_sparse() from the same model —
that single-model property is exactly why BGE-M3 was chosen. The model is large, so
it is loaded once and cached (first call downloads it into the HF cache).

Swappable with: any embedder, but hybrid needs a model that emits sparse weights too.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from config import settings


@lru_cache(maxsize=1)
def _model():
    from FlagEmbedding import BGEM3FlagModel  # imported lazily: heavy import

    # fp16 only helps on GPU; on CPU it would be slower/unsupported.
    use_fp16 = settings.embedding_device.lower().startswith("cuda")
    return BGEM3FlagModel(settings.embedding_model, use_fp16=use_fp16)


def embed_dense(texts: list[str]) -> np.ndarray:
    """Return an (n, dense_dim) float32 array of dense embeddings."""
    out = _model().encode(
        texts,
        batch_size=settings.embedding_batch_size,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return np.asarray(out["dense_vecs"], dtype=np.float32)


def embed_sparse(texts: list[str]) -> list[dict]:
    """Lexical weights {token_id: weight} per text — used by the Phase-3 hybrid ablation."""
    out = _model().encode(
        texts,
        batch_size=settings.embedding_batch_size,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return out["lexical_weights"]


def embed_both(texts: list[str]) -> tuple[np.ndarray, list[dict]]:
    """Dense vectors + sparse lexical weights in ONE pass (used at indexing time).

    Computing both together is the whole reason for BGE-M3: one model call yields the
    dense vector for semantic search and the sparse weights for lexical (BM25-like)
    search, so the hybrid index needs no second model.
    """
    out = _model().encode(
        texts,
        batch_size=settings.embedding_batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return np.asarray(out["dense_vecs"], dtype=np.float32), out["lexical_weights"]


def embed_query(text: str) -> np.ndarray:
    """Convenience: dense embedding for a single query string."""
    return embed_dense([text])[0]


def embed_query_sparse(text: str) -> dict:
    """Convenience: sparse lexical weights for a single query string."""
    return embed_sparse([text])[0]


def sparse_to_indices_values(weights: dict) -> tuple[list[int], list[float]]:
    """Convert BGE-M3 lexical weights {token_id: weight} to Qdrant sparse format."""
    indices: list[int] = []
    values: list[float] = []
    for token_id, weight in weights.items():
        w = float(weight)
        if w <= 0:
            continue
        indices.append(int(token_id))
        values.append(w)
    return indices, values
