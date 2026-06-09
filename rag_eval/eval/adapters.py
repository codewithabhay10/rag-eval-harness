"""
eval/adapters.py — a LangChain Embeddings adapter around BGE-M3.

RAGAS needs a LangChain-style embeddings object (e.g. answer relevancy embeds the
question and answer). We deliberately reuse the SAME BGE-M3 model the retriever uses,
so eval embeds text exactly like the system under test — no second embedding model,
no drift between what we index and what we score.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from rag_eval.ingestion.embed import embed_dense, embed_query


class BGEEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in embed_dense(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text).tolist()
