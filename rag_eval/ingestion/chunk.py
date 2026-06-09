"""
ingestion/chunk.py — split paper text into overlapping chunks.

chunk_size / chunk_overlap are CONFIG values because chunk size is an ablation axis
(256 / 512 / 1024). Size is measured in TOKENS of the embedding model's own
tokenizer, so the number maps directly to what BGE-M3 sees (and stays under its
context limit). We split recursively on paragraph -> sentence -> word boundaries
(RecursiveCharacterTextSplitter) so chunks don't get cut mid-sentence, and overlap
preserves context that straddles a boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config import settings


@dataclass
class Chunk:
    chunk_id: str  # f"{paper_id}::{index}" — stable, human-readable citation handle
    paper_id: str
    title: str
    index: int
    text: str


@lru_cache(maxsize=1)
def _tokenizer():
    # BGE-M3 uses an XLM-RoBERTa tokenizer; we reuse it purely to count tokens.
    return AutoTokenizer.from_pretrained(settings.embedding_model)


@lru_cache(maxsize=8)
def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _tokenizer(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_paper(paper_id: str, title: str, text: str) -> list[Chunk]:
    splitter = _splitter(settings.chunk_size, settings.chunk_overlap)
    chunks: list[Chunk] = []
    for i, piece in enumerate(splitter.split_text(text)):
        piece = piece.strip()
        if len(piece) < 30:  # drop near-empty fragments (page numbers, stray lines)
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{paper_id}::{i}",
                paper_id=paper_id,
                title=title,
                index=i,
                text=piece,
            )
        )
    return chunks
