"""
ingestion/run.py — end-to-end ingestion: fetch -> parse -> chunk -> embed -> index.

Usage (from the project root):
    python -m rag_eval.ingestion.run --num-papers 50

Start small (50) to validate the pipeline end to end, then scale up. Re-running is
cheap: PDFs are cached on disk; only embedding + indexing repeat.
"""
from __future__ import annotations

import argparse

from config import settings
from rag_eval.ingestion.chunk import chunk_paper
from rag_eval.ingestion.corpus import get_corpus_source, save_corpus_meta
from rag_eval.ingestion.index import index_chunks
from rag_eval.ingestion.parse import extract_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the corpus into Qdrant.")
    parser.add_argument(
        "--num-papers",
        type=int,
        default=settings.num_papers,
        help=f"How many papers to fetch (default {settings.num_papers}).",
    )
    args = parser.parse_args()

    settings.ensure_dirs()

    print(f"[1/4] Fetching up to {args.num_papers} papers from "
          f"'{settings.corpus_source}' (query: {settings.arxiv_query!r}) ...")
    source = get_corpus_source()
    papers = source.fetch(args.num_papers)
    save_corpus_meta(papers)
    print(f"      got {len(papers)} papers")
    if not papers:
        raise SystemExit("No papers fetched — check your network / arXiv query.")

    print("[2/4] Parsing + chunking ...")
    all_chunks = []
    for p in papers:
        text = extract_text(p.pdf_path)
        if not text:
            print(f"      WARN: no text extracted from {p.paper_id}")
            continue
        all_chunks.extend(chunk_paper(p.paper_id, p.title, text))
    print(f"      produced {len(all_chunks)} chunks "
          f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})")
    if not all_chunks:
        raise SystemExit("No chunks produced — aborting before indexing.")

    print(f"[3/4] Embedding with {settings.embedding_model} "
          f"(first run downloads the model, ~2GB) ...")
    print(f"[4/4] Indexing into Qdrant collection '{settings.collection_name}' ...")
    n = index_chunks(all_chunks)

    print(f"\nDone. Indexed {n} chunks from {len(papers)} papers "
          f"into '{settings.collection_name}'.")
    print("Next: python -m rag_eval.generation.ask \"<your question>\"")


if __name__ == "__main__":
    main()
