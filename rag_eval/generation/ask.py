"""
generation/ask.py — CLI sanity check for the baseline RAG pipeline.

Usage (from the project root):
    python -m rag_eval.generation.ask "What is contrastive learning?"

Prints the grounded answer plus the cited source chunks (title, paper id, chunk id,
score, snippet) so you can eyeball whether retrieval and citations look right.
"""
from __future__ import annotations

import argparse

from rag_eval.generation.generate import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG system a question.")
    parser.add_argument("question", type=str, help="The question to ask.")
    args = parser.parse_args()

    result = answer_question(args.question)

    print("\n=== ANSWER ===")
    print(result.answer)

    print("\n=== SOURCES ===")
    if not result.sources:
        print("(none retrieved)")
    for i, s in enumerate(result.sources, 1):
        print(
            f"[{i}] {s.title}\n"
            f"     paper={s.paper_id}  chunk={s.chunk_id}  score={s.score:.3f}"
        )
        snippet = s.text[:200].replace("\n", " ")
        print(f"     {snippet}...")


if __name__ == "__main__":
    main()
