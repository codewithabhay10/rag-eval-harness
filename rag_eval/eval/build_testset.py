"""
eval/build_testset.py — generate a synthetic QA test set from the indexed corpus.

For each sampled chunk we ask the LLM to write one self-contained question that is
answerable from THAT chunk alone, plus a concise ground-truth answer. The chunk is
the gold context, so every item carries everything needed to score generation
(RAGAS) and retrieval (Recall@k / MRR).

Design notes:
- We sample chunks straight out of Qdrant so the gold context is guaranteed to be
  retrievable (identical text + chunk_id).
- Sampling is round-robin across papers (not uniform over chunks) so a single long
  paper can't dominate the test set.
- The LLM may reply {"skip": true} for unsuitable passages (reference lists, table
  fragments, math-only), which we drop — this keeps the set answerable.

Usage:
    python -m rag_eval.eval.build_testset                 # ~100 pairs (config default)
    python -m rag_eval.eval.build_testset --num-questions 30
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from rag_eval.eval.schema import TestItem, save_testset
from rag_eval.generation.llm import get_llm
from rag_eval.ingestion.index import scroll_all_chunks

MIN_CHARS = 300  # passages shorter than this rarely yield a good standalone question

SYSTEM_PROMPT = (
    "You create evaluation data for a retrieval-QA system over research papers. "
    "You always respond with a single JSON object and nothing else."
)

USER_TEMPLATE = (
    "Given ONE passage from a research paper, write a single self-contained question "
    "that can be answered using ONLY this passage, plus a concise factual answer drawn "
    "only from the passage.\n\n"
    "Rules:\n"
    "- The question must be answerable from this passage alone.\n"
    "- The question must be standalone: do NOT say 'the passage', 'this text', "
    "'the authors', 'the paper', or 'the figure'. Name the actual subject.\n"
    "- Prefer specific, factual questions (a definition, method, number, or finding).\n"
    "- If the passage is unsuitable (reference list, table/figure fragment, math only, "
    "too little content), respond with {{\"skip\": true}}.\n"
    "- Output ONLY JSON: {{\"question\": \"...\", \"answer\": \"...\"}} or {{\"skip\": true}}.\n\n"
    'Passage:\n"""\n{chunk}\n"""'
)


def _balanced_order(chunks: list[dict], seed: int) -> list[dict]:
    """Round-robin chunks across papers so coverage is even, not length-weighted."""
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_paper[c.get("paper_id", "?")].append(c)

    rnd = random.Random(seed)
    for bucket in by_paper.values():
        rnd.shuffle(bucket)
    papers = list(by_paper.keys())
    rnd.shuffle(papers)

    ordered: list[dict] = []
    while any(by_paper[p] for p in papers):
        for p in papers:
            if by_paper[p]:
                ordered.append(by_paper[p].pop())
    return ordered


def _parse_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply, tolerating code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build(num_questions: int) -> list[TestItem]:
    print(f"Reading chunks from Qdrant collection '{settings.collection_name}' ...")
    chunks = scroll_all_chunks()
    candidates = [c for c in _balanced_order(chunks, settings.random_seed)
                  if len(c.get("text", "")) >= MIN_CHARS]
    print(f"  {len(chunks)} chunks indexed, {len(candidates)} long enough to use.")
    if not candidates:
        raise SystemExit("No usable chunks — ingest more papers or lower MIN_CHARS.")

    llm = get_llm()
    items: list[TestItem] = []
    skipped = 0
    for chunk in candidates:
        if len(items) >= num_questions:
            break
        prompt = USER_TEMPLATE.format(chunk=chunk["text"])
        resp = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        data = _parse_json(content)
        if not data or data.get("skip") or not data.get("question") or not data.get("answer"):
            skipped += 1
            continue
        items.append(
            TestItem(
                id=f"q{len(items):04d}",
                question=str(data["question"]).strip(),
                ground_truth=str(data["answer"]).strip(),
                reference_contexts=[chunk["text"]],
                source_chunk_ids=[chunk["chunk_id"]],
                source_paper_id=chunk.get("paper_id", "?"),
            )
        )
        print(f"  generated {len(items)}/{num_questions} (skipped {skipped})", end="\r")
    print()
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a synthetic QA test set.")
    parser.add_argument("--num-questions", type=int, default=settings.testset_size)
    args = parser.parse_args()

    items = build(args.num_questions)
    if not items:
        raise SystemExit("No test items generated — check the LLM provider/key.")
    save_testset(items, settings.testset_path)
    print(f"Saved {len(items)} QA pairs to {settings.testset_path}")
    print("Next: review them with  python -m rag_eval.eval.review_testset")


if __name__ == "__main__":
    main()
