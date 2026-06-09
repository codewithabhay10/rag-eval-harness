"""
eval/run.py — evaluate the CURRENT pipeline config against the test set.

Flow: for each test question, run the real pipeline (retrieve + generate), collect
the answer + retrieved contexts + retrieved chunk ids, then score with RAGAS
(generation quality) and with our retrieval metrics (Recall@k / MRR). Results are
printed as a table and saved to results/ with a snapshot of the config that produced
them — so every number is traceable to a configuration (and Phase 3 can compare runs).

run_eval() returns the metrics dict so the Phase-3 ablation runner can reuse it.

Usage:
    python -m rag_eval.eval.run                # full kept test set
    python -m rag_eval.eval.run --limit 10     # quick smoke run
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from config import settings
from rag_eval.eval.ragas_eval import score_with_ragas
from rag_eval.eval.retrieval_metrics import aggregate
from rag_eval.eval.schema import kept_items, load_testset
from rag_eval.generation.generate import answer_question


def config_snapshot() -> dict:
    """The knobs that define this run, recorded alongside its metrics."""
    return {
        "retrieval_strategy": settings.retrieval_strategy,
        "use_reranker": settings.use_reranker,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "candidate_k": settings.candidate_k,
        "embedding_model": settings.embedding_model,
        "llm_provider": settings.llm_provider,
        "use_agentic": settings.use_agentic,
        "collection": settings.collection_name,
    }


def run_eval(limit: int | None = None) -> dict:
    items = kept_items(load_testset(settings.testset_path))
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit("Test set is empty (no kept items). Build/review it first.")

    print(f"Running pipeline over {len(items)} questions "
          f"(strategy={settings.retrieval_strategy}, reranker={settings.use_reranker}, "
          f"chunk={settings.chunk_size}, top_k={settings.top_k}) ...")

    samples: list[dict] = []
    retrieval_pairs: list[tuple[list[str], list[str]]] = []
    for n, it in enumerate(items, 1):
        ans = answer_question(it.question)
        retrieved_ids = [s.chunk_id for s in ans.sources]
        retrieved_texts = [s.text for s in ans.sources]
        samples.append(
            {
                "user_input": it.question,
                "response": ans.answer,
                "retrieved_contexts": retrieved_texts or [""],
                "reference": it.ground_truth,
                "reference_contexts": it.reference_contexts,
            }
        )
        retrieval_pairs.append((it.source_chunk_ids, retrieved_ids))
        print(f"  answered {n}/{len(items)}", end="\r")
    print()

    print("Scoring generation with RAGAS (this calls the LLM judge per item) ...")
    gen_scores, details = score_with_ragas(samples)

    retr_scores = aggregate(retrieval_pairs, k=settings.top_k)
    metrics = {**gen_scores, **retr_scores}

    _report(metrics, details, len(items))
    return metrics


def _report(metrics: dict, details: pd.DataFrame, n: int) -> None:
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": stamp,
        "n_questions": n,
        "config": config_snapshot(),
        "metrics": metrics,
    }
    json_path = settings.results_dir / f"eval_{stamp}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = settings.results_dir / f"eval_{stamp}_details.csv"
    details.to_csv(csv_path, index=False)

    table = pd.DataFrame(
        [(k, round(v, 4)) for k, v in metrics.items()], columns=["metric", "value"]
    )
    print("\n=== METRICS ===")
    print(table.to_string(index=False))
    print(f"\nConfig: {config_snapshot()}")
    print(f"Saved: {json_path.name} (+ {csv_path.name}) in {settings.results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current config.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N questions (quick smoke run).")
    args = parser.parse_args()
    run_eval(limit=args.limit)


if __name__ == "__main__":
    main()
