"""
eval/ablation.py — the differentiator: sweep configs and compare them fairly.

For each configuration we (re)build the index if needed, run the SAME eval harness
(RAGAS + Recall@k/MRR) over the SAME test set, and tabulate the results so the only
thing that changes between rows is the knob under test. Three core ablations:

  (a) retrieval strategy : dense   vs hybrid (dense + BGE-M3 sparse, RRF fusion)
  (b) reranker           : off     vs on (BGE cross-encoder)
  (c) chunk size         : 256     vs 512 vs 1024 tokens

(The agentic-vs-vanilla comparison is added in Phase 4 via --include-agentic.)

Each unique config is evaluated once (deduped) and reused across tables. Results are
saved to results/ablation.csv + results/ablation.md (paste-ready), and the winning
config (by faithfulness + context precision) is reported.

Usage:
    python -m rag_eval.eval.ablation                  # full test set, core ablations
    python -m rag_eval.eval.ablation --limit 20       # cap questions per config
    python -m rag_eval.eval.ablation --include-agentic
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from config import settings
from rag_eval.eval.run import run_eval
from rag_eval.ingestion.chunk import chunk_paper
from rag_eval.ingestion.corpus import load_corpus_meta
from rag_eval.ingestion.index import get_client, index_chunks
from rag_eval.ingestion.parse import extract_text

# --- unique configs (deduped); tables below reference these by key ---
def _cfg(strategy="dense", reranker=False, chunk=512, agentic=False):
    return {
        "retrieval_strategy": strategy,
        "use_reranker": reranker,
        "chunk_size": chunk,
        "use_agentic": agentic,
    }


CONFIGS = {
    "dense_512": _cfg(),
    "hybrid_512": _cfg(strategy="hybrid"),
    "dense_512_rerank": _cfg(reranker=True),
    "dense_256": _cfg(chunk=256),
    "dense_1024": _cfg(chunk=1024),
    "agentic_512": _cfg(agentic=True),
}


def ensure_index(chunk_size: int) -> None:
    """Build the index for this chunk size if it isn't already populated."""
    prev = settings.chunk_size
    settings.chunk_size = chunk_size
    try:
        name = settings.collection_name
        client = get_client()
        if client.collection_exists(name) and client.count(name).count > 0:
            return
        print(f"  building index '{name}' (chunk_size={chunk_size}) ...")
        papers = load_corpus_meta()
        chunks = []
        for p in papers:
            text = extract_text(p.pdf_path)
            if text:
                chunks.extend(chunk_paper(p.paper_id, p.title, text))
        if not chunks:
            raise SystemExit(f"No chunks at chunk_size={chunk_size}.")
        index_chunks(chunks)
    finally:
        settings.chunk_size = prev


def run_config(key: str, limit: int | None) -> dict:
    cfg = CONFIGS[key]
    print(f"\n=== Ablation config: {key} -> {cfg} ===")
    ensure_index(cfg["chunk_size"])
    settings.retrieval_strategy = cfg["retrieval_strategy"]
    settings.use_reranker = cfg["use_reranker"]
    settings.chunk_size = cfg["chunk_size"]
    settings.use_agentic = cfg["use_agentic"]
    metrics = run_eval(limit=limit)
    return metrics


def _row(key: str, metrics: dict) -> dict:
    rk = f"recall@{settings.top_k}"
    return {
        "config": key,
        **CONFIGS[key],
        "faithfulness": metrics.get("faithfulness"),
        "answer_relevancy": metrics.get("answer_relevancy"),
        "context_precision": metrics.get("context_precision"),
        "context_recall": metrics.get("context_recall"),
        "recall@k": metrics.get(rk),
        "mrr": metrics.get("mrr"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ablation study.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap questions per config (speed).")
    parser.add_argument("--include-agentic", action="store_true",
                        help="Also run the agentic-vs-vanilla comparison (Phase 4).")
    args = parser.parse_args()

    keys = ["dense_512", "hybrid_512", "dense_512_rerank", "dense_256", "dense_1024"]
    if args.include_agentic:
        keys.append("agentic_512")

    # snapshot settings so we can restore them afterwards
    saved = (settings.retrieval_strategy, settings.use_reranker,
             settings.chunk_size, settings.use_agentic)

    results: dict[str, dict] = {}
    try:
        for key in keys:
            results[key] = run_config(key, args.limit)
    finally:
        (settings.retrieval_strategy, settings.use_reranker,
         settings.chunk_size, settings.use_agentic) = saved

    rows = [_row(k, results[k]) for k in keys]
    df = pd.DataFrame(rows)

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = settings.results_dir / "ablation.csv"
    df.to_csv(csv_path, index=False)
    (settings.results_dir / "ablation.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    md = _markdown_tables(df, keys)
    (settings.results_dir / "ablation.md").write_text(md, encoding="utf-8")

    print("\n\n================  ABLATION SUMMARY  ================")
    print(df.to_string(index=False))
    print(md)

    # winning config by (faithfulness + context_precision)
    scored = df.dropna(subset=["faithfulness", "context_precision"]).copy()
    if not scored.empty:
        scored["combined"] = scored["faithfulness"] + scored["context_precision"]
        best = scored.sort_values("combined", ascending=False).iloc[0]
        print(f"\nWinning config (faithfulness + context precision): {best['config']} "
              f"(faithfulness={best['faithfulness']:.3f}, "
              f"context_precision={best['context_precision']:.3f})")
    print(f"\nSaved: {csv_path.name}, ablation.json, ablation.md in {settings.results_dir}")


def _fmt(v) -> str:
    return "TBD" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.3f}"


def _markdown_tables(df: pd.DataFrame, keys: list[str]) -> str:
    by = {r["config"]: r for r in df.to_dict("records")}
    out = ["\n#### Ablation: retrieval strategy",
           "| Config | Faithfulness | Context Precision | Context Recall |",
           "|--------|--------------|-------------------|----------------|"]
    for k, label in [("dense_512", "Dense only"), ("hybrid_512", "Hybrid")]:
        if k in by:
            r = by[k]
            out.append(f"| {label} | {_fmt(r['faithfulness'])} | "
                       f"{_fmt(r['context_precision'])} | {_fmt(r['context_recall'])} |")

    out += ["\n#### Ablation: reranker",
            "| Config | Faithfulness | Context Precision |",
            "|--------|--------------|-------------------|"]
    for k, label in [("dense_512", "Reranker off"), ("dense_512_rerank", "Reranker on")]:
        if k in by:
            r = by[k]
            out.append(f"| {label} | {_fmt(r['faithfulness'])} | {_fmt(r['context_precision'])} |")

    out += ["\n#### Ablation: chunk size",
            "| Chunk size | Faithfulness | Context Precision | Context Recall |",
            "|------------|--------------|-------------------|----------------|"]
    for k, label in [("dense_256", "256"), ("dense_512", "512"), ("dense_1024", "1024")]:
        if k in by:
            r = by[k]
            out.append(f"| {label} | {_fmt(r['faithfulness'])} | "
                       f"{_fmt(r['context_precision'])} | {_fmt(r['context_recall'])} |")

    if "agentic_512" in by:
        out += ["\n#### Ablation: agentic vs vanilla",
                "| Config | Faithfulness | Context Recall |",
                "|--------|--------------|----------------|"]
        for k, label in [("dense_512", "Vanilla"), ("agentic_512", "Agentic")]:
            if k in by:
                r = by[k]
                out.append(f"| {label} | {_fmt(r['faithfulness'])} | {_fmt(r['context_recall'])} |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
