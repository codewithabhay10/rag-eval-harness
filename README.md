# RAG Eval Harness

A Retrieval-Augmented Generation system over a corpus of arXiv ML papers, built around a
**measurable** core: an automated evaluation pipeline (RAGAS) plus an **ablation study**
that proves which design choices actually improve answer quality.

> The RAG demo is not the point — the eval harness and ablation are. The system is built to
> answer one question: *"Which configuration is best, and by how much, measured against a
> labeled test set?"*

The full pipeline is implemented end to end: ingestion, dense **and** hybrid retrieval,
cross-encoder reranking, cited generation, an agentic (LangGraph) layer, a RAGAS + retrieval
metrics harness, an ablation runner, unit tests, and a Streamlit demo.

## Architecture

```
arXiv PDFs ─► parse ─► chunk ─► BGE-M3 embed ─► Qdrant (local, dense + sparse)
                                                      │
                               query ─► retrieve (dense | hybrid, ± reranker)
                                                      │
                                  [optional] agentic: decompose + self-correct
                                                      │
                                        LLM (Ollama | Groq | Gemini)
                                                      │
                                   grounded, cited answer + sources
                                                      │
                        RAGAS + Recall@k / MRR  ◄──────┘
```

| Layer         | Choice                                   | Why |
|---------------|------------------------------------------|-----|
| Orchestration | LangGraph                                | Agentic layer: query decomposition + retrieval self-correction |
| Embeddings    | BGE-M3                                    | Dense **and** sparse from one model → enables the hybrid ablation |
| Vector DB     | Qdrant (local mode)                      | No server; persists to a folder; native hybrid search |
| Reranker      | BGE cross-encoder (`bge-reranker-v2-m3`) | The reranker on/off ablation |
| LLM           | Configurable: Ollama / Groq / Gemini     | Local-first; default needs no paid key |
| Eval          | RAGAS                                     | Canonical RAG eval; integrates with the LangChain stack |
| Demo UI       | Streamlit                                | Fast to build, easy to show |

### Project layout

```
rag_eval/
  ingestion/   corpus.py · parse.py · chunk.py · embed.py · index.py · run.py
  retrieval/   dense.py · hybrid.py · rerank.py · retriever.py
  generation/  llm.py · prompts.py · generate.py · ask.py
  agentic/     graph.py                       # LangGraph decompose + self-correct
  eval/        schema.py · adapters.py · build_testset.py · review_testset.py
               retrieval_metrics.py · ragas_eval.py · run.py · ablation.py
  app/         main.py                        # Streamlit demo
tests/         test_core.py                   # fast unit tests (no models/DB)
config.py            # single source of truth (pydantic-settings + .env)
.env.example         # documented; copy to .env (gitignored)
requirements.txt     # pinned
data/testset.json    # the curated QA test set (version-controlled)
results/ablation.*   # the ablation comparison tables (csv / json / md)
```

## Setup

Requires **Python 3.11**. (Windows / PowerShell shown; Unix equivalent in parentheses.)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # (source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env                 # (cp .env.example .env) — then edit if needed
```

### Choose an LLM (local-first)

Default provider is **Ollama** (no API key). Install Ollama, start it, and pull a model:

```powershell
ollama pull qwen2.5:3b      # ~2GB, fits a 4GB-VRAM GPU; good at the structured output RAGAS needs
```

Prefer a hosted free tier? Set `LLM_PROVIDER=groq` (or `gemini`) and the matching API key in
`.env`. Keys are never committed. Note: hosted free tiers have low token limits — fine for the
demo `ask`, but a full ablation is token-heavy, so local is recommended for batch eval.

## How to run

Run every command **from the project root** so the top-level `config` module is importable.

```powershell
# 1. Ingest the corpus  (first run downloads BGE-M3, ~2GB, into the HuggingFace cache)
python -m rag_eval.ingestion.run --num-papers 10

# 2. Ask a question — grounded answer + cited sources  (needs Ollama running + a pulled model)
python -m rag_eval.generation.ask "What does OmniGameArena benchmark?"

# 3. Build + review a labeled QA test set
python -m rag_eval.eval.build_testset --num-questions 20   # synthetic QA pairs
python -m rag_eval.eval.review_testset                     # hand-review / keep / drop / edit

# 4. Score the CURRENT config against the test set (RAGAS + Recall@k / MRR)
python -m rag_eval.eval.run

# 5. Run the full ablation study (dense vs hybrid · reranker · chunk size · agentic)
python -m rag_eval.eval.ablation --include-agentic

# 6. Launch the demo
streamlit run rag_eval/app/main.py

# Tests
python -m pytest
```

`eval.run` scores the current config and writes a timestamped `results/eval_*.json` (metrics +
a snapshot of the config that produced them) plus a per-question `*_details.csv`.
`eval.ablation` sweeps configs and writes `results/ablation.{csv,json,md}`. Every number comes
from a real run — nothing is hand-written.

## Configuration

Every knob lives in `config.py` (overridable via `.env`). The ones that drive results:
`chunk_size`, `chunk_overlap`, `retrieval_strategy` (`dense` | `hybrid`), `use_reranker`,
`top_k`, `use_agentic`, `llm_provider`, `embedding_model`. Nothing that affects quality is
hardcoded elsewhere, so the ablation runner can flip exactly one variable at a time.

## Design decisions (the "why")

- **One model for dense + sparse (BGE-M3).** Hybrid retrieval needs both a semantic and a
  lexical signal. BGE-M3 emits both from a single forward pass, so the hybrid index needs no
  second model and the comparison stays apples-to-apples.
- **Late fusion (RRF) for hybrid.** Dense cosine and sparse dot scores live on different
  scales, so blending raw scores is unreliable. Reciprocal Rank Fusion combines by *rank* —
  scale-free — and Qdrant does it natively via prefetch + `FusionQuery`.
- **Chunk size measured in tokens of the embedding tokenizer.** `chunk_size` then maps directly
  to what BGE-M3 actually sees and stays under its context limit.
- **Cross-encoder reranking over a small candidate pool.** First-stage retrieval scores query
  and chunk independently; a cross-encoder reads them together (more accurate, more costly), so
  we only rerank `candidate_k` candidates and keep the top `top_k`.
- **Context-only prompt with a fixed refusal string.** No outside knowledge can leak, which is
  what makes faithfulness measurable and gives a clean abstention signal for recall.
- **Gold chunk ids in the test set.** Lets us score *retrieval* (Recall@k / MRR) directly, not
  just the generated answer — the two failure modes stay separable.
- **The agentic layer returns the same `Answer` object as vanilla RAG.** So "agentic vs vanilla"
  is a real ablation: only the retrieval/answering strategy changes, nothing downstream.

## Results

> Every number below comes from an **actual** run (`results/ablation.csv`), not by hand.
> **Run config:** 10 arXiv papers · 10-question test set · BGE-M3 embeddings · Qdrant local ·
> judge = local Ollama `qwen2.5:3b` on a 4 GB GPU. This is a deliberately small,
> laptop-runnable demonstration (see **Scale & caveats**) — but the pipeline and the numbers
> are real. Re-run at scale by editing `config.py` / `.env`; nothing else changes.

### Baseline (dense, no reranker, chunk 512)
| Metric            | Value |
|-------------------|-------|
| Faithfulness      | 0.148 |
| Answer Relevancy  | 0.370 |
| Context Precision | 0.667 |
| Context Recall    | 0.630 |
| Recall@k          | 0.700 |
| MRR               | 0.600 |

### Ablation: retrieval strategy — hybrid wins
| Config     | Faithfulness | Context Precision | Context Recall |
|------------|--------------|-------------------|----------------|
| Dense only | 0.148        | 0.667             | 0.630          |
| **Hybrid** | **0.200**    | **0.796**         | **0.750**      |

### Ablation: reranker — lifts retrieval + faithfulness
| Config          | Faithfulness | Context Precision | Recall@k  | MRR       |
|-----------------|--------------|-------------------|-----------|-----------|
| Reranker off    | 0.148        | 0.667             | 0.700     | 0.600     |
| **Reranker on** | **0.219**    | 0.665             | **0.900** | **0.717** |

### Ablation: chunk size — 256 wins here
| Chunk size | Faithfulness | Context Precision | Context Recall |
|------------|--------------|-------------------|----------------|
| **256**    | **0.361**    | **0.767**         | **0.833**      |
| 512        | 0.148        | 0.667             | 0.630          |
| 1024       | 0.296        | 0.445             | 0.310          |

### Ablation: agentic vs vanilla — agentic improves recall
| Config      | Faithfulness | Context Recall |
|-------------|--------------|----------------|
| Vanilla     | 0.148        | 0.630          |
| **Agentic** | **0.213**    | **0.750**      |

### Winning configuration

**`dense_256`** (chunk size 256), by faithfulness + context precision (0.361 / 0.767).
Headline findings on this corpus:

- **Hybrid retrieval** raised context precision **0.667 → 0.796** and recall **0.63 → 0.75**.
- The **reranker** raised Recall@k **0.70 → 0.90** and MRR **0.60 → 0.72**.
- **Smaller chunks (256)** beat 512 / 1024 on every context metric.
- The **agentic** layer raised context recall **0.63 → 0.75**.

### Scale & caveats

- **Absolute faithfulness reads low (0.15–0.36).** The local 3B judge is strict and weak at
  faithfulness's claim-by-claim NLI step (spot-checked answers *are* grounded and correct), so
  the **relative** ordering across configs is the informative signal, not the absolute value. A
  stronger judge (Groq `llama-3.3-70b` / GPT-class) gives higher absolute faithfulness; a Groq
  reference run was attempted but the free tier's daily token cap was exhausted. Re-run with a
  paid/dev tier via `LLM_PROVIDER=groq`.
- **`Recall@k` / `MRR` use exact gold-chunk-id matching**, so they are only comparable across
  configs that share a chunk size. The chunk-size ablation is therefore judged on the RAGAS
  context metrics (chunk-id-independent); in `results/ablation.csv` the 256/1024 rows show
  `recall@k = 0` for exactly this reason.
- **To scale up:** raise `--num-papers`, rebuild the test set (`--num-questions 100`), and set
  `LLM_PROVIDER=groq` (or a larger local model) for a stronger judge.

## Status

All phases implemented and verified:

- [x] **Ingestion + baseline RAG** — parse, chunk, BGE-M3 embed, Qdrant index, dense retrieval, cited generation, `ask` CLI
- [x] **Eval harness** — synthetic test set + human review + RAGAS (faithfulness, answer relevancy, context precision/recall) + Recall@k / MRR
- [x] **Ablation study** — dense vs hybrid · reranker off/on · chunk 256/512/1024
- [x] **Agentic layer** — LangGraph query decomposition + retrieval self-correction (agentic vs vanilla)
- [x] **Demo + docs** — Streamlit app (answer, cited sources, retrieval scores) + this README with real results
