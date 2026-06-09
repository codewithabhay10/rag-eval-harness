# RAG Eval Harness

A Retrieval-Augmented Generation system over a technical-paper corpus, built around a
**measurable** core: an automated evaluation pipeline (RAGAS) plus an **ablation study**
that proves which design choices actually improve answer quality.

> The RAG demo is not the point — the eval harness and ablation are. The system is
> designed to answer: *"Which configuration is best, and by how much, measured against a
> labeled test set?"*

## Architecture

```
arXiv PDFs ─► parse ─► chunk ─► BGE-M3 embed ─► Qdrant (local)
                                                     │
                              query ─► retrieve (dense | hybrid, ±rerank)
                                                     │
                                       LLM (Ollama | Groq | Gemini)
                                                     │
                                  grounded, cited answer + sources
                                                     │
                            RAGAS eval  ◄────────────┘   (Phase 2-3)
```

| Layer         | Choice                          | Why |
|---------------|---------------------------------|-----|
| Orchestration | LangGraph                       | For the agentic RAG layer (Phase 4) |
| Embeddings    | BGE-M3                          | Dense **and** sparse from one model → enables the hybrid ablation |
| Vector DB     | Qdrant (local mode)             | No server; persists to a folder; supports hybrid |
| Reranker      | BGE cross-encoder (`bge-reranker-v2-m3`) | The reranker on/off ablation |
| LLM           | Configurable: Ollama / Groq / Gemini | Local-first; default needs no paid key |
| Eval          | RAGAS                           | Canonical RAG eval; integrates with the LangChain stack |
| Demo UI       | Streamlit                       | Fast to build, easy to show |

### Project layout
```
rag_eval/
  ingestion/   corpus.py · parse.py · chunk.py · embed.py · index.py · run.py
  retrieval/   dense.py · retriever.py        (hybrid + rerank: Phase 3)
  generation/  llm.py · prompts.py · generate.py · ask.py
  eval/        (Phase 2-3)   agentic/ (Phase 4)   app/ (Phase 5)
config.py      single source of truth (pydantic-settings + .env)
.env.example   documented; copy to .env (gitignored)
requirements.txt  pinned
```

## Setup

Requires **Python 3.11**. (Windows / PowerShell shown; Unix in parentheses.)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # (source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env                 # (cp .env.example .env) — then edit if needed
```

### Choose an LLM (local-first)
Default provider is **Ollama** (no API key). Install Ollama, then pull a model:
```powershell
ollama pull llama3.1
```
Prefer a hosted free tier instead? Set `LLM_PROVIDER=groq` (or `gemini`) and the matching
API key in `.env`. Keys are never committed.

## How to run

Run all commands **from the project root** so top-level `config` is importable.

```powershell
# Phase 1 — ingest the corpus (start small, then scale up)
python -m rag_eval.ingestion.run --num-papers 50
#   first run downloads BGE-M3 (~2GB) into the HuggingFace cache

# Phase 1 — ask a question (sanity check; needs Ollama running + a pulled model)
python -m rag_eval.generation.ask "What is contrastive learning?"
```

# Phase 2 — build, review, and score against a labeled test set
python -m rag_eval.eval.build_testset --num-questions 100   # synthetic QA pairs
python -m rag_eval.eval.review_testset                      # hand-review/edit them
python -m rag_eval.eval.run                                 # RAGAS + Recall@k/MRR

# Later phases (commands land as they're built)
python -m rag_eval.eval.ablation           # Phase 3
streamlit run rag_eval/app/main.py         # Phase 5
```

The eval run scores the **current** config (from `.env`/`config.py`) and writes a
timestamped `results/eval_*.json` (metrics + a snapshot of the config that produced
them) plus a per-question `*_details.csv`. Numbers come only from real runs.

## Configuration

Every knob lives in `config.py` (overridable via `.env`). The ones that drive results:
`chunk_size`, `chunk_overlap`, `retrieval_strategy` (dense|hybrid), `use_reranker`,
`top_k`, `use_agentic`, `llm_provider`, `embedding_model`. Nothing that affects quality is
hardcoded elsewhere, so the ablation runner can flip exactly one variable at a time.

## Design decisions (the "why")

- **One model for dense + sparse (BGE-M3).** Hybrid retrieval needs both a semantic and
  a lexical signal. BGE-M3 emits both from a single forward pass, so the hybrid index
  needs no second model and the comparison is apples-to-apples.
- **Late fusion (RRF) for hybrid.** Dense cosine and sparse dot scores live on different
  scales, so blending raw scores is unreliable. Reciprocal Rank Fusion combines by *rank*,
  which is scale-free — Qdrant does it natively via prefetch + `FusionQuery`.
- **Chunk size measured in tokens of the embedding tokenizer.** `chunk_size` then maps
  directly to what BGE-M3 actually sees and stays under its context limit.
- **Cross-encoder reranking over a small candidate pool.** First-stage retrieval scores
  query and chunk independently; a cross-encoder reads them together (more accurate, more
  costly), so we only rerank `candidate_k` and keep `top_k`.
- **Context-only prompt with a fixed refusal string.** No outside knowledge can leak, which
  is what makes faithfulness measurable and gives a clean abstention signal for recall.
- **Gold chunk ids in the test set.** Lets us score *retrieval* (Recall@k / MRR) directly,
  not just the generated answer — the two failure modes are separable.
- **Agentic layer returns the same `Answer` object as vanilla.** So "agentic vs vanilla" is
  a real ablation: only the retrieval/answering strategy changes, nothing downstream.

## Results

> Every number below comes from an **actual** RAGAS run (`results/ablation.csv`), not by
> hand. **Run config:** 10 arXiv papers · 10-question test set · BGE-M3 embeddings ·
> Qdrant local · judge = local Ollama `qwen2.5:3b` on a 4 GB GPU. This is a deliberately
> small, laptop-runnable demonstration — see *Scale & caveats* below — but the pipeline and
> numbers are real. Re-run at scale by editing `config.py`/`.env`; nothing else changes.

### Baseline (dense, no reranker, chunk 512)
| Metric            | Value |
|-------------------|-------|
| Faithfulness      | 0.148 |
| Answer Relevancy  | 0.370 |
| Context Precision | 0.667 |
| Context Recall    | 0.630 |
| Recall@k          | 0.700 |
| MRR               | 0.600 |

### Ablation: retrieval strategy — **hybrid wins**
| Config     | Faithfulness | Context Precision | Context Recall |
|------------|--------------|-------------------|----------------|
| Dense only | 0.148        | 0.667             | 0.630          |
| **Hybrid** | **0.200**    | **0.796**         | **0.750**      |

### Ablation: reranker — **lifts retrieval + faithfulness**
| Config       | Faithfulness | Context Precision | Recall@k | MRR   |
|--------------|--------------|-------------------|----------|-------|
| Reranker off | 0.148        | 0.667             | 0.700    | 0.600 |
| **Reranker on** | **0.219** | 0.665            | **0.900**| **0.717** |

### Ablation: chunk size — **256 wins here**
| Chunk size | Faithfulness | Context Precision | Context Recall |
|------------|--------------|-------------------|----------------|
| **256**    | **0.361**    | **0.767**         | **0.833**      |
| 512        | 0.148        | 0.667             | 0.630          |
| 1024       | 0.296        | 0.445             | 0.310          |

### Ablation: agentic vs vanilla — **agentic improves recall**
| Config  | Faithfulness | Context Recall |
|---------|--------------|----------------|
| Vanilla | 0.148        | 0.630          |
| **Agentic** | **0.213** | **0.750**     |

### Winning configuration
**`dense_256`** (chunk size 256) by faithfulness + context precision (faithfulness 0.361,
context precision 0.767). Headline findings on this corpus: **hybrid retrieval** raised
context precision 0.667 → 0.796 and recall 0.63 → 0.75; the **reranker** raised Recall@k
0.70 → 0.90 and MRR 0.60 → 0.72; **smaller chunks (256)** beat 512/1024 on every context
metric; and the **agentic** layer raised context recall 0.63 → 0.75.

### Scale & caveats
- **Absolute faithfulness reads low** (0.15–0.36): the local 3B judge is strict and weak at
  faithfulness's claim-by-claim NLI step (spot-checked answers *are* grounded and correct).
  The **relative** ordering across configs is the informative signal. A stronger judge
  (Groq `llama-3.3-70b` / GPT-class) gives higher absolute faithfulness — a Groq reference
  run was attempted but the free tier's 100k-tokens/day cap was exhausted; re-run with a
  paid/dev tier via `LLM_PROVIDER=groq`.
- **`Recall@k`/`MRR` use exact gold-chunk-id matching**, so they are only comparable across
  configs that share a chunk size; the chunk-size ablation is therefore judged on the RAGAS
  context metrics (chunk-id-independent). In `results/ablation.csv` the 256/1024 rows show
  `recall@k = 0` for exactly this reason.
- **To scale up:** raise `--num-papers`, rebuild the test set (`--num-questions 100`), and
  set `LLM_PROVIDER=groq` (or a larger local model) for a stronger judge.

## Build status
- [x] **Phase 1** — Ingestion + baseline RAG (dense retrieval, cited generation, `ask` CLI)
- [x] **Phase 2** — Eval harness (synthetic test set + review + RAGAS + Recall@k/MRR)
- [x] **Phase 3** — Ablation study (dense vs hybrid · reranker off/on · chunk 256/512/1024)
- [x] **Phase 4** — Agentic layer (LangGraph decomposition + self-correction)
- [x] **Phase 5** — Streamlit demo + final docs
