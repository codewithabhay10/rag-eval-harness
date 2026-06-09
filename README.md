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

Later phases (commands land as they're built):
```powershell
python -m rag_eval.eval.build_testset      # Phase 2
python -m rag_eval.eval.run                # Phase 2
python -m rag_eval.eval.ablation           # Phase 3
streamlit run rag_eval/app/main.py         # Phase 5
```

## Configuration

Every knob lives in `config.py` (overridable via `.env`). The ones that drive results:
`chunk_size`, `chunk_overlap`, `retrieval_strategy` (dense|hybrid), `use_reranker`,
`top_k`, `use_agentic`, `llm_provider`, `embedding_model`. Nothing that affects quality is
hardcoded elsewhere, so the ablation runner can flip exactly one variable at a time.

## Results

> **No metric below is filled in by hand.** Every number must come from actually running
> RAGAS on the test set; tables stay `TBD` until a real run produces them. Fabricated
> metrics would defeat the purpose of the project.

### Baseline (dense, no reranker, chunk 512)
| Metric            | Value |
|-------------------|-------|
| Faithfulness      | TBD   |
| Answer Relevancy  | TBD   |
| Context Precision | TBD   |
| Context Recall    | TBD   |
| Recall@k          | TBD   |
| MRR               | TBD   |

### Ablation: retrieval strategy
| Config     | Faithfulness | Context Precision | Context Recall |
|------------|--------------|-------------------|----------------|
| Dense only | TBD          | TBD               | TBD            |
| Hybrid     | TBD          | TBD               | TBD            |

### Ablation: reranker
| Config       | Faithfulness | Context Precision |
|--------------|--------------|-------------------|
| Reranker off | TBD          | TBD               |
| Reranker on  | TBD          | TBD               |

### Ablation: chunk size
| Chunk size | Faithfulness | Context Precision | Context Recall |
|------------|--------------|-------------------|----------------|
| 256        | TBD          | TBD               | TBD            |
| 512        | TBD          | TBD               | TBD            |
| 1024       | TBD          | TBD               | TBD            |

### Ablation: agentic vs vanilla
| Config  | Faithfulness | Context Recall (multi-hop) |
|---------|--------------|----------------------------|
| Vanilla | TBD          | TBD                        |
| Agentic | TBD          | TBD                        |

### Winning configuration
TBD — recorded once the ablation has actually run.

## Build status
- [x] **Phase 1** — Ingestion + baseline RAG (dense retrieval, cited generation, `ask` CLI)
- [ ] Phase 2 — Eval harness (synthetic test set + RAGAS + Recall@k/MRR)
- [ ] Phase 3 — Ablation study (dense vs hybrid · reranker off/on · chunk 256/512/1024)
- [ ] Phase 4 — Agentic layer (LangGraph decomposition + self-correction)
- [ ] Phase 5 — Streamlit demo + final docs
