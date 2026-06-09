"""
config.py — the single source of truth for every tunable in the system.

Why one file: results are only comparable if experiments differ by exactly the knob
under test. The ablation runner (Phase 3) flips one of these fields at a time, so
nothing that affects retrieval/generation quality may be hardcoded elsewhere.

Values are read from environment / a local .env via pydantic-settings. Defaults are
laptop-friendly and require no paid API key (Ollama + local Qdrant + CPU embeddings).

Swappable with: any settings backend — but keep exactly ONE source of truth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- paths (all derived artifacts live under data/) ----
    data_dir: Path = PROJECT_ROOT / "data"
    raw_pdf_dir: Path = PROJECT_ROOT / "data" / "pdfs"
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant"
    corpus_meta_path: Path = PROJECT_ROOT / "data" / "corpus.json"
    testset_path: Path = PROJECT_ROOT / "data" / "testset.json"
    results_dir: Path = PROJECT_ROOT / "results"

    # ---- corpus (source is swappable; see ingestion/corpus.py) ----
    corpus_source: Literal["arxiv"] = "arxiv"
    arxiv_query: str = "cat:cs.LG OR cat:cs.CL OR cat:cs.AI"
    num_papers: int = 50

    # ---- chunking (ablation axis: 256 / 512 / 1024 tokens) ----
    # Size is measured in tokens of the embedding tokenizer, so chunk_size maps
    # directly to what BGE-M3 actually sees.
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ---- embeddings (BGE-M3 gives dense + sparse from one model) ----
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 4
    dense_dim: int = 1024  # BGE-M3 dense vector dimension

    # ---- vector db (Qdrant local mode — persists to a folder, no server) ----
    qdrant_collection_prefix: str = "papers"

    # ---- retrieval (ablation knobs; hybrid + reranker arrive in Phase 3) ----
    retrieval_strategy: Literal["dense", "hybrid"] = "dense"
    top_k: int = 5
    candidate_k: int = 20  # candidates fetched before reranking
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # ---- generation / LLM (configurable, local-first) ----
    llm_provider: Literal["ollama", "groq", "gemini"] = "ollama"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    # Ollama defaults to a 2048-token context window, which TRUNCATES RAGAS prompts
    # (question + top_k contexts + JSON format spec) and breaks the judge's output.
    # Raise it so the judge sees the whole prompt. 8192 fits BGE-M3 chunks comfortably.
    ollama_num_ctx: int = 8192
    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"
    google_api_key: Optional[str] = None

    # ---- eval harness (Phase 2) ----
    testset_size: int = 100  # target number of synthetic QA pairs
    random_seed: int = 42  # makes test-set sampling reproducible
    # RAGAS client concurrency. With a SERIAL local LLM (one GPU), keep this at 1:
    # higher values just queue multi-call metrics (faithfulness, context precision)
    # behind each other until they hit the timeout and return NaN. With a hosted API
    # (Groq/Gemini) that serves requests in parallel, raise it (e.g. 4-8).
    eval_max_workers: int = 1
    eval_timeout: int = 600  # per-metric RAGAS timeout (seconds); generous for CPU/local
    eval_max_retries: int = 2  # RAGAS retries; raise for rate-limited hosted APIs (Groq TPM)

    # ---- agentic layer (Phase 4 toggle → becomes an ablation axis) ----
    use_agentic: bool = False
    agentic_max_iterations: int = 2  # self-correction re-retrieval attempts
    agentic_max_subquestions: int = 3  # cap on query decomposition

    @property
    def collection_name(self) -> str:
        """Collection is keyed by chunk size so the chunk-size ablation can keep
        several indexes side by side without re-embedding on every switch."""
        return f"{self.qdrant_collection_prefix}_c{self.chunk_size}"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.raw_pdf_dir, self.qdrant_path, self.results_dir):
            p.mkdir(parents=True, exist_ok=True)


# Import this everywhere: `from config import settings`
settings = Settings()
