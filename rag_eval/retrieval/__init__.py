"""retrieval — dense (Phase 1), hybrid + reranking (Phase 3). All toggleable.

retriever.retrieve() is the single entry point; it reads config (retrieval_strategy,
use_reranker, top_k) so callers never change when an ablation flips a switch.
"""
