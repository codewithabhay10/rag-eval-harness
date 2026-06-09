"""
eval/ragas_eval.py — score generation quality with RAGAS.

We run four canonical RAG metrics, each using the configured LLM (as judge) and
BGE-M3 (for embedding-based similarity), so the judge stack is consistent with the
system under test:

- Faithfulness          — are the answer's claims supported by the retrieved context?
- Answer Relevancy      — does the answer actually address the question?
- Context Precision     — are the retrieved contexts relevant (and well-ranked)?
- Context Recall        — does retrieval cover what the reference answer needs?

Context precision/recall use the reference answer, which is why the test set carries
ground-truth answers.
"""
from __future__ import annotations

import pandas as pd
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from config import settings
from rag_eval.eval.adapters import BGEEmbeddings
from rag_eval.generation.llm import get_llm

# Map RAGAS internal metric names -> the friendly names used in our tables/README.
FRIENDLY = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}


def score_with_ragas(samples: list[dict]) -> tuple[dict[str, float], pd.DataFrame]:
    """Run RAGAS over prebuilt samples. Returns (mean_scores, per_question_df)."""
    dataset = EvaluationDataset(
        samples=[SingleTurnSample(**s) for s in samples]
    )
    judge_llm = LangchainLLMWrapper(get_llm())
    judge_emb = LangchainEmbeddingsWrapper(BGEEmbeddings())
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    run_config = RunConfig(
        max_workers=settings.eval_max_workers,  # gentle on free-tier rate limits
        timeout=240,
        max_retries=5,
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
        show_progress=True,
    )

    df = result.to_pandas()
    # The metric score columns are exactly the numeric columns RAGAS adds.
    numeric = df.select_dtypes(include="number")
    scores = {
        FRIENDLY.get(col, col): float(numeric[col].mean()) for col in numeric.columns
    }
    return scores, df
