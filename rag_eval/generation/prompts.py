"""
generation/prompts.py — prompt templates for grounded, cited answers.

The system prompt forces context-only answering and a fixed refusal string when the
answer isn't present. This is a deliberate eval-driven choice: it makes faithfulness
measurable (no outside knowledge to leak) and gives a clean signal for context
recall (the model abstains when retrieval misses). Passages are labelled [1], [2]...
so the model can cite them and we can trace answers back to sources.
"""
from __future__ import annotations

REFUSAL = "I cannot answer this from the provided context."

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the question USING ONLY the "
    "provided context passages. Do not use any outside knowledge. If the context "
    f"does not contain the answer, reply with exactly: '{REFUSAL}' "
    "Cite the passages you rely on with bracketed numbers like [1], [2] that match "
    "the passage labels."
)


def build_user_prompt(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (with citations):"
    )
