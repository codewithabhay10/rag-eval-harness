"""
generation/generate.py — retrieve + generate a grounded, cited answer.

Ties retrieval to the LLM: fetch top-k chunks, build a context-only prompt, call the
configured LLM, and return the answer together with its source chunks. The sources
are returned (not just printed) because eval (Phase 2) needs the exact retrieved
contexts to score faithfulness, context precision, and recall.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from rag_eval.generation.llm import get_llm
from rag_eval.generation.prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt
from rag_eval.retrieval.dense import Retrieved
from rag_eval.retrieval.retriever import retrieve


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Retrieved] = field(default_factory=list)


def answer_question(question: str) -> Answer:
    sources = retrieve(question)
    if not sources:
        # No context retrieved -> refuse rather than let the LLM hallucinate.
        return Answer(question=question, answer=REFUSAL, sources=[])

    prompt = build_user_prompt(question, [s.text for s in sources])
    llm = get_llm()
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    text = response.content if isinstance(response.content, str) else str(response.content)
    return Answer(question=question, answer=text.strip(), sources=sources)
