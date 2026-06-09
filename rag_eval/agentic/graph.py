"""
agentic/graph.py — LangGraph agentic RAG: decompose + self-correcting retrieval.

Vanilla RAG retrieves once for the literal question. Multi-hop or under-specified
questions break that. This layer adds two ideas, wired as a LangGraph state machine:

  1. Query decomposition — split the question into the minimal set of sub-questions
     and retrieve for each, so multi-hop questions gather evidence from several places.
  2. Self-correction loop — grade whether the gathered context can actually answer the
     question; if not, reformulate the query and retrieve again (up to a cap).

Graph:  decompose -> retrieve -> grade --(sufficient/at cap)--> generate
                          ^                       |
                          +---- reformulate <-----+ (insufficient)

It is toggled by config.use_agentic and returns the SAME Answer object as vanilla RAG,
so eval and the demo treat it as a drop-in — which is exactly what makes "agentic vs
vanilla" a clean ablation.
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from config import settings
from rag_eval.generation.generate import Answer
from rag_eval.generation.llm import get_llm
from rag_eval.generation.prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt
from rag_eval.retrieval.dense import Retrieved
from rag_eval.retrieval.retriever import retrieve


class AgentState(TypedDict, total=False):
    question: str
    queries: list[str]  # the queries to retrieve for in the next retrieve step
    contexts: list[Retrieved]  # accumulated, deduped by chunk_id
    attempts: int
    sufficient: bool
    answer: str


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _llm_json(system: str, user: str):
    resp = get_llm().invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return _parse_json(content)


# --- nodes ---

def decompose(state: AgentState) -> AgentState:
    q = state["question"]
    data = _llm_json(
        "You break research questions into minimal standalone sub-questions. "
        "Respond ONLY with a JSON array of strings.",
        f"Question: {q}\n\nIf this question is already atomic, return [\"{q}\"]. "
        f"Otherwise return up to {settings.agentic_max_subquestions} sub-questions "
        "that together answer it. JSON array only.",
    )
    subs = [s.strip() for s in data if isinstance(s, str)] if isinstance(data, list) else []
    subs = subs[: settings.agentic_max_subquestions] or [q]
    return {"queries": subs, "contexts": [], "attempts": 0}


def _merge(existing: list[Retrieved], new: list[Retrieved]) -> list[Retrieved]:
    by_id = {c.chunk_id: c for c in existing}
    for c in new:
        prev = by_id.get(c.chunk_id)
        if prev is None or c.score > prev.score:
            by_id[c.chunk_id] = c
    return sorted(by_id.values(), key=lambda c: c.score, reverse=True)


def retrieve_node(state: AgentState) -> AgentState:
    contexts = list(state.get("contexts", []))
    for query in state.get("queries", []):
        contexts = _merge(contexts, retrieve(query))
    return {"contexts": contexts}


def grade(state: AgentState) -> AgentState:
    q = state["question"]
    context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(state["contexts"][:8]))
    data = _llm_json(
        "You judge whether retrieved context is sufficient to fully answer a question. "
        'Respond ONLY with JSON: {"sufficient": true|false}.',
        f"Question: {q}\n\nContext:\n{context}\n\nIs this context sufficient to fully "
        "answer the question? JSON only.",
    )
    sufficient = bool(data.get("sufficient")) if isinstance(data, dict) else True
    return {"sufficient": sufficient}


def reformulate(state: AgentState) -> AgentState:
    q = state["question"]
    have = "; ".join(c.title for c in state["contexts"][:3]) or "nothing useful"
    data = _llm_json(
        "You write a single improved search query to find missing information. "
        'Respond ONLY with JSON: {"query": "..."}.',
        f"Original question: {q}\nRetrieval so far found: {have}\n"
        "Write one better search query to fill the gap. JSON only.",
    )
    new_q = data.get("query", q) if isinstance(data, dict) else q
    return {"queries": [str(new_q)], "attempts": state.get("attempts", 0) + 1}


def generate_node(state: AgentState) -> AgentState:
    contexts = state["contexts"][: settings.top_k]
    if not contexts:
        return {"answer": REFUSAL, "contexts": []}
    prompt = build_user_prompt(state["question"], [c.text for c in contexts])
    resp = get_llm().invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {"answer": text.strip(), "contexts": contexts}


def _route_after_grade(state: AgentState) -> str:
    if state.get("sufficient") or state.get("attempts", 0) >= settings.agentic_max_iterations:
        return "generate"
    return "reformulate"


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("decompose", decompose)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade)
    g.add_node("reformulate", reformulate)
    g.add_node("generate", generate_node)
    g.add_edge(START, "decompose")
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route_after_grade,
                            {"generate": "generate", "reformulate": "reformulate"})
    g.add_edge("reformulate", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


_GRAPH = None


def agentic_answer(question: str) -> Answer:
    """Run the agentic graph and return the same Answer shape as vanilla RAG."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    final = _GRAPH.invoke({"question": question})
    return Answer(
        question=question,
        answer=final.get("answer", REFUSAL),
        sources=final.get("contexts", []),
    )
