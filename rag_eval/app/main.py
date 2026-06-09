"""
app/main.py — Streamlit demo for the RAG system.

Ask a question and see: the grounded answer, the cited source chunks with their
retrieval scores, and the live configuration. The sidebar exposes the same knobs the
ablation studies (strategy, reranker, top_k, agentic) so you can feel their effect
interactively — every toggle just sets config, which the pipeline reads per query.

Run:  streamlit run rag_eval/app/main.py
"""
from __future__ import annotations

import streamlit as st

from config import settings
from rag_eval.generation.generate import answer_question

st.set_page_config(page_title="RAG Eval Harness", page_icon="🔎", layout="wide")
st.title("🔎 RAG over technical papers")
st.caption("Answers are generated ONLY from retrieved context, with citations.")

with st.sidebar:
    st.header("Configuration")
    settings.retrieval_strategy = st.selectbox(
        "Retrieval strategy", ["dense", "hybrid"],
        index=0 if settings.retrieval_strategy == "dense" else 1,
    )
    settings.use_reranker = st.checkbox("Use reranker", value=settings.use_reranker)
    settings.use_agentic = st.checkbox("Agentic (decompose + self-correct)",
                                       value=settings.use_agentic)
    settings.top_k = st.slider("top_k", 1, 10, settings.top_k)
    st.divider()
    st.markdown(
        f"**LLM:** {settings.llm_provider} · "
        f"`{getattr(settings, settings.llm_provider + '_model', '?')}`\n\n"
        f"**Embeddings:** {settings.embedding_model}\n\n"
        f"**Collection:** {settings.collection_name}"
    )
    st.info("Toggles affect the next question. Ingest a corpus first if results are empty.")

question = st.text_input("Ask a question about the corpus:",
                         placeholder="e.g. What is OmniGameArena and what does it benchmark?")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving and generating ..."):
        result = answer_question(question.strip())

    st.subheader("Answer")
    st.write(result.answer)

    st.subheader(f"Sources ({len(result.sources)})")
    if not result.sources:
        st.warning("No sources retrieved — is the corpus ingested for this chunk size?")
    for i, s in enumerate(result.sources, 1):
        with st.expander(f"[{i}] {s.title}  ·  score {s.score:.3f}  ·  {s.chunk_id}"):
            st.caption(f"paper {s.paper_id} · chunk {s.chunk_id}")
            st.write(s.text)
