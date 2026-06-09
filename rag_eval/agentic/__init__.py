"""agentic — LangGraph query decomposition + retrieval self-correction (Phase 4).

graph.py builds a state machine: decompose the question into sub-questions, retrieve
for each, grade whether the gathered context is sufficient, and if not reformulate and
re-retrieve (up to a cap) before generating. Toggled by config.use_agentic and returns
the same Answer object as vanilla RAG, so it slots into eval as an ablation axis.
"""
