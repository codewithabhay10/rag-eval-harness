"""generation — grounded, cited answer generation over retrieved context.

The LLM answers ONLY from retrieved passages and cites them, so answers are
verifiable and faithfulness is measurable in eval. Provider is configurable
(Ollama / Groq / Gemini) via generation/llm.py.
"""
