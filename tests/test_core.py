"""
tests/test_core.py — fast unit tests for pure logic (no models, no DB, no LLM).

These cover the deterministic pieces the metrics depend on: retrieval scoring,
sparse-vector conversion, LLM-output JSON parsing, balanced sampling, and test-set
serialization. The heavyweight, non-deterministic parts (RAGAS, retrieval) are
exercised by the eval harness itself, not here.

Run:  python -m pytest
"""
from __future__ import annotations

from pathlib import Path

from rag_eval.agentic.graph import _parse_json as agentic_parse
from rag_eval.eval.build_testset import _balanced_order, _parse_json
from rag_eval.eval.retrieval_metrics import aggregate, recall_at_k, reciprocal_rank
from rag_eval.eval.schema import TestItem, load_testset, save_testset
from rag_eval.generation.prompts import build_user_prompt
from rag_eval.ingestion.embed import sparse_to_indices_values


# --- retrieval metrics ---

def test_recall_at_k_hit_and_miss():
    assert recall_at_k(["a"], ["x", "a", "y"], k=3) == 1.0
    assert recall_at_k(["a"], ["x", "y", "z"], k=3) == 0.0
    # gold not within the cutoff
    assert recall_at_k(["a"], ["x", "y", "a"], k=2) == 0.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "b"], ["a", "z"], k=5) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(["a"], ["x", "a", "y"]) == 0.5
    assert reciprocal_rank(["a"], ["a", "b"]) == 1.0
    assert reciprocal_rank(["a"], ["b", "c"]) == 0.0


def test_aggregate_means():
    pairs = [(["a"], ["a", "b"]), (["a"], ["b", "a"])]
    out = aggregate(pairs, k=2)
    assert out["recall@2"] == 1.0
    assert out["mrr"] == 0.75  # (1/1 + 1/2) / 2


# --- sparse vector conversion ---

def test_sparse_to_indices_values_drops_nonpositive():
    idx, val = sparse_to_indices_values({"5": 0.3, "9": 0.0, "12": -1.0, "7": 2.0})
    assert idx == [5, 7]
    assert val == [0.3, 2.0]


# --- JSON parsing from LLM output ---

def test_parse_json_plain_and_fenced():
    assert _parse_json('{"question": "q", "answer": "a"}')["question"] == "q"
    assert _parse_json('```json\n{"skip": true}\n```')["skip"] is True
    assert _parse_json("no json here") is None


def test_agentic_parse_array():
    assert agentic_parse('["one", "two"]') == ["one", "two"]
    assert agentic_parse('{"sufficient": false}')["sufficient"] is False


# --- balanced sampling across papers ---

def test_balanced_order_is_round_robin():
    chunks = [{"paper_id": "A", "text": t} for t in "12345"] + \
             [{"paper_id": "B", "text": t} for t in "12"]
    ordered = _balanced_order(chunks, seed=1)
    assert len(ordered) == 7
    # B's two chunks should appear within the first four (round-robin), not all at the end
    positions = [i for i, c in enumerate(ordered) if c["paper_id"] == "B"]
    assert min(positions) < 2


# --- prompt construction ---

def test_build_user_prompt_labels_passages():
    p = build_user_prompt("What is X?", ["alpha", "beta"])
    assert "[1] alpha" in p and "[2] beta" in p and "What is X?" in p


# --- test-set serialization round-trip ---

def test_testset_round_trip(tmp_path: Path):
    items = [TestItem(id="q0", question="q", ground_truth="a",
                      reference_contexts=["ctx"], source_chunk_ids=["p::0"],
                      source_paper_id="p")]
    path = tmp_path / "ts.json"
    save_testset(items, path)
    loaded = load_testset(path)
    assert loaded[0].question == "q"
    assert loaded[0].source_chunk_ids == ["p::0"]
