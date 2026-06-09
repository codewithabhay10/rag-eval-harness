"""
eval/schema.py — the test-set data model + JSON load/save.

A TestItem is one labeled example: a question, its ground-truth answer, and the gold
context it was derived from (chunk text + chunk id). Keeping the gold chunk id lets
us score RETRIEVAL (Recall@k / MRR) as well as generation (RAGAS). Stored as plain
JSON so it is easy to inspect and hand-edit during review.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TestItem:
    id: str
    question: str
    ground_truth: str  # reference answer (used by RAGAS context precision/recall)
    reference_contexts: list[str]  # gold passage text(s)
    source_chunk_ids: list[str]  # gold chunk id(s) — for retrieval metrics
    source_paper_id: str
    kept: bool = True  # set False during review to exclude without deleting


def save_testset(items: list[TestItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(i) for i in items], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_testset(path: Path) -> list[TestItem]:
    if not path.exists():
        raise FileNotFoundError(
            f"Test set not found at {path}. Build it first: "
            f"python -m rag_eval.eval.build_testset"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TestItem(**d) for d in raw]


def kept_items(items: list[TestItem]) -> list[TestItem]:
    """Only the items that survived review (kept=True)."""
    return [i for i in items if i.kept]
