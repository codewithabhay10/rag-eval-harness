"""
eval/review_testset.py — manually review/edit the synthetic test set.

Synthetic questions are noisy; a human pass over ~100 pairs is what makes the eval
trustworthy. This is a small interactive CLI: step through each item, keep it, drop
it (without deleting — sets kept=False), or edit the question/answer. Changes are
saved back to the same JSON file.

Usage:
    python -m rag_eval.eval.review_testset
"""
from __future__ import annotations

import argparse

from config import settings
from rag_eval.eval.schema import load_testset, save_testset

MENU = (
    "  [Enter]/k keep   d drop   e edit   c show full context   "
    "b back   w save   q save+quit"
)


def _edit(item) -> None:
    print("  (blank input keeps the current value)")
    q = input(f"  question [{item.question}]\n  > ").strip()
    if q:
        item.question = q
    a = input(f"  answer [{item.ground_truth}]\n  > ").strip()
    if a:
        item.ground_truth = a


def main() -> None:
    parser = argparse.ArgumentParser(description="Review the synthetic test set.")
    parser.add_argument("--path", type=str, default=str(settings.testset_path))
    args = parser.parse_args()

    from pathlib import Path

    path = Path(args.path)
    items = load_testset(path)
    kept = sum(1 for i in items if i.kept)
    print(f"Loaded {len(items)} items ({kept} currently kept) from {path}\n")

    i = 0
    while 0 <= i < len(items):
        it = items[i]
        flag = "KEEP" if it.kept else "DROP"
        print(f"\n[{i + 1}/{len(items)}] ({flag})  paper={it.source_paper_id}  "
              f"chunk={it.source_chunk_ids[0] if it.source_chunk_ids else '?'}")
        print(f"  Q: {it.question}")
        print(f"  A: {it.ground_truth}")
        print(MENU)
        cmd = input("  action> ").strip().lower()

        if cmd in ("", "k"):
            it.kept = True
            i += 1
        elif cmd == "d":
            it.kept = False
            i += 1
        elif cmd == "e":
            _edit(it)
        elif cmd == "c":
            ctx = it.reference_contexts[0] if it.reference_contexts else "(none)"
            print("\n--- gold context ---\n" + ctx + "\n--------------------")
        elif cmd == "b":
            i = max(0, i - 1)
        elif cmd == "w":
            save_testset(items, path)
            print("  saved.")
        elif cmd == "q":
            break
        else:
            print("  (unrecognized — try again)")

    save_testset(items, path)
    kept = sum(1 for x in items if x.kept)
    print(f"\nSaved. {kept}/{len(items)} items kept -> {path}")


if __name__ == "__main__":
    main()
