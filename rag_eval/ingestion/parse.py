"""
ingestion/parse.py — extract plain text from a PDF.

Uses pypdf (pure-Python, no system deps) so the project runs on any laptop.
Swappable with pymupdf or GROBID for higher-fidelity extraction (better at columns,
tables, references) if extraction quality becomes the bottleneck. We do only light
cleanup here; chunking handles structure.
"""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


def extract_text(pdf_path: Path) -> str:
    """Return cleaned text for the whole PDF, or '' if it can't be read."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"      WARN: cannot open {pdf_path.name}: {e}")
        return ""

    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue  # one bad page shouldn't sink the whole paper
    return _clean("\n".join(parts))


def _clean(text: str) -> str:
    # Join words split across line breaks ("repre-\nsentation" -> "representation").
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse runs of spaces/tabs and excessive blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
