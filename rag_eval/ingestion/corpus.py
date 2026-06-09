"""
ingestion/corpus.py — fetch the raw corpus. The source is swappable.

Default source is the arXiv API (recent ML papers). To use a different corpus,
implement the CorpusSource protocol (`fetch` -> list[PaperMeta] with local PDF
paths) and wire it into get_corpus_source(). Downloaded PDFs are cached on disk, so
re-running ingestion does not re-download.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import arxiv  # thin client over the arXiv API

from config import settings


@dataclass
class PaperMeta:
    paper_id: str
    title: str
    authors: list[str]
    pdf_path: Path

    def to_json(self) -> dict:
        d = asdict(self)
        d["pdf_path"] = str(self.pdf_path)
        return d


class CorpusSource(Protocol):
    def fetch(self, num_papers: int) -> list[PaperMeta]:
        ...


class ArxivSource:
    """Pull papers via the arXiv API and download their PDFs."""

    def __init__(self, query: str, pdf_dir: Path):
        self.query = query
        self.pdf_dir = pdf_dir

    def fetch(self, num_papers: int) -> list[PaperMeta]:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        # delay_seconds respects arXiv's rate-limit guidance.
        client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
        search = arxiv.Search(
            query=self.query,
            max_results=num_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        papers: list[PaperMeta] = []
        for result in client.results(search):
            short_id = result.get_short_id().replace("/", "_")
            filename = f"{short_id}.pdf"
            pdf_path = self.pdf_dir / filename
            if not pdf_path.exists():
                try:
                    result.download_pdf(dirpath=str(self.pdf_dir), filename=filename)
                except Exception as e:  # skip a bad download, keep the run going
                    print(f"      WARN: failed to download {short_id}: {e}")
                    continue
            papers.append(
                PaperMeta(
                    paper_id=short_id,
                    title=result.title.strip().replace("\n", " "),
                    authors=[a.name for a in result.authors],
                    pdf_path=pdf_path,
                )
            )
        return papers


def get_corpus_source() -> CorpusSource:
    if settings.corpus_source == "arxiv":
        return ArxivSource(settings.arxiv_query, settings.raw_pdf_dir)
    raise ValueError(f"Unknown corpus_source: {settings.corpus_source!r}")


def save_corpus_meta(papers: list[PaperMeta]) -> None:
    """Persist corpus metadata so later stages (eval) can map ids -> titles."""
    settings.corpus_meta_path.parent.mkdir(parents=True, exist_ok=True)
    settings.corpus_meta_path.write_text(
        json.dumps([p.to_json() for p in papers], indent=2), encoding="utf-8"
    )
