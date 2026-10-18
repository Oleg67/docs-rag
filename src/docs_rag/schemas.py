from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Section(BaseModel):
    heading: str | None = None
    page: int | None = None  # 1-based, PDFs only
    text: str  # paragraphs separated by blank lines


class Document(BaseModel):
    source: str  # page URL or absolute file path; the citation target
    origin: str  # name of the config.yaml source it came from
    kind: Literal["web", "pdf", "html", "markdown", "text"]
    title: str
    sections: list[Section]
    fetched_at: datetime

    @property
    def text(self) -> str:
        parts = []
        for section in self.sections:
            if section.page is not None:
                parts.append(f"[page {section.page}]")
            if section.heading:
                parts.append(f"## {section.heading}")
            parts.append(section.text)
        return "\n\n".join(part for part in parts if part)


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    origin: str
    title: str
    section: str | None = None
    page: int | None = None
    position: int


class Citation(BaseModel):
    title: str
    source: str
    section: str | None
    page: int | None
    text: str
    # Dense cosine similarity; the relevance cutoff (search.min_score) applies to it.
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[Citation]
