"""Collect documents from every configured source into data/documents.jsonl."""

import logging
from datetime import UTC, datetime
from pathlib import Path

from .config import FolderSource, Settings, WebSource
from .crawler import WebCrawler
from .parsers import parse_html, parse_markdown, parse_pdf, parse_text
from .schemas import Document

logger = logging.getLogger(__name__)

_KINDS = {".pdf": "pdf", ".md": "markdown", ".markdown": "markdown", ".txt": "text",
          ".html": "html", ".htm": "html"}


def parse_file(path: Path, origin: str) -> Document | None:
    kind = _KINDS.get(path.suffix.lower())
    if kind is None:
        return None
    title: str | None = None
    if kind == "pdf":
        title, sections = parse_pdf(path.read_bytes())
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        if kind == "markdown":
            title, sections = parse_markdown(text)
        elif kind == "html":
            title, sections, _ = parse_html(text)
        else:
            sections = parse_text(text)
    return Document(
        source=str(path.resolve()), origin=origin, kind=kind, title=title or path.stem,
        sections=sections, fetched_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
    )


def read_folder(source: FolderSource) -> list[Document]:
    if not source.path.is_dir():
        raise FileNotFoundError(f"Folder of source {source.name!r} not found: {source.path}")
    paths = sorted({path for pattern in source.patterns for path in source.path.glob(pattern)})
    documents = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            document = parse_file(path, source.name)
        except Exception as error:  # one unreadable file must not stop the rest
            logger.warning("Skipping %s: %s", path, error)
            continue
        if document is None:
            continue
        if not document.sections:
            logger.warning("Skipping %s: no text (a scanned PDF needs OCR first)", path)
            continue
        documents.append(document)
    return documents


def collect(settings: Settings, names: list[str] | None = None) -> dict[str, list[Document]]:
    """Documents per source name. Raises if a source fails or yields nothing, so a broken source
    never replaces its previously saved documents with an empty list."""
    unknown = set(names or []) - {source.name for source in settings.sources}
    if unknown:
        raise ValueError(f"Unknown source(s): {', '.join(sorted(unknown))}")
    collected: dict[str, list[Document]] = {}
    for source in settings.sources:
        if names and source.name not in names:
            continue
        if isinstance(source, WebSource):
            documents = WebCrawler(source, settings.crawler).crawl()
        else:
            documents = read_folder(source)
        if not documents:
            raise RuntimeError(f"Source {source.name!r} yielded no documents.")
        logger.info("%s: %d documents", source.name, len(documents))
        collected[source.name] = documents
    return collected


def merge(previous: list[Document], collected: dict[str, list[Document]],
          configured: set[str]) -> list[Document]:
    """Replace the collected sources' documents, keep the others', and drop sources that are no
    longer in config.yaml."""
    kept = [d for d in previous if d.origin in configured and d.origin not in collected]
    return kept + [document for documents in collected.values() for document in documents]


def write_documents(documents: list[Document], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(document.model_dump_json() + "\n")
    temporary.replace(destination)  # atomic: a crash never leaves a truncated file


def read_documents(source: Path) -> list[Document]:
    with source.open(encoding="utf-8") as file:
        return [Document.model_validate_json(line) for line in file if line.strip()]
