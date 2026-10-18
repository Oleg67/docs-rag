import re
from collections.abc import Iterable
from uuid import NAMESPACE_URL, uuid5

from .config import ChunkingSettings
from .schemas import Chunk, Document

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_long(text: str, limit: int) -> list[str]:
    """Split on sentence boundaries, then on spaces, then anywhere, so no part exceeds limit."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(text):
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit + 1)
            cut = cut if cut > 0 else limit
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        pieces.append(sentence)
    parts: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > limit:
            parts.append(current)
            current = piece
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def chunk_documents(documents: Iterable[Document], settings: ChunkingSettings) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document, settings)]


def chunk_document(document: Document, settings: ChunkingSettings) -> list[Chunk]:
    """Pack paragraphs into chunks of at most max_characters; chunks never cross sections (or PDF
    pages), and consecutive chunks of a section share overlap_paragraphs paragraphs."""
    limit = settings.max_characters
    chunks: list[Chunk] = []
    for section in document.sections:
        paragraphs = [" ".join(raw.split()) for raw in section.text.split("\n\n")]
        parts = [part for paragraph in paragraphs if paragraph
                 for part in _split_long(paragraph, limit)]
        current: list[str] = []
        fresh = 0  # paragraphs in current that no earlier chunk holds
        for part in parts:
            if current and len("\n\n".join([*current, part])) > limit:
                chunks.append(_chunk(document, section.heading, section.page, current, len(chunks)))
                overlap = settings.overlap_paragraphs
                current = current[-overlap:] if overlap else []
                # Drop overlap that would push the next chunk over the limit.
                while current and len("\n\n".join([*current, part])) > limit:
                    current.pop(0)
                fresh = 0
            current.append(part)
            fresh += 1
        if fresh:
            chunks.append(_chunk(document, section.heading, section.page, current, len(chunks)))
    return chunks


def _chunk(document: Document, heading: str | None, page: int | None, paragraphs: list[str],
           position: int) -> Chunk:
    text = "\n\n".join(paragraphs)
    return Chunk(
        # Stable for unchanged text, so re-indexing an unchanged corpus rewrites the same points.
        id=str(uuid5(NAMESPACE_URL, f"{document.source}#{position}:{text}")),
        text=text,
        source=document.source,
        origin=document.origin,
        title=document.title,
        section=heading,
        page=page,
        position=position,
    )
