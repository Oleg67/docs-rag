"""Stdio MCP server for Claude Code: search the indexed documents and read them in full.

It returns passages with their source (URL or file path); the client's model writes the answer.
Stdout carries the protocol, so nothing here may print to it.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .config import get_settings
from .embeddings import get_encoder
from .ingest import read_documents
from .schemas import Document, SearchResponse
from .store import open_store

server = MCPServer(
    "docs-rag",
    instructions=(
        "Search a local index of crawled web pages and local documents (PDF, Markdown, HTML, "
        "text). Use search for passages, then get_document to read around a result. Cite each "
        "passage's source (URL or file path, plus page for PDFs). If search returns nothing, say "
        "the indexed documents do not cover it."
    ),
)

# Tools run in worker threads; one store at a time, since the embedded Qdrant folder is locked
# by whoever opens it.
_store_lock = threading.Lock()
_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


@contextmanager
def _reported() -> Iterator[None]:
    """Pass errors to the client, which would otherwise see only "Error executing tool"."""
    try:
        yield
    except ToolError:
        raise
    except Exception as error:
        raise ToolError(f"{type(error).__name__}: {error}") from error


def _documents() -> list[Document]:
    path = get_settings().documents_path
    if not path.exists():
        raise ToolError(f"{path} not found; run `docs-rag ingest` and `docs-rag index` first.")
    with _reported():
        return read_documents(path)


@server.tool(annotations=_READ_ONLY)
def search(
    query: Annotated[str, Field(min_length=2, max_length=1_000)],
    limit: Annotated[int | None, Field(ge=1, le=50)] = None,
    origins: list[str] | None = None,
) -> SearchResponse:
    """Find passages relevant to a query (hybrid: semantic vectors + BM25 keywords).

    Args:
        query: A question or keywords.
        limit: Number of passages; defaults to search.limit in config.yaml.
        origins: Restrict to these config.yaml source names (see list_documents).

    Passages below the relevance cutoff are dropped, so an empty result means nothing in the
    index is close enough.
    """
    settings = get_settings()
    query = query.strip()
    with _reported():
        encoder = get_encoder(settings.embedding)  # loaded once, before taking the store lock
        with _store_lock, open_store(settings, encoder) as store:
            results = store.search(query, limit or settings.search.limit, origins)
    return SearchResponse(query=query, results=results)


class DocumentEntry(BaseModel):
    source: str
    origin: str
    kind: str
    title: str
    characters: int


class DocumentList(BaseModel):
    documents: list[DocumentEntry]


@server.tool(annotations=_READ_ONLY)
def list_documents(origin: str | None = None) -> DocumentList:
    """List the indexed documents (source URL or path, title, size), optionally of one source."""
    return DocumentList(
        documents=[
            DocumentEntry(
                source=d.source, origin=d.origin, kind=d.kind, title=d.title,
                characters=len(d.text),
            )
            for d in _documents()
            if origin is None or d.origin == origin
        ]
    )


class DocumentText(BaseModel):
    source: str
    title: str
    text: str
    offset: int
    total_characters: int
    next_offset: int | None  # pass as offset to read on; None at the end


@server.tool(annotations=_READ_ONLY)
def get_document(
    source: str,
    offset: Annotated[int, Field(ge=0)] = 0,
    max_characters: Annotated[int, Field(ge=1_000, le=100_000)] = 20_000,
) -> DocumentText:
    """Read a document's text in pages of max_characters (PDF pages are marked "[page N]").

    Args:
        source: A search result's or list_documents entry's source.
        offset: Character offset to start from; use the previous call's next_offset.
        max_characters: How much to return.
    """
    for document in _documents():
        if document.source == source.strip():
            text = document.text
            end = min(offset + max_characters, len(text))
            return DocumentText(
                source=document.source, title=document.title, text=text[offset:end],
                offset=offset, total_characters=len(text),
                next_offset=end if end < len(text) else None,
            )
    raise ToolError(f"No document {source!r}; take sources from search or list_documents.")


@server.tool(annotations=_READ_ONLY)
def index_status() -> dict[str, int | str]:
    """Report the collection, the number of indexed passages and of ingested documents."""
    settings = get_settings()
    with _reported(), _store_lock, open_store(settings) as store:
        status = store.status()
    path = settings.documents_path
    status["documents"] = len(read_documents(path)) if path.exists() else 0
    return status
