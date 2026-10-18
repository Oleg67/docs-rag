import logging
import os
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from .config import Settings, config_path, get_settings

app = typer.Typer(help="Local RAG over crawled web pages and documents, served over MCP.")


@app.callback()
def main(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="config.yaml to use (or DOCS_RAG_CONFIG).")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    if config:
        os.environ["DOCS_RAG_CONFIG"] = str(config.resolve())
        get_settings.cache_clear()
    # Logs go to stderr, which keeps stdout clean for the MCP protocol.
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")


def _settings() -> Settings:
    try:
        return get_settings()
    except FileNotFoundError as error:
        typer.echo(f"{config_path()} not found; copy config.example.yaml to it.", err=True)
        raise typer.Exit(1) from error
    except (ValidationError, ValueError) as error:
        typer.echo(f"{config_path()} is not valid:\n{error}", err=True)
        raise typer.Exit(1) from error


@app.command()
def ingest(
    source: Annotated[
        list[str] | None, typer.Option("--source", "-s", help="Only these source names.")
    ] = None,
) -> None:
    """Crawl web sources and parse folder sources into data/documents.jsonl."""
    from .ingest import collect, merge, read_documents, write_documents

    settings = _settings()
    if not settings.sources:
        typer.echo(f"No sources in {config_path()}.", err=True)
        raise typer.Exit(1)
    try:
        collected = collect(settings, source)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        typer.echo(f"{error}\nNothing was saved.", err=True)
        raise typer.Exit(1) from error
    path = settings.documents_path
    previous = read_documents(path) if path.exists() else []
    documents = merge(previous, collected, {s.name for s in settings.sources})
    write_documents(documents, path)
    for name, items in collected.items():
        typer.echo(f"{name}: {len(items)} documents")
    typer.echo(f"Saved {len(documents)} documents to {path}. Run `docs-rag index` next.")


@app.command()
def index() -> None:
    """Chunk, embed and index data/documents.jsonl, replacing the previous index."""
    from .chunking import chunk_documents
    from .embeddings import get_encoder
    from .ingest import read_documents
    from .store import StoreBusyError, open_store

    settings = _settings()
    path = settings.documents_path
    if not path.exists():
        typer.echo(f"{path} not found. Run `docs-rag ingest` first.", err=True)
        raise typer.Exit(1)
    chunks = chunk_documents(read_documents(path), settings.chunking)
    if not chunks:
        typer.echo(f"No text to index in {path}.", err=True)
        raise typer.Exit(1)
    encoder = get_encoder(settings.embedding)
    try:
        with open_store(settings, encoder) as store:
            count = store.index(chunks)
    except StoreBusyError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Indexed {count} chunks into {settings.collection}.")


@app.command()
def status() -> None:
    """Show the collection and its number of indexed chunks."""
    from .store import open_store

    with open_store(_settings()) as store:
        typer.echo(store.status())


@app.command()
def search(
    query: str,
    limit: Annotated[int | None, typer.Option("--limit", "-n")] = None,
) -> None:
    """Search from the terminal (the same search the MCP server runs)."""
    from .embeddings import get_encoder
    from .store import open_store

    settings = _settings()
    with open_store(settings, get_encoder(settings.embedding)) as store:
        results = store.search(query, limit or settings.search.limit)
    if not results:
        typer.echo("No passage passes the relevance cutoff (search.min_score).")
    for result in results:
        where = f" p. {result.page}" if result.page else ""
        typer.echo(f"[{result.score:.3f}] {result.title} — {result.source}{where}")
        typer.echo(f"    {result.text[:300]}\n")


@app.command("mcp")
def mcp_server() -> None:
    """Run the MCP server over stdio (Claude Code starts it)."""
    _settings()  # fail on a bad config before the client connects
    from .mcp_server import server

    server.run("stdio")
