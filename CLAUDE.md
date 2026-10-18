# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Local retrieval over crawled web pages and local documents (PDF, Markdown, HTML, text), served as a stdio MCP server for Claude Code. Retrieval only: there is no answer generation. It follows the design of `~/sports-physiology-rag`, generalised to any source and language.

## Commands

```sh
. .venv/bin/activate                 # uv pip install --python .venv/bin/python -e '.[dev]'
pytest                               # offline; no models or network
ruff check .                         # line length 100

docs-rag [-c config.yaml] ingest [-s NAME]   # sources -> data/documents.jsonl
docs-rag index                       # chunk + embed + replace the Qdrant index
docs-rag search "query" [-n 5]
docs-rag status
docs-rag mcp                         # stdio MCP server; stdout is the protocol, never print
```

## Architecture

`config.yaml` (sources) → `ingest.py` (`crawler.py` for web sources, `parsers.py` for files) → `data/documents.jsonl` → `chunking.py` → `store.py` (embedded Qdrant in `data/qdrant`) → `mcp_server.py` / `cli.py search`.

- **Config:** `config.py` parses `config.yaml` with pydantic (`extra="forbid"`). Relative paths resolve against the config file. `get_settings()` is cached; `DOCS_RAG_CONFIG` or `-c` selects the file. `config.yaml` is git-ignored and `config.example.yaml` is the template; keep the two in sync.
- **Documents:** a `Document` has `sections` (heading, page, paragraphs separated by blank lines). `source` is the URL or absolute path and is the citation target. `origin` is the config source name. `ingest` replaces only the sources it collected, and refuses to save when a source fails or is empty.
- **Crawler:** BFS from the start URL, same host and path prefix (same-host PDFs may leave the prefix), robots.txt per RFC 9309 (fails closed on 5xx), `max(delay, Crawl-delay)` after every request, and a request hook that also checks redirects. Keep it polite: never below 1 s per request.
- **Store:** one point per chunk with a `dense` vector (cosine) and a `bm25` sparse vector (`Modifier.IDF`; documents carry saturated TF, token ids are CRC32). Search is two prefetches fused by RRF, then a cutoff on the dense cosine (`search.min_score`). `index()` upserts first, then deletes points not in the new id set. `open_store()` opens the embedded store per operation, because only one process can hold it; the MCP server opens it per tool call.
- **MCP:** `mcp` 2.x SDK (`MCPServer`, snake_case fields such as `read_only_hint` and `structured_content`). Unexpected errors are re-raised as `ToolError` so the client sees the message.

## Conventions

- Tests use `tests/conftest.py`: `FakeEncoder` (bag of words), `QdrantClient(":memory:")` and `httpx.MockTransport`. Keep them offline.
- Every network call has a timeout (`crawler.timeout_seconds`).
