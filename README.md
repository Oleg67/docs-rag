# docs-rag

Local retrieval over web pages and your own documents, served to Claude Code as an MCP server.

- **Ingest:** crawl web sites politely (same host, a path prefix, robots.txt, at most one request per second, linked PDFs) and parse local folders of PDF, Markdown, HTML and text files.
- **Index:** section-aware chunks (they never cross a heading or a PDF page), each with a dense multilingual-E5 vector and a BM25 sparse vector, in an embedded Qdrant store. No Docker and no server.
- **Search:** hybrid retrieval. Dense and BM25 candidates are fused with reciprocal rank fusion (RRF), then passages below a dense-similarity cutoff are dropped, so an unrelated question returns nothing rather than noise.
- **MCP:** `search`, `list_documents`, `get_document` and `index_status` tools. Every passage is cited with its URL or file path and, for PDFs, its page.

There is no answer generation: the client's model (Claude) writes the answer from the cited passages.

## Installation

```sh
cd docs-rag
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
cp config.example.yaml config.yaml
```

The embedding model (`intfloat/multilingual-e5-base`, about 1 GB) downloads on first use. A GPU is used when available; a CPU works for small corpora.

## Configuration

Everything is in `config.yaml` (git-ignored; `config.example.yaml` documents every option). `-c PATH` or `DOCS_RAG_CONFIG` selects another file. Relative paths resolve against the config file's directory.

```yaml
sources:
  - name: papers            # unique; used by `ingest -s` and the search origin filter
    type: folder
    path: ~/Documents/papers
  - name: guide
    type: web
    url: https://docs.example.com/guide/
    max_pages: 100           # path_prefix defaults to /guide/; pdfs: true also fetches linked PDFs
bm25:
  language: english          # Snowball stemmer for BM25: english, russian, german, ...
```

Unknown keys are errors, so a typo cannot silently fall back to a default.

## Usage

```sh
. .venv/bin/activate
docs-rag ingest              # all sources -> data/documents.jsonl
docs-rag ingest -s papers    # refresh one source and keep the others
docs-rag index               # chunk, embed and replace the index
docs-rag search "how is the lactate threshold measured?"
docs-rag status
```

`ingest` saves nothing when a source fails, yields no documents, or loses more than 5% of its pages to server errors, so a bad run never replaces good data. Sources removed from `config.yaml` are dropped. `index` writes the new points before deleting stale ones, so a failed run leaves the old index searchable.

Scanned PDFs have no text layer and are skipped with a warning; run OCR (for example `ocrmypdf`) first.

## Claude Code

Register the server once for all projects:

```sh
claude mcp add docs-rag --scope user -- /path/to/docs-rag/.venv/bin/docs-rag mcp
```

Run `claude mcp list` to check it, or `/mcp` inside Claude Code. Then ask Claude, for example: "Search my docs for how the lactate threshold is measured and cite the sources."

The server opens the embedded index only while a tool runs, so `docs-rag ingest` and `docs-rag index` work while Claude Code is connected. A search that coincides with indexing returns a "try again" error. The first search loads the model (about 8 s); later ones take about 0.25 s on a 2,000-chunk corpus. Opening the store costs time in proportion to its size, so for corpora of hundreds of thousands of chunks a Qdrant server would be the better choice.

| Tool | Purpose |
|---|---|
| `search(query, limit?, origins?)` | Hybrid search; returns passages with source, section, page and dense score |
| `list_documents(origin?)` | Ingested documents with their source, kind, title and size |
| `get_document(source, offset?, max_characters?)` | The full text, in pages of up to 100k characters |
| `index_status()` | Collection, passage and document counts |

## Retrieval details

- Dense: multilingual-E5 with its `query: ` / `passage: ` prefixes, cosine similarity. It works across languages, so an English question finds Russian text and vice versa.
- BM25: lowercase word tokens, Snowball stemming, k1 = 1.2 and b = 0.75 with the corpus's average chunk length. Qdrant applies IDF at query time (`Modifier.IDF`), so there is no vocabulary to maintain. BM25 catches exact terms (names, codes, numbers) that dense vectors blur.
- Cutoff: `search.min_score` (0.80) applies to the dense cosine. E5 scores unrelated text at about 0.75, so lower values filter almost nothing.
- Changing `embedding.model` changes the vector size; the next `index` rebuilds the collection.

## Development

```sh
pytest            # offline: fake encoder, in-memory Qdrant, mock HTTP transport
ruff check .
```

## Project structure

```text
src/docs_rag/
  config.py      config.yaml schema and path resolution
  crawler.py     robots-aware same-host web crawler
  parsers.py     HTML, Markdown, text and PDF to sections
  ingest.py      sources -> documents.jsonl (folders, merge, atomic write)
  chunking.py    section-aware chunks with paragraph overlap
  bm25.py        BM25 sparse vectors
  embeddings.py  sentence-transformers encoder
  store.py       Qdrant index and hybrid search
  mcp_server.py  MCP tools
  cli.py         ingest, index, search, status, mcp
```

## License

Apache License 2.0; see [LICENSE](LICENSE).
