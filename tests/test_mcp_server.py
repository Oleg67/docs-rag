import asyncio
from contextlib import contextmanager

import pytest
from mcp import Client
from qdrant_client import QdrantClient

from docs_rag import mcp_server
from docs_rag.chunking import chunk_documents
from docs_rag.config import Settings
from docs_rag.ingest import write_documents
from docs_rag.store import Store

from conftest import FakeEncoder, make_document


@pytest.fixture(autouse=True)
def indexed(settings: Settings, encoder: FakeEncoder, monkeypatch: pytest.MonkeyPatch) -> None:
    documents = [
        make_document("https://x.org/lactate", "Lactate threshold rises with training.",
                      origin="web", title="Lactate"),
        make_document("/docs/long.md", "word " * 5_000, title="Long"),
    ]
    write_documents(documents, settings.documents_path)
    client = QdrantClient(":memory:")
    Store(settings, client, encoder).index(chunk_documents(documents, settings.chunking))

    @contextmanager
    def fake_open_store(settings_, encoder_=None):
        yield Store(settings_, client, encoder_)

    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)
    monkeypatch.setattr(mcp_server, "get_encoder", lambda _: encoder)
    monkeypatch.setattr(mcp_server, "open_store", fake_open_store)


def _with_client(action):
    async def run():
        async with Client(mcp_server.server) as client:
            return await action(client)

    return asyncio.run(run())


def _call(name: str, arguments: dict | None = None):
    return _with_client(lambda client: client.call_tool(name, arguments or {}))


def test_tools_are_read_only() -> None:
    tools = _with_client(lambda client: client.list_tools()).tools
    assert {tool.name for tool in tools} == {
        "search", "list_documents", "get_document", "index_status",
    }
    assert all(tool.annotations.read_only_hint for tool in tools)


def test_search_returns_cited_passages() -> None:
    result = _call("search", {"query": " lactate threshold ", "limit": 3})

    assert not result.is_error
    assert result.structured_content["query"] == "lactate threshold"
    assert result.structured_content["results"][0]["source"] == "https://x.org/lactate"


def test_search_validates_arguments() -> None:
    assert _call("search", {"query": "x"}).is_error
    assert _call("search", {"query": "lactate", "limit": 0}).is_error


def test_list_documents_by_origin() -> None:
    result = _call("list_documents", {"origin": "web"})
    assert [d["title"] for d in result.structured_content["documents"]] == ["Lactate"]


def test_get_document_pages_through_long_text() -> None:
    first = _call("get_document", {"source": "/docs/long.md", "max_characters": 10_000})
    content = first.structured_content
    assert len(content["text"]) == 10_000 and content["next_offset"] == 10_000

    rest = _call("get_document", {"source": "/docs/long.md", "offset": content["next_offset"],
                                  "max_characters": 100_000}).structured_content
    assert rest["next_offset"] is None
    assert len(content["text"]) + len(rest["text"]) == content["total_characters"]


def test_unknown_document_and_missing_corpus(settings: Settings) -> None:
    assert "No document" in _call("get_document", {"source": "nope"}).content[0].text
    settings.documents_path.unlink()
    assert "docs-rag ingest" in _call("list_documents").content[0].text


def test_index_status() -> None:
    status = _call("index_status").structured_content
    assert status["collection"] == "test" and status["documents"] == 2 and status["points"] > 2
