import pytest
from qdrant_client import QdrantClient

from docs_rag.chunking import chunk_documents
from docs_rag.config import Settings
from docs_rag.store import Store, StoreBusyError, open_store

from conftest import FakeEncoder, make_document

DOCUMENTS = [
    make_document("https://x.org/lactate", "Lactate threshold rises with endurance training.",
                  origin="web", title="Lactate"),
    make_document("/docs/sleep.pdf", "Sleep improves recovery after hard sessions.",
                  origin="inbox", title="Sleep", page=4),
]


@pytest.fixture
def store(settings: Settings, encoder: FakeEncoder) -> Store:
    store = Store(settings, QdrantClient(":memory:"), encoder)
    store.index(chunk_documents(DOCUMENTS, settings.chunking))
    return store


def test_hybrid_search_ranks_the_matching_passage_first(store: Store) -> None:
    results = store.search("lactate threshold", 5)

    assert results[0].source == "https://x.org/lactate"
    assert results[0].score > 0.3


def test_citation_keeps_pdf_page(store: Store) -> None:
    result = store.search("sleep recovery", 1)[0]
    assert (result.source, result.page, result.section) == ("/docs/sleep.pdf", 4, "S0")


def test_cutoff_drops_unrelated_passages(store: Store) -> None:
    assert store.search("quantum chromodynamics", 5) == []


def test_origin_filter(store: Store) -> None:
    results = store.search("lactate threshold sleep recovery", 5, origins=["inbox"])
    assert {result.source for result in results} == {"/docs/sleep.pdf"}


def test_reindex_removes_stale_chunks(store: Store, settings: Settings) -> None:
    store.index(chunk_documents(DOCUMENTS[1:], settings.chunking))

    assert store.status()["points"] == 1
    assert store.search("lactate threshold", 5) == []


def test_new_vector_size_rebuilds_the_collection(store: Store, settings: Settings) -> None:
    class WiderEncoder(FakeEncoder):
        dimension = 32

    store.encoder = WiderEncoder()
    store.index(chunk_documents(DOCUMENTS, settings.chunking))
    assert store.status()["points"] == 2


def test_open_store_reports_a_locked_index(settings: Settings, encoder: FakeEncoder) -> None:
    with open_store(settings, encoder):
        with pytest.raises(StoreBusyError, match="in use"):
            with open_store(settings, encoder):
                pass
