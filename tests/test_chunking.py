from docs_rag.chunking import chunk_document
from docs_rag.config import ChunkingSettings

from conftest import make_document


def test_chunks_respect_the_limit_even_for_unpunctuated_text() -> None:
    settings = ChunkingSettings(max_characters=200, overlap_paragraphs=1)
    long_paragraphs = "\n\n".join(["word " * 39] * 3)  # ~195 characters each
    document = make_document("a.md", long_paragraphs, "x" * 450, "list item " * 60)

    chunks = chunk_document(document, settings)

    assert chunks and all(len(chunk.text) <= 200 for chunk in chunks)


def test_overlap_and_sections() -> None:
    settings = ChunkingSettings(max_characters=200, overlap_paragraphs=1)
    paragraphs = "\n\n".join(f"Paragraph {i} " + "text " * 15 for i in range(4))
    document = make_document("a.md", paragraphs, "Other section.", page=3)

    chunks = chunk_document(document, settings)

    first, second = chunks[0].text.split("\n\n"), chunks[1].text.split("\n\n")
    assert first[-1] == second[0]  # one paragraph of overlap
    assert chunks[-1].text == "Other section." and chunks[-1].section == "S1"
    assert {chunk.page for chunk in chunks} == {3}
    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.id for chunk in chunks}) == len(chunks)
