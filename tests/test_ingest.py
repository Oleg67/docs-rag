from pathlib import Path

import pytest

from docs_rag.config import FolderSource, load_settings
from docs_rag.ingest import collect, merge, read_documents, read_folder, write_documents

from conftest import make_document


def test_folder_reads_supported_files(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("# Alpha\n\nText.", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("Plain text.", encoding="utf-8")
    (tmp_path / "sub" / "c.html").write_text("<title>Gamma</title><p>Hi.</p>", encoding="utf-8")
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    documents = read_folder(FolderSource(type="folder", name="inbox", path=tmp_path))

    assert [(d.title, d.kind) for d in documents] == [
        ("Alpha", "markdown"), ("b", "text"), ("Gamma", "html"),
    ]
    assert documents[0].source == str((tmp_path / "a.md").resolve())


def test_config_paths_resolve_against_the_config_file(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("sources:\n  - {name: inbox, type: folder, path: docs}\n", encoding="utf-8")

    settings = load_settings(config)

    assert settings.data_dir == tmp_path / "data"
    assert settings.sources[0].path == tmp_path / "docs"


def test_config_rejects_typos_and_duplicate_names(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("serch: {limit: 3}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="serch"):
        load_settings(config)
    config.write_text(
        "sources:\n  - {name: a, type: folder, path: x}\n  - {name: a, type: folder, path: y}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_settings(config)


def test_empty_source_is_an_error(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    (tmp_path / "docs").mkdir()
    config.write_text("sources:\n  - {name: inbox, type: folder, path: docs}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no documents"):
        collect(load_settings(config))


def test_merge_replaces_refreshed_sources_and_drops_removed_ones() -> None:
    old_a, old_b, gone = (make_document(s, "t", origin=o) for s, o in
                          [("a1", "a"), ("b1", "b"), ("c1", "c")])
    new_a = make_document("a2", "t", origin="a")

    merged = merge([old_a, old_b, gone], {"a": [new_a]}, {"a", "b"})

    assert [d.source for d in merged] == ["b1", "a2"]


def test_documents_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    write_documents([make_document("a", "x", page=2)], path)
    assert read_documents(path)[0].sections[0].page == 2
    assert not path.with_suffix(".tmp").exists()
