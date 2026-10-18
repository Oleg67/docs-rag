"""Settings come from one YAML file: `config.yaml` in the project root, or DOCS_RAG_CONFIG.

Relative paths in it (data_dir, folder sources) resolve against the file's directory, so the CLI
and the MCP server find the same data whatever their working directory.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in config.yaml is an error, not a default


class WebSource(_Strict):
    type: Literal["web"]
    name: str
    url: HttpUrl
    # Only URLs under this prefix are crawled; defaults to the start URL's directory.
    path_prefix: str | None = None
    max_pages: int = Field(default=100, ge=1)
    pdfs: bool = True  # also fetch linked PDFs on the same host


class FolderSource(_Strict):
    type: Literal["folder"]
    name: str
    path: Path
    patterns: list[str] = ["**/*.pdf", "**/*.md", "**/*.txt", "**/*.html", "**/*.htm"]


Source = Annotated[WebSource | FolderSource, Field(discriminator="type")]


class EmbeddingSettings(_Strict):
    model_config = ConfigDict(extra="forbid", frozen=True)  # hashable: the key of get_encoder

    model: str = "intfloat/multilingual-e5-base"
    # E5 models are trained with these prefixes; set both to "" for models without them.
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    device: str | None = None  # None picks CUDA when available
    batch_size: int = Field(default=32, ge=1)


class Bm25Settings(_Strict):
    # Snowball stemmer language (PyStemmer), e.g. "english" or "russian"; None disables stemming.
    language: str | None = "english"
    k1: float = Field(default=1.2, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)


class ChunkingSettings(_Strict):
    # 1,600 characters stay within E5's 512-token window for typical prose.
    max_characters: int = Field(default=1_600, ge=200)
    overlap_paragraphs: int = Field(default=1, ge=0)


class SearchSettings(_Strict):
    limit: int = Field(default=6, ge=1, le=50)
    # Cutoff on the dense cosine similarity. E5 scores unrelated text at about 0.75, so lower
    # values filter almost nothing.
    min_score: float = Field(default=0.80, ge=-1, le=1)
    candidates: int = Field(default=50, ge=1)  # per retriever, before RRF fusion


class CrawlerSettings(_Strict):
    user_agent: str = "docs-rag/0.1 (local research crawler)"
    delay_seconds: float = Field(default=1.0, ge=1.0)  # polite: never faster than 1 request/s
    timeout_seconds: float = Field(default=30, gt=0)
    max_pdf_bytes: int = Field(default=50_000_000, ge=1)
    # Refuse to save a crawl that lost more pages than this to outages or rate limiting.
    max_failed_fraction: float = Field(default=0.05, ge=0, le=1)


class Settings(_Strict):
    data_dir: Path = Path("data")
    collection: str = "docs"
    sources: list[Source] = []
    embedding: EmbeddingSettings = EmbeddingSettings()
    bm25: Bm25Settings = Bm25Settings()
    chunking: ChunkingSettings = ChunkingSettings()
    search: SearchSettings = SearchSettings()
    crawler: CrawlerSettings = CrawlerSettings()

    @property
    def documents_path(self) -> Path:
        return self.data_dir / "documents.jsonl"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"


def config_path() -> Path:
    return Path(os.environ.get("DOCS_RAG_CONFIG") or PROJECT_ROOT / "config.yaml")


def load_settings(path: Path) -> Settings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = Settings.model_validate(raw)
    base = path.resolve().parent

    def resolve(value: Path) -> Path:
        value = value.expanduser()
        return value if value.is_absolute() else base / value

    settings.data_dir = resolve(settings.data_dir)
    for source in settings.sources:
        if isinstance(source, FolderSource):
            source.path = resolve(source.path)
    names = [source.name for source in settings.sources]
    if len(names) != len(set(names)):
        raise ValueError(f"Source names must be unique: {names}")
    return settings


@lru_cache
def get_settings() -> Settings:
    return load_settings(config_path())
