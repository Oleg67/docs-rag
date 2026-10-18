import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from docs_rag.config import Settings
from docs_rag.schemas import Document, Section


class FakeEncoder:
    """Bag-of-words vectors: texts sharing words are similar. No model download."""

    dimension = 64

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, np.float32)
        for word in text.lower().split():
            vector[zlib.crc32(word.strip(".,?!").encode()) % self.dimension] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def queries(self, texts: list[str]) -> np.ndarray:
        return self.passages(texts)


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings.model_validate(
        {"data_dir": str(tmp_path / "data"), "collection": "test", "search": {"min_score": 0.1}}
    )


def make_document(source: str, *sections: str, origin: str = "inbox", title: str = "Doc",
                  page: int | None = None) -> Document:
    return Document(
        source=source, origin=origin, kind="markdown", title=title,
        sections=[
            Section(heading=f"S{i}", page=page, text=text) for i, text in enumerate(sections)
        ],
        fetched_at=datetime.now(UTC),
    )
