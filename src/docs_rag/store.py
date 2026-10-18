"""Qdrant storage: one point per chunk with a dense E5 vector and a BM25 sparse vector.

The embedded store (data/qdrant) can be opened by one process at a time, so callers open it only
for the duration of an operation (`open_store`) and the MCP server never blocks `docs-rag index`
for longer than one search.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient, models

from .bm25 import Bm25
from .config import Settings
from .schemas import Chunk, Citation

UPSERT_BATCH = 64


class TextEncoder(Protocol):
    dimension: int

    def passages(self, texts: list[str]) -> np.ndarray: ...

    def queries(self, texts: list[str]) -> np.ndarray: ...


class StoreBusyError(RuntimeError):
    pass


class Store:
    def __init__(
        self, settings: Settings, client: QdrantClient, encoder: TextEncoder | None = None
    ) -> None:
        self.settings = settings
        self.client = client
        self.encoder = encoder
        self.bm25 = Bm25(settings.bm25)
        self.collection = settings.collection

    def _ensure_collection(self) -> None:
        """Create the collection; recreate one with another schema or vector size (for example
        after changing embedding.model), since its vectors are unusable."""
        assert self.encoder is not None, "indexing needs an encoder"
        size = self.encoder.dimension
        if self.client.collection_exists(self.collection):
            params = self.client.get_collection(self.collection).config.params
            dense = params.vectors.get("dense") if isinstance(params.vectors, dict) else None
            if dense and dense.size == size and "bm25" in (params.sparse_vectors or {}):
                return
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={
                "dense": models.VectorParams(size=size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )

    def index(self, chunks: list[Chunk]) -> int:
        """Make the collection hold exactly these chunks. New points are written first and stale
        ones deleted afterwards, so a failed run never leaves the index empty."""
        self._ensure_collection()
        # BM25 length normalisation uses the whole corpus's average chunk length.
        sparse = self.bm25.documents([chunk.text for chunk in chunks])
        for start in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[start : start + UPSERT_BATCH]
            dense = self.encoder.passages([chunk.text for chunk in batch])
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=chunk.id,
                        vector={
                            "dense": dense[i].tolist(),
                            "bm25": models.SparseVector(
                                indices=sparse[start + i][0], values=sparse[start + i][1]
                            ),
                        },
                        payload=chunk.model_dump(),
                    )
                    for i, chunk in enumerate(batch)
                ],
                wait=True,
            )
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must_not=[models.HasIdCondition(has_id=[chunk.id for chunk in chunks])]
                )
            ),
            wait=True,
        )
        return len(chunks)

    def search(self, query: str, limit: int, origins: list[str] | None = None) -> list[Citation]:
        """Dense and BM25 candidates fused with reciprocal rank fusion (RRF), then filtered by the
        dense cosine similarity so that an unrelated query returns nothing."""
        if not self.client.collection_exists(self.collection):
            return []
        assert self.encoder is not None, "searching needs an encoder"
        dense = self.encoder.queries([query])[0]
        indices, values = self.bm25.query(query)
        where = (
            models.Filter(must=[models.FieldCondition(key="origin",
                                                      match=models.MatchAny(any=origins))])
            if origins
            else None
        )
        candidates = self.settings.search.candidates
        prefetch = [
            models.Prefetch(query=dense.tolist(), using="dense", limit=candidates, filter=where)
        ]
        if indices:  # a query of one-letter words has no BM25 terms
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="bm25", limit=candidates, filter=where,
                )
            )
        points = self.client.query_points(
            self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=["dense"],
        ).points
        citations = []
        for point in points:
            vector = point.vector["dense"] if isinstance(point.vector, dict) else point.vector
            similarity = float(np.dot(np.asarray(vector, dtype=np.float32), dense))
            if similarity < self.settings.search.min_score:
                continue
            payload = point.payload or {}
            citations.append(
                Citation(
                    title=payload["title"], source=payload["source"],
                    section=payload.get("section"), page=payload.get("page"),
                    text=payload["text"], score=round(similarity, 4),
                )
            )
        return citations

    def status(self) -> dict[str, int | str]:
        points = (
            self.client.count(self.collection, exact=True).count
            if self.client.collection_exists(self.collection)
            else 0
        )
        return {"collection": self.collection, "points": points}


@contextmanager
def open_store(settings: Settings, encoder: TextEncoder | None = None) -> Iterator[Store]:
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    try:
        client = QdrantClient(path=str(settings.qdrant_path))
    except RuntimeError as error:  # the storage folder is locked by another process
        raise StoreBusyError(
            f"The index at {settings.qdrant_path} is in use by another process (probably "
            "`docs-rag index`); try again when it finishes."
        ) from error
    try:
        yield Store(settings, client, encoder)
    finally:
        client.close()
