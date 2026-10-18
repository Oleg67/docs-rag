from functools import lru_cache

import numpy as np

from .config import EmbeddingSettings


class Encoder:
    """Sentence-transformers model with E5-style query/passage prefixes; vectors are unit-length."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        from sentence_transformers import SentenceTransformer  # slow import; only when needed

        self.settings = settings
        self._model = SentenceTransformer(settings.model, device=settings.device)
        # sentence-transformers 5 renamed the method.
        dimension = getattr(self._model, "get_embedding_dimension", None)
        self.dimension = (dimension or self._model.get_sentence_embedding_dimension)()

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, batch_size=self.settings.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)

    def passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.settings.passage_prefix + text for text in texts])

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.settings.query_prefix + text for text in texts])


@lru_cache
def get_encoder(settings: EmbeddingSettings) -> Encoder:
    return Encoder(settings)
