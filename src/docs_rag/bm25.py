"""BM25 as Qdrant sparse vectors: documents carry the saturated term frequency, and Qdrant
applies IDF at query time (sparse vector `modifier=IDF`), so the index needs no vocabulary."""

import re
import zlib
from collections import Counter

import Stemmer

from .config import Bm25Settings

_WORD = re.compile(r"\w+")


class Bm25:
    def __init__(self, settings: Bm25Settings) -> None:
        self.settings = settings
        self._stemmer = Stemmer.Stemmer(settings.language) if settings.language else None

    def tokens(self, text: str) -> list[str]:
        words = [word for word in _WORD.findall(text.lower()) if len(word) > 1 or word.isdigit()]
        return self._stemmer.stemWords(words) if self._stemmer else words

    @staticmethod
    def _index(token: str) -> int:
        return zlib.crc32(token.encode())  # stable across processes, unlike hash()

    def _vector(self, weights: Counter[int] | dict[int, float]) -> tuple[list[int], list[float]]:
        indices = sorted(weights)
        return indices, [float(weights[i]) for i in indices]

    def documents(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        tokenised = [self.tokens(text) for text in texts]
        average = sum(map(len, tokenised)) / max(len(tokenised), 1) or 1.0
        k1, b = self.settings.k1, self.settings.b
        vectors = []
        for tokens in tokenised:
            counts: Counter[int] = Counter(self._index(token) for token in tokens)
            norm = k1 * (1 - b + b * len(tokens) / average)
            weights = {i: tf * (k1 + 1) / (tf + norm) for i, tf in counts.items()}
            vectors.append(self._vector(weights))
        return vectors

    def query(self, text: str) -> tuple[list[int], list[float]]:
        return self._vector({self._index(token): 1.0 for token in self.tokens(text)})
