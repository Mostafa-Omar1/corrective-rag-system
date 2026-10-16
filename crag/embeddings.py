"""Embedding backends.

The default is a local sentence-transformers model (no API key needed).
If it cannot be loaded (offline machine, tiny CI container) we fall back to a
hashing TF-IDF style embedder so the pipeline still runs end to end.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")


class HashingEmbedder:
    """Dependency-light fallback: hashed unigrams + bigrams, L2-normalised."""

    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _tokens(self, text: str) -> list[str]:
        words = self._TOKEN.findall(text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        import zlib

        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for tok in self._tokens(text):
                h = zlib.crc32(tok.encode())
                out[row, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


@lru_cache(maxsize=2)
def get_embedder(model_name: str):
    if model_name.startswith("hashing"):
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("Falling back to HashingEmbedder (%s)", exc)
        return HashingEmbedder()
