"""Persistent FAISS vector database with chunk metadata."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss

from .ingestion import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


class VectorStore:
    INDEX_FILE = "index.faiss"
    META_FILE = "chunks.json"

    def __init__(self, embedder, storage_dir: Path | None = None):
        self.embedder = embedder
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.index = faiss.IndexFlatIP(embedder.dim)  # inner product on unit vectors = cosine
        self.chunks: list[Chunk] = []
        if self.storage_dir:
            self._load()

    # ------------------------------------------------------------------ write
    def add(self, chunks: list[Chunk]) -> int:
        """Embed and index chunks, skipping documents that are already stored."""
        known = {c.metadata.get("doc_hash") for c in self.chunks}
        new = [c for c in chunks if c.metadata.get("doc_hash") not in known]
        if not new:
            return 0
        vectors = self.embedder.embed([c.text for c in new])
        self.index.add(vectors)
        self.chunks.extend(new)
        self._save()
        return len(new)

    def clear(self) -> None:
        self.index.reset()
        self.chunks = []
        self._save()

    # ------------------------------------------------------------------- read
    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if not self.chunks:
            return []
        vector = self.embedder.embed([query])
        scores, ids = self.index.search(vector, min(k, len(self.chunks)))
        return [
            SearchResult(self.chunks[i], float(s))
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]

    @property
    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------ persistence
    def _meta_path(self) -> Path:
        return self.storage_dir / self.META_FILE

    def _save(self) -> None:
        if not self.storage_dir:
            return
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.storage_dir / self.INDEX_FILE))
        payload = {"embedder": self.embedder.name, "chunks": [asdict(c) for c in self.chunks]}
        self._meta_path().write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _load(self) -> None:
        index_path = self.storage_dir / self.INDEX_FILE
        if not (index_path.exists() and self._meta_path().exists()):
            return
        payload = json.loads(self._meta_path().read_text(encoding="utf-8"))
        if payload.get("embedder") != self.embedder.name:
            return  # stored vectors came from a different model; start fresh
        index = faiss.read_index(str(index_path))
        if index.d != self.embedder.dim:
            return
        self.index = index
        self.chunks = [Chunk(**c) for c in payload["chunks"]]
