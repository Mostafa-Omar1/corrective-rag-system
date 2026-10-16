"""Central configuration for the Corrective RAG system.

Every value can be overridden with an environment variable so the same code
runs locally, in tests and on Streamlit Community Cloud / Hugging Face Spaces.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # --- LLM -----------------------------------------------------------------
    model: str = field(default_factory=lambda: _env("CRAG_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: _env("CRAG_EFFORT", "low"))

    # --- Embeddings / vector store ------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: _env("CRAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    storage_dir: Path = field(
        default_factory=lambda: Path(_env("CRAG_STORAGE_DIR", str(ROOT_DIR / "storage")))
    )

    # --- Chunking -------------------------------------------------------------
    chunk_size: int = 800        # characters
    chunk_overlap: int = 150     # characters

    # --- Retrieval & correction ---------------------------------------------
    top_k: int = 5
    # A chunk scoring >= upper is "relevant"; < lower is "irrelevant";
    # anything in between is "ambiguous" (the three CRAG confidence buckets).
    upper_threshold: float = 0.6
    lower_threshold: float = 0.3
    max_retries: int = 2         # how many times the query may be rewritten


def get_settings() -> Settings:
    return Settings()
