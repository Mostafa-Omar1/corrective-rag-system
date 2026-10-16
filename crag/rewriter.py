"""Query rewriting used when the retrieved context is weak."""
from __future__ import annotations

from collections import Counter

from .llm import ClaudeLLM
from .text_utils import STOPWORDS, tokenize

REWRITE_SYSTEM = """You rewrite search queries for a document retrieval system (semantic vector search).
The previous query retrieved passages that were judged weak or irrelevant.
Write ONE better standalone search query that keeps the user's intent but:
- expands abbreviations and adds likely synonyms / domain terms
- removes conversational filler
- avoids repeating any previously tried query
Return only the query."""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


def heuristic_rewrite(question: str, tried: list[str], weak_context: list[str] | None = None) -> str:
    """Offline fallback.

    1st rewrite: strip filler and keep only the content keywords.
    Later rewrites: pseudo-relevance feedback - expand the keywords with the most
    frequent terms of the best-scoring (but still weak) passages.
    """
    words = list(dict.fromkeys(w for w in tokenize(question) if w not in STOPWORDS))
    candidates = [" ".join(words)]
    if weak_context:
        counts = Counter(
            w for text in weak_context[:2] for w in tokenize(text)
            if w not in STOPWORDS and len(w) > 3 and w not in words
        )
        expansion = [w for w, _ in counts.most_common(6)]
        candidates += [" ".join(words + expansion[:3]), " ".join(words + expansion)]
    for c in candidates:
        if c and c not in tried:
            return c
    return question


def rewrite_query(
    question: str, tried: list[str], weak_context: list[str], llm: ClaudeLLM | None
) -> str:
    if llm is None:
        return heuristic_rewrite(question, tried, weak_context)

    previous = "\n".join(f"- {q}" for q in tried)
    snippets = "\n".join(f"- {t[:200]}" for t in weak_context[:3]) or "- (nothing retrieved)"
    data = llm.json(
        REWRITE_SYSTEM,
        f"<question>{question}</question>\n"
        f"<tried_queries>\n{previous}\n</tried_queries>\n"
        f"<weak_results>\n{snippets}\n</weak_results>",
        REWRITE_SCHEMA,
        max_tokens=1000,
    )
    return data.get("query", "").strip() or heuristic_rewrite(question, tried, weak_context)
