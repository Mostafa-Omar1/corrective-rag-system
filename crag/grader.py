"""Retrieval evaluator: scores each retrieved chunk for relevance to the question."""
from __future__ import annotations

from dataclasses import dataclass

from .llm import ClaudeLLM
from .text_utils import keyword_overlap
from .vector_store import SearchResult

RELEVANT, AMBIGUOUS, IRRELEVANT = "relevant", "ambiguous", "irrelevant"


@dataclass
class GradedChunk:
    result: SearchResult
    score: float     # relevance in [0, 1]
    verdict: str     # relevant | ambiguous | irrelevant
    reason: str

    @property
    def chunk(self):
        return self.result.chunk


GRADER_SYSTEM = """You are the retrieval evaluator of a Corrective RAG system.
For each numbered passage decide how useful it is for answering the user's question.
Score 0.0-1.0:
- 1.0  the passage directly contains the information needed to answer (fully or partly)
- 0.5  the passage is on the right topic but does not contain the specific answer
- 0.0  the passage is unrelated
Judge only the passage text, not your own knowledge. Give a short reason (max 20 words)."""

GRADER_SCHEMA = {
    "type": "object",
    "properties": {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["grades"],
    "additionalProperties": False,
}


def _verdict(score: float, upper: float, lower: float) -> str:
    if score >= upper:
        return RELEVANT
    if score < lower:
        return IRRELEVANT
    return AMBIGUOUS


def heuristic_score(question: str, result: SearchResult) -> float:
    """Blend lexical coverage with embedding similarity (used without an LLM)."""
    overlap = keyword_overlap(question, result.chunk.text)
    similarity = max(0.0, result.score)
    return round(0.6 * overlap + 0.4 * similarity, 3)


def grade_chunks(
    question: str,
    results: list[SearchResult],
    llm: ClaudeLLM | None,
    upper: float = 0.6,
    lower: float = 0.3,
) -> list[GradedChunk]:
    if not results:
        return []

    if llm is None:
        graded = []
        for r in results:
            s = heuristic_score(question, r)
            graded.append(
                GradedChunk(r, s, _verdict(s, upper, lower),
                            f"keyword overlap {keyword_overlap(question, r.chunk.text):.2f}, "
                            f"cosine {r.score:.2f}")
            )
        return graded

    passages = "\n\n".join(
        f"<passage index=\"{i}\">\n{r.chunk.text}\n</passage>" for i, r in enumerate(results)
    )
    data = llm.json(
        GRADER_SYSTEM,
        f"<question>{question}</question>\n\n{passages}\n\nGrade every passage.",
        GRADER_SCHEMA,
    )
    by_index = {g["index"]: g for g in data.get("grades", [])}
    graded = []
    for i, r in enumerate(results):
        g = by_index.get(i, {"score": 0.0, "reason": "not graded"})
        s = min(1.0, max(0.0, float(g["score"])))
        graded.append(GradedChunk(r, s, _verdict(s, upper, lower), g["reason"]))
    return graded
