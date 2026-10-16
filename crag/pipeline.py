"""Corrective RAG orchestration.

    retrieve ─► grade ─► any relevant? ──yes──► generate from verified chunks
                  ▲            │no
                  │            ▼
                  └──── rewrite query (up to max_retries)
                               │ still nothing
                               ▼
                 ambiguous chunks? ─► cautious answer   none ─► "can't answer"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import Settings
from .generator import NO_ANSWER, Answer, generate_answer
from .grader import AMBIGUOUS, RELEVANT, GradedChunk, grade_chunks
from .llm import ClaudeLLM, LLMError
from .rewriter import rewrite_query
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Final retrieval verdicts (named after the CRAG paper's three actions)
CORRECT, AMBIGUOUS_CTX, INCORRECT = "correct", "ambiguous", "incorrect"


@dataclass
class Attempt:
    query: str
    graded: list[GradedChunk]
    action: str  # what the system decided after grading this attempt

    @property
    def counts(self) -> dict[str, int]:
        out = {"relevant": 0, "ambiguous": 0, "irrelevant": 0}
        for g in self.graded:
            out[g.verdict] += 1
        return out


@dataclass
class CRAGResult:
    question: str
    answer: Answer
    status: str                      # correct | ambiguous | incorrect
    sources: list[GradedChunk]       # the context the answer was generated from
    attempts: list[Attempt] = field(default_factory=list)
    used_llm: bool = False
    warning: str | None = None

    @property
    def cited_sources(self) -> list[tuple[int, GradedChunk]]:
        """Sources the answer actually cites (falls back to all sources)."""
        if self.answer.cited:
            return [(i, self.sources[i - 1]) for i in self.answer.cited]
        return list(enumerate(self.sources, 1)) if self.answer.answered else []


class CorrectiveRAG:
    def __init__(self, store: VectorStore, settings: Settings, llm: ClaudeLLM | None = None):
        self.store = store
        self.settings = settings
        self.llm = llm

    def ask(self, question: str) -> CRAGResult:
        question = question.strip()
        s = self.settings
        if not question:
            return CRAGResult(question, Answer("Please enter a question.", False), INCORRECT, [])
        if len(self.store) == 0:
            return CRAGResult(
                question,
                Answer("No documents are indexed yet. Upload documents first.", False),
                INCORRECT,
                [],
            )

        attempts: list[Attempt] = []
        tried: list[str] = []
        verified: dict[str, GradedChunk] = {}
        ambiguous: dict[str, GradedChunk] = {}
        query = question

        try:
            for attempt_no in range(s.max_retries + 1):
                tried.append(query)
                results = self.store.search(query, s.top_k)
                # Always grade against the ORIGINAL question, not the rewritten query.
                graded = grade_chunks(question, results, self.llm, s.upper_threshold, s.lower_threshold)

                for g in graded:
                    bucket = verified if g.verdict == RELEVANT else ambiguous if g.verdict == AMBIGUOUS else None
                    if bucket is not None:
                        prev = bucket.get(g.chunk.chunk_id)
                        if prev is None or g.score > prev.score:
                            bucket[g.chunk.chunk_id] = g

                if verified:
                    attempts.append(Attempt(query, graded, "context verified → generate"))
                    break
                if attempt_no == s.max_retries:
                    attempts.append(Attempt(query, graded, "retry budget exhausted"))
                    break

                weak = [g.chunk.text for g in sorted(graded, key=lambda g: -g.score)]
                query = rewrite_query(question, tried, weak, self.llm)
                attempts.append(Attempt(tried[-1], graded, f"context weak → rewrite to: “{query}”"))

            if verified:
                status = CORRECT
                context = sorted(verified.values(), key=lambda g: -g.score)
            elif ambiguous:
                status = AMBIGUOUS_CTX
                context = sorted(ambiguous.values(), key=lambda g: -g.score)[: s.top_k]
            else:
                status, context = INCORRECT, []

            answer = generate_answer(question, context, self.llm)
        except LLMError as exc:
            logger.exception("LLM failure")
            return CRAGResult(question, Answer(f"⚠️ {exc}", False), INCORRECT, [], attempts,
                              used_llm=True, warning=str(exc))

        warning = None
        if status == AMBIGUOUS_CTX and answer.answered:
            warning = ("No passage was rated clearly relevant; this answer is based on partially "
                       "relevant context and may be incomplete.")
        if not answer.answered:
            answer = Answer(NO_ANSWER, False)

        return CRAGResult(question, answer, status, context if answer.answered else [],
                          attempts, used_llm=self.llm is not None, warning=warning)
