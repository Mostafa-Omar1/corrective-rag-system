"""Answer generation grounded strictly in verified context."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .grader import GradedChunk
from .llm import ClaudeLLM
from .text_utils import keyword_overlap, split_sentences

NO_ANSWER = (
    "I couldn't find enough reliable information in the uploaded documents to answer "
    "this question. Try rephrasing it, or upload a document that covers this topic."
)


@dataclass
class Answer:
    text: str
    answered: bool
    cited: list[int] = field(default_factory=list)  # 1-based indices into the context


GENERATOR_SYSTEM = """You answer questions using ONLY the numbered context passages provided.
Rules:
- Use no outside knowledge. If the passages do not contain the answer, set "answerable" to false.
- Cite every claim with the passage number in square brackets, e.g. [1] or [2][3].
- Be concise and direct; use short bullet points for lists.
- If the passages only partly answer the question, answer that part and say what is missing."""

GENERATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "cited_passages": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answerable", "answer", "cited_passages"],
    "additionalProperties": False,
}


def format_context(chunks: list[GradedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {g.chunk.citation()})\n{g.chunk.text}" for i, g in enumerate(chunks, 1)
    )


def _is_list_item(line: str) -> bool:
    line = line.lstrip()
    return line.startswith(("-", "*", "•")) or line[:1].isdigit()


def extractive_answer(question: str, chunks: list[GradedChunk], max_sentences: int = 3) -> Answer:
    """Offline fallback: return the most question-relevant sentences, with citations."""
    scored = []  # (score, chunk index, sentence position, text)
    for i, g in enumerate(chunks, 1):
        sentences = split_sentences(g.chunk.text)
        for pos, sent in enumerate(sentences):
            overlap = keyword_overlap(question, sent)
            if overlap == 0:
                continue
            if sent.endswith(":"):  # a lead-in: keep the whole list that follows it
                after = g.chunk.text.split(sent, 1)[-1].lstrip("\n").split("\n")
                items = [ln for ln in itertools.takewhile(_is_list_item, after)]
                sent += "".join("\n" + ln for ln in items) + ("\n" if items else "")
            scored.append((overlap + 0.1 * g.score, i, pos, sent))
    if not scored:
        return Answer(NO_ANSWER, answered=False)
    scored.sort(key=lambda x: x[0], reverse=True)
    floor = scored[0][0] * 0.5  # drop sentences far weaker than the best match
    best = [s for s in scored[:max_sentences] if s[0] >= floor]
    best.sort(key=lambda x: (x[1], x[2]))  # keep document order for readability
    text = " ".join(f"{sent} [{i}]" for _, i, _, sent in best)
    return Answer(text, answered=True, cited=sorted({i for _, i, _, _ in best}))


def generate_answer(question: str, chunks: list[GradedChunk], llm: ClaudeLLM | None) -> Answer:
    if not chunks:
        return Answer(NO_ANSWER, answered=False)
    if llm is None:
        return extractive_answer(question, chunks)

    data = llm.json(
        GENERATOR_SYSTEM,
        f"<context>\n{format_context(chunks)}\n</context>\n\n<question>{question}</question>",
        GENERATOR_SCHEMA,
        max_tokens=8000,
    )
    if not data.get("answerable") or not data.get("answer", "").strip():
        return Answer(NO_ANSWER, answered=False)
    cited = sorted({i for i in data.get("cited_passages", []) if 1 <= i <= len(chunks)})
    return Answer(data["answer"].strip(), answered=True, cited=cited)
