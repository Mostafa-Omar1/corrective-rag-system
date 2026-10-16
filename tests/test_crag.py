"""Offline tests: they use the hashing embedder and no LLM, so no network or key is needed."""
from __future__ import annotations

from pathlib import Path

import pytest

from crag import CorrectiveRAG, Settings, VectorStore, process_document
from crag.embeddings import HashingEmbedder
from crag.generator import NO_ANSWER
from crag.ingestion import clean_text, split_text
from crag.pipeline import CORRECT, INCORRECT

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


# ------------------------------------------------------------------ ingestion
def test_clean_text_joins_hyphens_and_collapses_whitespace():
    raw = "Retrie-\nval   augmented\r\n\n\n\ngeneration\x00"
    assert clean_text(raw) == "Retrieval augmented\n\ngeneration"


def test_split_text_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(200))
    chunks = split_text(text, chunk_size=300, chunk_overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # consecutive chunks share some text because of the overlap
    assert any(c1[-30:].split()[-1] in c2 for c1, c2 in zip(chunks, chunks[1:]))


def test_split_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, chunk_overlap=100)


def test_process_document_rejects_unknown_extension():
    with pytest.raises(ValueError):
        process_document("image.png", b"\x89PNG")


def test_process_document_produces_chunks_with_metadata():
    chunks = process_document("policy.md", (SAMPLES / "company_policy.md").read_bytes(), 400, 80)
    assert chunks and all(c.source == "policy.md" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# --------------------------------------------------------------- vector store
@pytest.fixture()
def store(tmp_path):
    s = VectorStore(HashingEmbedder(), tmp_path)
    for path in SAMPLES.glob("*.md"):
        s.add(process_document(path.name, path.read_bytes(), 500, 100))
    return s


def test_store_deduplicates_and_persists(store, tmp_path):
    n = len(store)
    path = SAMPLES / "rag_overview.md"
    assert store.add(process_document(path.name, path.read_bytes(), 500, 100)) == 0
    reloaded = VectorStore(HashingEmbedder(), tmp_path)
    assert len(reloaded) == n


def test_search_returns_relevant_source(store):
    top = store.search("How many days of annual leave do employees get?", k=3)
    assert top[0].chunk.source == "company_policy.md"


# ------------------------------------------------------------------- pipeline
def make_rag(store, **overrides) -> CorrectiveRAG:
    settings = Settings(embedding_model="hashing", **overrides)
    return CorrectiveRAG(store, settings, llm=None)


def test_answerable_question_is_grounded_with_sources(store):
    result = make_rag(store).ask("How many days of annual leave do full-time employees receive?")
    assert result.status == CORRECT
    assert result.answer.answered
    assert "21" in result.answer.text
    assert result.cited_sources
    assert all(g.chunk.source == "company_policy.md" for _, g in result.cited_sources)


def test_unanswerable_question_is_declined_after_rewrites(store):
    result = make_rag(store, max_retries=2).ask("What is the boiling point of mercury on Venus?")
    assert not result.answer.answered
    assert result.answer.text == NO_ANSWER
    assert result.sources == []
    assert len(result.attempts) == 3  # original + 2 rewrites
    assert len({a.query for a in result.attempts}) == 3  # every rewrite is a new query


def test_weak_query_triggers_rewrite_and_recovers(store):
    # Conversational filler dilutes the first search; the rewrite strips it.
    result = make_rag(store, upper_threshold=0.55).ask(
        "Hey, could you please tell me, if you know, what the IT help desk extension is?"
    )
    assert result.answer.answered
    assert "4040" in result.answer.text


def test_empty_index_and_empty_question(tmp_path):
    empty = VectorStore(HashingEmbedder(), tmp_path)
    assert not make_rag(empty).ask("anything").answer.answered
    assert make_rag(empty).ask("   ").status == INCORRECT


# ----------------------------------------------------------- LLM code path
class FakeLLM:
    """Stands in for ClaudeLLM: grades everything irrelevant on the first query,
    then relevant after the rewrite, and answers with a citation."""

    def __init__(self):
        self.calls: list[str] = []
        self.rewrites = 0

    def json(self, system, prompt, schema, max_tokens=4000):
        props = schema["properties"]
        if "grades" in props:
            self.calls.append("grade")
            n = prompt.count("<passage ")
            score = 0.9 if self.rewrites else 0.1
            return {"grades": [{"index": i, "score": score, "reason": "test"} for i in range(n)]}
        if "query" in props:
            self.calls.append("rewrite")
            self.rewrites += 1
            return {"query": "annual leave days allowance"}
        self.calls.append("generate")
        assert "<context>" in prompt
        return {"answerable": True, "answer": "21 days [1].", "cited_passages": [1, 99]}


def test_llm_path_rewrites_then_generates(store):
    llm = FakeLLM()
    rag = CorrectiveRAG(store, Settings(embedding_model="hashing"), llm=llm)
    result = rag.ask("vacation?")
    assert llm.calls == ["grade", "rewrite", "grade", "generate"]
    assert result.attempts[1].query == "annual leave days allowance"
    assert result.status == CORRECT and result.answer.answered
    assert result.answer.cited == [1]  # out-of-range citation 99 is dropped
