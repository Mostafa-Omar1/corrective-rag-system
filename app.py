"""Streamlit UI for the Corrective RAG system.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from crag import CorrectiveRAG, VectorStore, get_embedder, get_llm, get_settings, process_document
from crag.ingestion import SUPPORTED_EXTENSIONS
from crag.pipeline import AMBIGUOUS_CTX, CORRECT

SAMPLE_DIR = Path(__file__).parent / "data" / "sample_docs"
STATUS_BADGE = {
    CORRECT: ("✅ Context verified", "green"),
    AMBIGUOUS_CTX: ("⚠️ Ambiguous context", "orange"),
}

st.set_page_config(page_title="Corrective RAG", page_icon="🛠️", layout="wide")
settings = get_settings()


# ----------------------------------------------------------------------------
# Resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model…")
def load_store(model_name: str, storage_dir: str) -> VectorStore:
    return VectorStore(get_embedder(model_name), Path(storage_dir))


def secret_api_key() -> str:
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:  # no secrets.toml present
        return ""


store = load_store(settings.embedding_model, str(settings.storage_dir))
st.session_state.setdefault("history", [])

# ----------------------------------------------------------------------------
# Sidebar: configuration + ingestion
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    env_key = os.getenv("ANTHROPIC_API_KEY") or secret_api_key()
    typed_key = st.text_input(
        "Anthropic API key",
        type="password",
        placeholder="Loaded from environment" if env_key else "sk-ant-…",
        help="Optional. Without a key the app runs in offline mode with heuristic grading "
        "and extractive answers.",
    )
    api_key = typed_key or env_key
    llm = get_llm(settings.model, settings.effort, api_key=api_key or None)
    st.caption(f"Mode: **{'Claude · ' + settings.model if llm else 'Offline (heuristic)'}**")

    with st.expander("Retrieval settings"):
        settings.top_k = st.slider("Top-k chunks", 2, 10, settings.top_k)
        settings.upper_threshold = st.slider("Relevant threshold", 0.3, 1.0, settings.upper_threshold, 0.05)
        settings.lower_threshold = st.slider(
            "Irrelevant threshold", 0.0, settings.upper_threshold, min(settings.lower_threshold, settings.upper_threshold), 0.05
        )
        settings.max_retries = st.slider("Max query rewrites", 0, 4, settings.max_retries)

    st.divider()
    st.header("📄 Documents")
    uploads = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD",
        type=[e.lstrip(".") for e in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )
    col1, col2 = st.columns(2)
    if col1.button("Ingest uploads", disabled=not uploads, use_container_width=True):
        added, errors = 0, []
        with st.spinner("Extracting, cleaning, splitting and embedding…"):
            for f in uploads:
                try:
                    chunks = process_document(f.name, f.getvalue(), settings.chunk_size, settings.chunk_overlap)
                    if not chunks:
                        errors.append(f"{f.name}: no extractable text (scanned PDF?)")
                    added += store.add(chunks)
                except Exception as exc:
                    errors.append(f"{f.name}: {exc}")
        if added:
            st.success(f"Indexed {added} new chunks.")
        else:
            st.info("Nothing new to index.")
        for e in errors:
            st.error(e)

    if col2.button("Load samples", use_container_width=True):
        added = 0
        with st.spinner("Indexing sample documents…"):
            for path in sorted(SAMPLE_DIR.glob("*")):
                added += store.add(process_document(path.name, path.read_bytes(),
                                                    settings.chunk_size, settings.chunk_overlap))
        if added:
            st.success(f"Indexed {added} sample chunks.")
        else:
            st.info("Samples already indexed.")

    if store.sources:
        st.caption(f"**{len(store)} chunks** in the vector database")
        st.dataframe(
            pd.DataFrame(store.sources.items(), columns=["Document", "Chunks"]),
            hide_index=True, use_container_width=True,
        )
        if st.button("🗑️ Clear index", use_container_width=True):
            store.clear()
            st.session_state.history = []
            st.rerun()


# ----------------------------------------------------------------------------
# Main: chat
# ----------------------------------------------------------------------------
st.title("🛠️ Corrective RAG System")
st.caption(
    "Ask questions about your documents. Every retrieved chunk is graded before it is used; "
    "weak retrieval triggers a query rewrite and a second search, and the answer is generated "
    "only from verified context."
)


def render_result(result) -> None:
    badge = STATUS_BADGE.get(result.status) if result.answer.answered else None
    if badge:
        st.markdown(f":{badge[1]}[{badge[0]}]")
    st.markdown(result.answer.text)
    if result.warning:
        st.warning(result.warning)

    if result.cited_sources:
        with st.expander(f"📚 Sources ({len(result.cited_sources)})", expanded=True):
            for i, g in result.cited_sources:
                st.markdown(f"**[{i}] {g.chunk.citation()}** · relevance `{g.score:.2f}`")
                st.caption(g.chunk.text)

    if result.attempts:
        with st.expander(f"🔍 Correction trace ({len(result.attempts)} retrieval attempt(s))"):
            for n, a in enumerate(result.attempts, 1):
                c = a.counts
                st.markdown(
                    f"**Attempt {n}** — query: `{a.query}`  \n"
                    f"✅ {c['relevant']} relevant · ⚠️ {c['ambiguous']} ambiguous · ❌ {c['irrelevant']} irrelevant  \n"
                    f"➡️ {a.action}"
                )
                st.dataframe(
                    pd.DataFrame(
                        [{"verdict": g.verdict, "score": round(g.score, 2),
                          "cosine": round(g.result.score, 2), "source": g.chunk.citation(),
                          "reason": g.reason, "preview": g.chunk.text[:120] + "…"}
                         for g in a.graded]
                    ),
                    hide_index=True, use_container_width=True,
                )


for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item.question)
    with st.chat_message("assistant"):
        render_result(item)

if not len(store):
    st.info("👈 Upload documents (or click **Load samples**) to get started.")

if question := st.chat_input("Ask a question about your documents", disabled=not len(store)):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving, grading and correcting…"):
            result = CorrectiveRAG(store, settings, llm).ask(question)
        render_result(result)
    st.session_state.history.append(result)
