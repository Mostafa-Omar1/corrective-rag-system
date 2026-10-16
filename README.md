# 🛠️ Corrective RAG System

A **Corrective Retrieval-Augmented Generation (CRAG)** system that answers questions from your documents, **evaluates the quality of every retrieved chunk**, and **corrects the retrieval** (query rewrite + re-retrieval) when the context is weak or irrelevant. Answers are generated **only from verified context**, always come **with sources**, and the system **declines to answer** rather than hallucinate.

Built with Python, FAISS, sentence-transformers, Claude (Anthropic API) and Streamlit.

---

## ✨ Features

| Requirement | Implementation |
|---|---|
| Prepare & process documents | PDF, DOCX, TXT, MD upload (`crag/ingestion.py`) |
| Ingestion pipeline: upload → extract → clean → split | Page-aware extraction, Unicode/whitespace/hyphenation cleaning, recursive splitter with overlap |
| Embeddings in a vector database | `all-MiniLM-L6-v2` embeddings stored in a persistent **FAISS** index (`crag/vector_store.py`) |
| Retrieve & evaluate relevant chunks | Retrieval evaluator scores every chunk 0–1 → *relevant / ambiguous / irrelevant* (`crag/grader.py`) |
| Rewrite the query & re-retrieve if context is weak | Query rewriter + configurable retry budget (`crag/rewriter.py`, `crag/pipeline.py`) |
| Generate answers using verified context only | Grounded generator with strict "context-only" rules and structured output (`crag/generator.py`) |
| Show answers with sources | Inline `[n]` citations + source panel with file, page and relevance score |
| Handle unanswered questions | Explicit "can't answer from the documents" response when no verified context exists |
| Deploy using Streamlit | `app.py` – chat UI with a full **correction trace** for every question |

### Two modes
- **Claude mode** (with `ANTHROPIC_API_KEY`): Claude grades chunks, rewrites queries and writes the answer, using JSON-schema structured outputs.
- **Offline mode** (no key): heuristic grader (keyword coverage + cosine similarity), keyword / pseudo-relevance-feedback query rewriting and extractive answers. Useful for demos, tests and CI.

---

## 🧠 How it works

```
             ┌────────────┐
 question ──►│  Retrieve  │◄─────────────────────────┐
             └─────┬──────┘                          │
                   ▼                                 │
             ┌────────────┐   none relevant   ┌──────┴───────┐
             │   Grade    │──────────────────►│ Rewrite query │  (≤ max_retries)
             │  each chunk│                   └──────────────┘
             └─────┬──────┘
        relevant ▼            retries exhausted
             ┌────────────┐   ├─ ambiguous chunks → cautious answer + warning
             │  Generate  │   └─ nothing usable   → "I couldn't find enough information…"
             │ (verified  │
             │  context)  │
             └─────┬──────┘
                   ▼
           answer + [n] citations + sources + correction trace
```

1. **Ingestion** – files are extracted (page numbers kept for PDFs), cleaned and split into ~800-character chunks with 150-character overlap.
2. **Indexing** – chunks are embedded and stored in FAISS (cosine similarity); the index persists in `storage/` and duplicate documents are skipped.
3. **Retrieval** – the top-k chunks are retrieved for the current query.
4. **Evaluation** – each chunk is scored against the *original* question. Score ≥ 0.6 → relevant, < 0.3 → irrelevant, in between → ambiguous (thresholds are adjustable in the UI).
5. **Correction** – if no chunk is relevant, the query is rewritten (keeping the intent, removing filler, adding domain terms) and retrieval runs again.
6. **Generation** – only verified chunks are passed to the generator, which must cite them; it may also declare the question unanswerable.

---

## 🚀 Quick start

```bash
git clone https://github.com/Mostafa-Omar1/corrective-rag-system.git
cd corrective-rag-system
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501, click **Load samples** (or upload your own files) and ask a question.

**Optional – enable Claude:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

or paste the key into the sidebar. Other settings (`CRAG_MODEL`, `CRAG_EFFORT`, `CRAG_EMBEDDING_MODEL`, `CRAG_STORAGE_DIR`) are listed in `.env.example`.

### Try these with the sample documents
| Question | Expected behaviour |
|---|---|
| *How many days of annual leave do employees get?* | Verified on the first try, answer cites `company_policy.md` |
| *What are the three confidence levels in CRAG?* | Verified, list answer from `corrective_rag.md` |
| *pls tell me about the hallucination thing* | Noisy query still answered with sources |
| *Who won the 2022 World Cup?* | All chunks irrelevant → 2 rewrites → politely declines |

---

## ☁️ Deploy on Streamlit Community Cloud
1. Fork / push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo, branch `main`, file `app.py`.
3. (Optional) In **Advanced settings → Secrets** add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Deploy.

---

## 🗂️ Project structure

```
├── app.py                  # Streamlit UI
├── crag/
│   ├── config.py           # settings (env-overridable)
│   ├── ingestion.py        # extract → clean → split
│   ├── embeddings.py       # sentence-transformers (+ hashing fallback)
│   ├── vector_store.py     # persistent FAISS index + metadata
│   ├── llm.py              # Claude wrapper with structured JSON output
│   ├── grader.py           # retrieval evaluator
│   ├── rewriter.py         # query rewriting
│   ├── generator.py        # grounded answer generation
│   ├── text_utils.py       # tokenisation / keyword helpers
│   └── pipeline.py         # Corrective RAG orchestration
├── data/sample_docs/       # demo documents
└── tests/test_crag.py      # offline test suite
```

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests run fully offline (hashing embedder, no LLM, plus a fake LLM to exercise the Claude code path).

## 📚 Reference
Yan, S.-Q., Gu, J.-C., Zhu, Y., & Ling, Z.-H. (2024). *Corrective Retrieval Augmented Generation.* arXiv:2401.15884.
