# Integration status — Data+RAG (Mahisha) + Agents (teammate)

This is the merged project: teammate's full agent system (retrieval → analysis
→ verification → risk → report → memory, LangGraph-orchestrated) now wired to
Mahisha's real Data + RAG pipeline instead of the mock evidence.

## What was changed to integrate

1. **`defence_rag_pipeline/`** — Mahisha's whole Data+RAG project, dropped in
   as a subfolder, unchanged.
2. **`services/retrieval_service.py`** — the `RealRetriever` class (previously
   an empty stub that raised `NotImplementedError`) now calls
   `defence_rag_pipeline/pipeline.py`'s `answer_query()` and maps its output
   onto the existing `RetrievedDocument` schema. The default retriever was
   switched from `MockRetriever()` to `RealRetriever()`.
3. **`backend/main.py`** — was empty; now a FastAPI app exposing:
   - `GET /health` — liveness check
   - `POST /assess` — runs the full LangGraph workflow for a question,
     requires an `X-API-Key` header (basic auth, per the original brief)
4. **`frontend/index.html`** — was an empty `app.py` (originally intended as
   Streamlit); replaced with a single-file HTML/JS dashboard. No framework,
   no build step — open it directly in a browser.
5. **`requirements.txt`** — merged in the pipeline's dependencies (pandas,
   chromadb, sentence-transformers, etc.) alongside the existing ones.

**Nothing in `agents/`, `graph/`, or `schemas/` was touched** — that's fully
the teammate's own work, unmodified.

## ⚠️ Required before this runs for real

This package's `defence_rag_pipeline/chroma_store/` is **empty**. The real,
built database (8,653 real SIPRI + GDELT records) only exists on Mahisha's
own machine — it wasn't available to generate this integration.

**Before running:** copy Mahisha's real `chroma_store/` folder (from her
already-built `defence_rag_pipeline` project) into this project's
`defence_rag_pipeline/chroma_store/`, replacing the empty one.

## ⚠️ Security — do this first, unrelated to the integration

The original `.env` file had a live Groq API key exposed in plain text. It
was **excluded from this package**. Whoever owns that key should regenerate
it on Groq's dashboard (treat the old one as compromised) and add `.env` to
`.gitignore` going forward.

## How to run it

```bash
pip install -r requirements.txt

# 1. Make sure defence_rag_pipeline/chroma_store/ has the real data (see above)

# 2. Add your Groq key and (optionally) a stable API key to .env:
#    GROQ_API_KEY=your_new_key_here
#    APP_API_KEY=some-shared-secret-for-the-frontend

# 3. Start the backend (from the project root)
uvicorn backend.main:app --reload --port 8000
#    -> if APP_API_KEY isn't set, the console prints a generated one to use

# 4. Open frontend/index.html directly in a browser
#    -> paste the API key from step 3 into the "API key" field
#    -> ask a question, e.g. "Assess the emerging security situation involving India"
```

## What's left to actually verify (not done in this integration pass)

- End-to-end run with the **real** embedding model + real ChromaDB data,
  on a machine with internet access to huggingface.co (this environment
  couldn't reach it — wiring was verified with a stand-in embedder instead).
- The LLM agents (Analysis, Verification, Risk, Report) haven't been
  re-tested against *real* evidence text — they were presumably built and
  tested against the mock documents. Real SIPRI/GDELT text has a different
  shape (shorter, more numeric for SIPRI) — worth a sanity check that the
  agents' prompts still produce sensible output.
- Frontend/backend CORS and auth are intentionally minimal — fine for a
  local demo, not meant as production security.
