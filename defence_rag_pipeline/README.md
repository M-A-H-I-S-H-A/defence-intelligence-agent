# Data + RAG Pipeline — Your Part of the Autonomous Defence Intelligence Agent

This is the complete implementation of **your side** of the project
(everything in section 5 of the project brief):

```
SIPRI + GDELT -> Clean Data -> Embeddings -> ChromaDB -> Retriever -> Ranked Evidence
```

It plugs into your teammate's LangGraph agent side via one function:
`pipeline.answer_query(query)` → returns a ranked, verified evidence
payload that her Analysis / Risk Assessment / Report Generation agents
consume (replacing the dummy evidence she's using now).

## What's implemented (maps 1:1 to your task list)

| # | Task | File |
|---|------|------|
| 1 | SIPRI data | `ingestion/sipri_ingest.py` |
| 2 | GDELT data | `ingestion/gdelt_ingest.py` |
| 3 | Data cleaning | `preprocessing/clean.py` |
| 4 | Data preprocessing | `preprocessing/clean.py` |
| 5 | EDA / visualizations | `eda/eda.py` (see `eda_output/`) |
| 6 | RAG-ready documents/chunks | `rag/chunker.py` |
| 7 | Embeddings | `rag/embedder.py` |
| 8 | ChromaDB | `rag/vector_store.py` |
| 9 | Retrieval | `rag/retriever.py` |
| 10 | Evidence ranking | `rag/ranker.py` |
| 11 | Basic verification / data-quality logic | `verification/quality_checks.py` |

`pipeline.py` orchestrates all of the above end-to-end and is the file
you'll actually run / demo from.

## Setup

```bash
pip install -r requirements.txt
```

## Getting the real data

- **SIPRI**: SIPRI has no API. Download the "Military Expenditure by
  Country" workbook from https://www.sipri.org/databases/milex and save
  it as `data/raw/sipri_milex.xlsx` (path is configurable in `config.py`).
- **GDELT**: no download needed — `ingestion/gdelt_ingest.py` pulls live
  articles from the free GDELT 2.0 DOC API for whichever
  countries/actors you list in `pipeline.run_build()`.

## Running it

```bash
# 1. Try it immediately with bundled synthetic sample data (no downloads needed)
python pipeline.py --demo

# 2. Once you have the real SIPRI file downloaded and internet access for GDELT:
python pipeline.py --build

# 3. Ask a question against whatever you've built:
python pipeline.py --query "Assess the emerging security situation involving India"
```

`--demo`/`--build` will:
1. Ingest + clean SIPRI and GDELT
2. Generate EDA plots + a summary into `eda_output/`
3. Turn every row into a self-contained RAG chunk (`data/chunks/rag_chunks.jsonl`)
4. Embed all chunks (sentence-transformers, all-MiniLM-L6-v2, CPU-friendly)
5. Load them into a **persistent** local ChromaDB at `chroma_store/`

`--query` will then:
1. Embed the query
2. Retrieve top-K candidates from ChromaDB
3. Run verification (freshness, cross-source corroboration, SIPRI
   internal consistency, source reliability)
4. Combine everything into one weighted score and return the top-N
   ranked evidence — this is what gets handed to the Analysis Agent.

## Handing off to your teammate

Her Supervisor/Retrieval LangGraph node should call:

```python
from pipeline import answer_query
evidence = answer_query("Assess the emerging security situation involving India")
# evidence -> list[{text, source, url, country, year, seendate, confidence}]
```

That's the exact "Ranked Evidence" object the architecture diagram shows
flowing from your side into her Verification/Analysis/Risk Assessment
agents. No other integration work should be needed on her end beyond
importing this function (or calling it over a small FastAPI wrapper if
you end up splitting into two services/repos).

## Design notes worth mentioning in your report/demo

- **Why row-per-chunk instead of arbitrary text splitting**: each SIPRI
  fact and each GDELT article becomes one self-contained, directly
  citable statement, so the final report can attribute claims cleanly
  ("According to SIPRI, ...") instead of citing a fragment of a larger
  blob.
- **Why 4 ranking signals, not just similarity**: pure vector similarity
  would happily rank a stale but on-topic article above a corroborated,
  recent one. Weighting in recency, source reliability, and a rule-based
  verification score keeps the evidence set trustworthy, which matters
  more here than in a generic chatbot RAG use case.
- **Why the verification step is rule-based, not another LLM call**:
  it's explainable, fast, deterministic, and cheap to run on every
  query — good for a decision-support tool where you want to be able to
  show *why* a score is what it is.
- **Extensibility**: swap `EMBEDDING_MODEL_NAME` in `config.py` for a
  higher-quality model later; add GDELT GKG export parsing in
  `ingestion/gdelt_ingest.py` if you want actor/theme/Goldstein-score
  fields; add more `where=` filters to `answer_query()` (e.g. filter to
  `source=sipri` only) if the Supervisor Agent wants source-scoped
  queries.

## Note on this sandbox's smoke test

`huggingface.co` (needed to download the embedding model) and the live
GDELT API aren't reachable from this build environment, so the full
pipeline was validated end-to-end (cleaning → EDA → chunking → ChromaDB
→ retrieval → verification → ranking) using the bundled sample data and
a stand-in embedder just to prove the wiring. The EDA plots in
`eda_output/` were generated from that same sample data and are real.
Run `python pipeline.py --demo` yourself with internet access and the
real `sentence-transformers` model will download once and be cached.
