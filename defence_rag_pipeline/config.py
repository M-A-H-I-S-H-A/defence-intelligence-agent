"""
Central configuration for the Data + RAG pipeline
(Your part of the Autonomous Defence Intelligence & Decision Support Agent).

Everything downstream (embeddings, ChromaDB collection name, chunk sizes,
GDELT query window, etc.) is controlled from here so the rest of the
codebase never hardcodes values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CHUNK_DIR = DATA_DIR / "chunks"
EDA_DIR = BASE_DIR / "eda_output"
CHROMA_DIR = BASE_DIR / "chroma_store"

for d in [RAW_DIR, CLEAN_DIR, CHUNK_DIR, EDA_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SIPRI
# ---------------------------------------------------------------------------
# SIPRI publishes its Military Expenditure Database as a downloadable Excel
# workbook (no public API). Point this at the file your teammate/you
# downloaded from https://www.sipri.org/databases/milex
SIPRI_RAW_FILE = RAW_DIR / "sipri_milex.xlsx"
SIPRI_SHEET_NAME = "Constant (2022) US$"  # change to whichever sheet you use
SIPRI_CLEAN_FILE = CLEAN_DIR / "sipri_clean.csv"

# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_GKG_BASE = "http://data.gdeltproject.org/gdeltv2"
GDELT_CLEAN_FILE = CLEAN_DIR / "gdelt_clean.csv"
GDELT_DEFAULT_TIMESPAN = "3months"   # GDELT DOC API timespan param
GDELT_MAX_RECORDS = 250

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE_TOKENS = 220          # approx tokens per chunk
CHUNK_OVERLAP_TOKENS = 40
CHUNK_OUTPUT_FILE = CHUNK_DIR / "rag_chunks.jsonl"

# ---------------------------------------------------------------------------
# Embeddings / Vector store
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, CPU-friendly
CHROMA_COLLECTION_NAME = "defence_intel_evidence"

# ---------------------------------------------------------------------------
# Retrieval / Ranking
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = 25          # candidates pulled from ChromaDB
RANKED_TOP_N = 8              # final evidence set handed to the Analysis Agent

# Weights used by the evidence ranker to combine multiple signals into
# one final score (must sum to 1.0)
RANKING_WEIGHTS = {
    "semantic_similarity": 0.55,
    "recency": 0.20,
    "source_reliability": 0.15,
    "verification_score": 0.10,
}

# Trusted source tiers used by the verification/quality module.
SOURCE_RELIABILITY = {
    "sipri": 1.0,          # curated, peer-reviewed dataset
    "gdelt_gkg": 0.75,      # aggregated from wire/news sources, mixed quality
}
