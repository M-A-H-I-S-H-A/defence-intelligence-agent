"""
End-to-end orchestrator for the Data + RAG side of the project:

    SIPRI + GDELT -> clean -> EDA -> chunks -> embeddings -> ChromaDB
        -> retrieval -> verification -> ranking -> evidence payload

Run modes:
    python pipeline.py --build        # ingest, clean, EDA, chunk, embed, load Chroma
    python pipeline.py --query "..."  # retrieve + verify + rank for a query
    python pipeline.py --demo         # run --build with bundled sample data, then a demo query

Your teammate's LangGraph Supervisor/Retrieval node should ultimately just
import `answer_query()` from this module (or call the underlying
rag.retriever / verification / rag.ranker functions directly) instead of
shelling out to this CLI.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

import config
from ingestion import sipri_ingest, gdelt_ingest
from preprocessing import clean
from eda import eda as eda_mod
from rag import chunker, embedder, vector_store, retriever, ranker
from verification import quality_checks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


def build_from_dataframes(sipri_raw: pd.DataFrame, gdelt_raw: pd.DataFrame) -> None:
    """Shared build path used by both --build (real SIPRI/GDELT sources)
    and --demo (bundled sample CSVs)."""

    # 1-3: clean
    sipri_clean = clean.clean_sipri(sipri_raw) if not sipri_raw.empty else sipri_raw
    gdelt_clean = clean.clean_gdelt(gdelt_raw) if not gdelt_raw.empty else gdelt_raw
    if not sipri_clean.empty:
        clean.save_clean(sipri_clean, config.SIPRI_CLEAN_FILE)
    if not gdelt_clean.empty:
        clean.save_clean(gdelt_clean, config.GDELT_CLEAN_FILE)

    # 4. EDA
    eda_mod.run_eda(sipri_clean, gdelt_clean, config.EDA_DIR)

    # 5-6: chunk + embed
    docs = chunker.build_chunks(
        sipri_clean, gdelt_clean,
        config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS,
    )
    chunker.save_chunks(docs, config.CHUNK_OUTPUT_FILE)

    if not docs:
        logger.warning("No documents to embed — check your input data.")
        return

    texts = [d["text"] for d in docs]
    embeddings = embedder.embed_texts(texts, config.EMBEDDING_MODEL_NAME)

    # 7. ChromaDB
    client = vector_store.get_client(config.CHROMA_DIR)
    collection = vector_store.get_or_create_collection(client, config.CHROMA_COLLECTION_NAME)
    vector_store.upsert_documents(collection, docs, embeddings)

    logger.info("Build complete: %d chunks embedded and stored in ChromaDB at %s",
                len(docs), config.CHROMA_DIR)


def run_build() -> None:
    sipri_raw = sipri_ingest.ingest_sipri(config.SIPRI_RAW_FILE, config.SIPRI_SHEET_NAME)
    gdelt_raw = gdelt_ingest.ingest_gdelt(
        queries=["India defence", "China military", "Pakistan military"],
        api_url=config.GDELT_DOC_API,
        timespan=config.GDELT_DEFAULT_TIMESPAN,
        max_records=config.GDELT_MAX_RECORDS,
    )
    build_from_dataframes(sipri_raw, gdelt_raw)


def run_demo_build() -> None:
    sample_dir = Path(__file__).parent / "sample_data"
    sipri_raw = pd.read_csv(sample_dir / "sipri_sample_long.csv")
    gdelt_raw = pd.read_csv(sample_dir / "gdelt_sample.csv")
    build_from_dataframes(sipri_raw, gdelt_raw)


def answer_query(query: str, where: dict | None = None) -> list[dict]:
    """The single function your teammate's agent graph calls:
    query in -> ranked, verified evidence payload out."""
    client = vector_store.get_client(config.CHROMA_DIR)
    collection = vector_store.get_or_create_collection(client, config.CHROMA_COLLECTION_NAME)

    candidates = retriever.retrieve_evidence(
        query, collection, config.EMBEDDING_MODEL_NAME, config.RETRIEVAL_TOP_K, where,
    )
    if not candidates:
        logger.warning("No candidates retrieved for query=%r", query)
        return []

    verified = quality_checks.verify_candidates(candidates, config.SOURCE_RELIABILITY)
    ranked = ranker.rank_evidence(verified, config.RANKING_WEIGHTS, config.RANKED_TOP_N)
    return ranker.to_evidence_payload(ranked)


def main():
    parser = argparse.ArgumentParser(description="Defence Intel Data+RAG pipeline")
    parser.add_argument("--build", action="store_true", help="Ingest real SIPRI/GDELT and load ChromaDB")
    parser.add_argument("--demo", action="store_true", help="Build using bundled sample data")
    parser.add_argument("--query", type=str, help="Run retrieval+verification+ranking for a query")
    args = parser.parse_args()

    if args.build:
        run_build()
    if args.demo:
        run_demo_build()
    if args.query:
        results = answer_query(args.query)
        print(f"\nTop {len(results)} ranked evidence for: {args.query!r}\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['source']}] (confidence={r['confidence']}) {r['text']}")
    if not any([args.build, args.demo, args.query]):
        parser.print_help()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
