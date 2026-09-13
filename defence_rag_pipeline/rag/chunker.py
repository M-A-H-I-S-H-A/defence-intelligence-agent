"""
Turn cleaned SIPRI + GDELT rows into RAG-ready text documents/chunks.

Design choice: rather than chunking arbitrary blobs of text, we generate
*semantically complete* natural-language statements per row (one SIPRI
fact = one chunk, one GDELT article = one chunk, splitting long article
titles/summaries further only if they exceed the token budget). This
keeps each chunk self-contained and directly citable in the final report
("According to SIPRI, India's 2023 military expenditure was $83.6B").

Every chunk carries rich metadata so the retriever/ranker downstream can
filter and score without re-parsing text.
"""

from __future__ import annotations
import json
import logging
import uuid
from pathlib import Path
from typing import Iterator, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

APPROX_CHARS_PER_TOKEN = 4  # rough heuristic, good enough for budgeting


def _word_chunks(text: str, chunk_size_tokens: int, overlap_tokens: int) -> Iterator[str]:
    """Simple sliding-window chunker over words, sized by an approximate
    token budget. Used only for GDELT text long enough to need splitting."""
    words = text.split()
    chunk_size_words = max(chunk_size_tokens, 20)
    overlap_words = max(overlap_tokens, 0)

    if len(words) <= chunk_size_words:
        yield text
        return

    start = 0
    while start < len(words):
        end = start + chunk_size_words
        yield " ".join(words[start:end])
        if end >= len(words):
            break
        start = end - overlap_words


def sipri_row_to_document(row: pd.Series) -> Dict[str, Any]:
    text = (
        f"According to SIPRI, {row['country']}'s military expenditure in "
        f"{int(row['year'])} was approximately ${row['expenditure_usd']:,.0f} "
        f"({row.get('currency_basis', 'reported basis')})."
    )
    return {
        "id": f"sipri-{row['country']}-{int(row['year'])}",
        "text": text,
        "metadata": {
            "source": "sipri",
            "country": row["country"],
            "year": int(row["year"]),
            "expenditure_usd": float(row["expenditure_usd"]),
            "doc_type": "military_expenditure",
        },
    }


def gdelt_row_to_documents(row: pd.Series, chunk_size_tokens: int, overlap_tokens: int) -> Iterator[Dict[str, Any]]:
    base_text = row["title"]
    seendate = row.get("seendate")
    seendate_str = seendate.isoformat() if pd.notna(seendate) else None

    parts = list(_word_chunks(base_text, chunk_size_tokens, overlap_tokens))
    for i, part in enumerate(parts):
        yield {
            "id": f"gdelt-{uuid.uuid5(uuid.NAMESPACE_URL, str(row.get('url', row.name)))}-{i}",
            "text": part,
            "metadata": {
                "source": "gdelt_gkg",
                "country_query": row.get("country_query"),
                "domain": row.get("domain"),
                "url": row.get("url"),
                "language": row.get("language"),
                "source_country": row.get("source_country"),
                "seendate": seendate_str,
                "doc_type": "news_event",
            },
        }


def build_chunks(
    sipri_df: pd.DataFrame,
    gdelt_df: pd.DataFrame,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[Dict[str, Any]]:
    docs: list[Dict[str, Any]] = []

    for _, row in sipri_df.iterrows():
        docs.append(sipri_row_to_document(row))

    for _, row in gdelt_df.iterrows():
        docs.extend(gdelt_row_to_documents(row, chunk_size_tokens, overlap_tokens))

    logger.info("Built %d RAG chunks (%d SIPRI + %d GDELT-derived)",
                len(docs), len(sipri_df), len(docs) - len(sipri_df))
    return docs


def save_chunks(docs: list[Dict[str, Any]], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info("Saved %d chunks to %s", len(docs), out_path)


def load_chunks(path: Path) -> list[Dict[str, Any]]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs
