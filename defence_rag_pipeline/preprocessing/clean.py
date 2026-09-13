"""
Data cleaning & preprocessing for SIPRI + GDELT.

Both sources land here in their tidy "long" form from ingestion/, and
leave as fully cleaned, deduplicated, type-correct DataFrames ready for
EDA and for turning into RAG documents.
"""

from __future__ import annotations
import logging
import re
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SIPRI cleaning
# ---------------------------------------------------------------------------
def clean_sipri(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalise country names (SIPRI has some inconsistent naming across years)
    df["country"] = (
        df["country"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Drop rows with no usable expenditure value at all
    df = df.dropna(subset=["expenditure_usd"])

    # Remove duplicate (country, year) pairs, keep the latest-loaded value
    df = df.drop_duplicates(subset=["country", "year"], keep="last")

    # Sanity bounds: expenditure can't be negative
    df = df[df["expenditure_usd"] >= 0]

    # Sort for readability / stable downstream chunking order
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    logger.info("SIPRI clean: %d rows remain", len(df))
    return df


# ---------------------------------------------------------------------------
# GDELT cleaning
# ---------------------------------------------------------------------------
def _clean_title(title: str) -> str:
    if not isinstance(title, str):
        return ""
    title = re.sub(r"\s+", " ", title).strip()
    return title


def clean_gdelt(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["title"] = df["title"].apply(_clean_title)

    # Drop junk / empty articles
    df = df[df["title"].str.len() > 8]

    # Parse GDELT's seendate (format: YYYYMMDDTHHMMSSZ) into a real datetime
    if "seendate" in df.columns:
        df["seendate"] = pd.to_datetime(
            df["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True
        )
        df = df.dropna(subset=["seendate"])

    # Deduplicate by URL (GDELT often returns the same article for
    # multiple overlapping queries) and by near-identical titles
    df = df.drop_duplicates(subset=["url"])
    df = df.drop_duplicates(subset=["title"])

    df = df.sort_values("seendate", ascending=False).reset_index(drop=True)

    logger.info("GDELT clean: %d rows remain", len(df))
    return df


def save_clean(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data to %s (%d rows)", path, len(df))
