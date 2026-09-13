"""
SIPRI Military Expenditure ingestion.

SIPRI does not expose a REST API - the standard way analysts get this data
is the downloadable Excel workbook from
https://www.sipri.org/databases/milex ("Military expenditure by country").

This module loads that raw workbook (wide format: one row per country,
one column per year) and reshapes it into a tidy long format:

    country, year, expenditure_usd, currency_basis, source

which is what the rest of the pipeline (cleaning -> chunking -> embeddings)
expects from every data source.
"""

from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def load_sipri_raw(file_path: Path, sheet_name: str) -> pd.DataFrame:
    """Load the raw SIPRI workbook. SIPRI sheets have a few metadata rows
    above the real header, so we detect the header row automatically by
    looking for the row that contains 'Country'."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"SIPRI file not found at {file_path}. Download the milex "
            f"Excel file from https://www.sipri.org/databases/milex and "
            f"place it there (see config.SIPRI_RAW_FILE)."
        )

    preview = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=15)
    header_row = None
    for i, row in preview.iterrows():
        if row.astype(str).str.contains("Country", case=False, na=False).any():
            header_row = i
            break
    if header_row is None:
        header_row = 0
        logger.warning("Could not auto-detect SIPRI header row; defaulting to row 0.")

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
    return df


def reshape_sipri_to_long(df: pd.DataFrame, currency_basis: str) -> pd.DataFrame:
    """Wide (country x year) -> long (country, year, expenditure_usd)."""
    country_col = next(c for c in df.columns if str(c).strip().lower() == "country")
    year_cols = [c for c in df.columns if str(c).strip().isdigit()]

    long_df = df.melt(
        id_vars=[country_col],
        value_vars=year_cols,
        var_name="year",
        value_name="expenditure_usd",
    )
    long_df = long_df.rename(columns={country_col: "country"})
    long_df["year"] = long_df["year"].astype(int)

    # SIPRI uses ".." for missing/undisclosed spending and "xxx" for N/A
    long_df["expenditure_usd"] = pd.to_numeric(
        long_df["expenditure_usd"], errors="coerce"
    )
    long_df["currency_basis"] = currency_basis
    long_df["source"] = "sipri"
    return long_df


def ingest_sipri(file_path: Path, sheet_name: str) -> pd.DataFrame:
    raw = load_sipri_raw(file_path, sheet_name)
    long_df = reshape_sipri_to_long(raw, currency_basis=sheet_name)
    logger.info("Ingested %d SIPRI (country, year) records", len(long_df))
    return long_df


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config

    df = ingest_sipri(config.SIPRI_RAW_FILE, config.SIPRI_SHEET_NAME)
    print(df.head())
