"""
GDELT ingestion.

Two ways in:
1. GDELT 2.0 DOC API (news article search) - simplest, JSON over HTTP,
   no key required. Good for "what's happening right now" queries tied
   to a country/actor/theme.
2. GDELT GKG (Global Knowledge Graph) daily/15-min export files - richer
   (actors, themes, locations, Goldstein score, tone, sources) but you
   download and parse zipped CSVs.

We implement the DOC API (fast to demo, matches the "current geopolitical
events" role GDELT plays in the project) and leave a clearly-marked hook
for GKG export ingestion if the team wants full event-level fields later.
"""

from __future__ import annotations
import logging
import time
from typing import List, Dict, Any
import requests
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_gdelt_doc_api(
    query: str,
    api_url: str,
    timespan: str = "3months",
    max_records: int = 250,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """Query the GDELT 2.0 DOC API and return a list of article records."""
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(max_records, 250),  # API hard cap per call
        "timespan": timespan,
        "sort": "hybridrel",
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(api_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("articles", [])
        except (requests.RequestException, ValueError) as e:
            logger.warning("GDELT fetch attempt %d/%d failed: %s", attempt, max_retries, e)
            time.sleep(2 * attempt)
    logger.error("GDELT DOC API failed after %d attempts for query=%r", max_retries, query)
    return []


def articles_to_dataframe(articles: List[Dict[str, Any]], query: str) -> pd.DataFrame:
    """Normalize raw GDELT article JSON into the tidy schema used downstream:
    title, url, domain, seendate, country_query, language, source, tone (if present)
    """
    if not articles:
        return pd.DataFrame(
            columns=["title", "url", "domain", "seendate", "country_query", "language", "source"]
        )

    df = pd.DataFrame(articles)
    rename_map = {
        "seendate": "seendate",
        "sourcecountry": "source_country",
        "socialimage": "social_image",
        "domain": "domain",
        "language": "language",
        "title": "title",
        "url": "url",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["country_query"] = query
    df["source"] = "gdelt_gkg"

    keep_cols = [c for c in [
        "title", "url", "domain", "seendate", "language",
        "source_country", "country_query", "source"
    ] if c in df.columns]
    return df[keep_cols]


def ingest_gdelt(
    queries: List[str],
    api_url: str,
    timespan: str,
    max_records: int,
) -> pd.DataFrame:
    """Run the DOC API for each query (e.g. one per country/actor of
    interest) and concatenate into a single tidy dataframe."""
    frames = []
    for q in queries:
        logger.info("Fetching GDELT articles for query=%r", q)
        articles = fetch_gdelt_doc_api(q, api_url, timespan, max_records)
        frames.append(articles_to_dataframe(articles, q))
        time.sleep(1)  # be polite to the free API

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url"])


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config

    df = ingest_gdelt(
        queries=["India defence", "China military"],
        api_url=config.GDELT_DOC_API,
        timespan=config.GDELT_DEFAULT_TIMESPAN,
        max_records=config.GDELT_MAX_RECORDS,
    )
    print(df.head())
    print(f"Total rows: {len(df)}")
