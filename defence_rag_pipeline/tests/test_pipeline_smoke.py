"""
Lightweight smoke tests (no pytest dependency required — run directly).
Validates the pure-logic parts of the pipeline that don't need network
or a downloaded embedding model: cleaning, chunking, verification, ranking.

Run with:  python tests/test_pipeline_smoke.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from preprocessing import clean
from rag import chunker, ranker
from verification import quality_checks
import config


def test_clean_sipri_drops_missing_and_dupes():
    df = pd.DataFrame({
        "country": ["India", "India", "India", "Nowhere"],
        "year": [2022, 2022, 2023, 2022],
        "expenditure_usd": [81.4e9, 81.4e9, None, -5],
        "currency_basis": ["x"] * 4,
        "source": ["sipri"] * 4,
    })
    out = clean.clean_sipri(df)
    assert len(out) == 1, f"expected 1 clean row, got {len(out)}"
    print("test_clean_sipri_drops_missing_and_dupes: PASS")


def test_chunker_builds_expected_ids():
    sipri = pd.DataFrame({
        "country": ["India"], "year": [2023], "expenditure_usd": [83.6e9],
        "currency_basis": ["x"], "source": ["sipri"],
    })
    gdelt = pd.DataFrame({
        "title": ["India conducts missile test near western border"],
        "url": ["https://x.com/1"], "domain": ["x.com"],
        "seendate": [pd.Timestamp("2026-08-10", tz="UTC")],
        "language": ["English"], "source_country": ["India"],
        "country_query": ["India defence"], "source": ["gdelt_gkg"],
    })
    docs = chunker.build_chunks(sipri, gdelt, config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS)
    assert len(docs) == 2
    assert docs[0]["id"].startswith("sipri-India-2023")
    assert docs[1]["metadata"]["source"] == "gdelt_gkg"
    print("test_chunker_builds_expected_ids: PASS")


def test_ranker_orders_by_weighted_score():
    candidates = [
        {"id": "a", "text": "t", "metadata": {"source": "sipri"},
         "semantic_similarity": 0.9, "freshness": 1.0, "source_reliability": 1.0, "verification_score": 1.0},
        {"id": "b", "text": "t", "metadata": {"source": "gdelt_gkg"},
         "semantic_similarity": 0.95, "freshness": 0.2, "source_reliability": 0.75, "verification_score": 0.4},
    ]
    ranked = ranker.rank_evidence(candidates, config.RANKING_WEIGHTS, top_n=2)
    assert ranked[0]["id"] == "a", "higher-quality-but-slightly-less-similar evidence should win"
    print("test_ranker_orders_by_weighted_score: PASS")


def test_verification_scores_are_bounded():
    candidates = [
        {"id": "a", "text": "India border tension rises", "metadata": {"source": "gdelt_gkg", "seendate": "2026-08-20T00:00:00+00:00", "domain": "x.com"}},
        {"id": "b", "text": "India tension rises at border", "metadata": {"source": "gdelt_gkg", "seendate": "2026-08-20T00:00:00+00:00", "domain": "y.com"}},
    ]
    out = quality_checks.verify_candidates(candidates, config.SOURCE_RELIABILITY)
    for c in out:
        assert 0 <= c["verification_score"] <= 1
    print("test_verification_scores_are_bounded: PASS")


if __name__ == "__main__":
    test_clean_sipri_drops_missing_and_dupes()
    test_chunker_builds_expected_ids()
    test_ranker_orders_by_weighted_score()
    test_verification_scores_are_bounded()
    print("\nAll smoke tests passed.")
