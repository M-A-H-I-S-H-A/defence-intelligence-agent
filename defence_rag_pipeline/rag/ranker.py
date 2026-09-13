"""
Evidence ranking.

Combines semantic similarity (from the retriever), recency, source
reliability, and the verification score (from verification/quality_checks.py)
into one weighted final score, and returns the top-N ranked evidence set —
this is exactly the "Ranked Evidence" hand-off point to your teammate's
Analysis Agent in the architecture diagram.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def rank_evidence(
    candidates: List[Dict[str, Any]],
    weights: Dict[str, float],
    top_n: int,
) -> List[Dict[str, Any]]:
    for cand in candidates:
        recency = cand.get("freshness", 0.5)
        score = (
            weights["semantic_similarity"] * cand.get("semantic_similarity", 0.0)
            + weights["recency"] * recency
            + weights["source_reliability"] * cand.get("source_reliability", 0.5)
            + weights["verification_score"] * cand.get("verification_score", 0.5)
        )
        cand["final_score"] = round(score, 4)

    ranked = sorted(candidates, key=lambda c: c["final_score"], reverse=True)
    top = ranked[:top_n]
    logger.info("Ranked %d candidates, returning top %d", len(candidates), len(top))
    return top


def to_evidence_payload(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shape the ranked evidence into the clean payload the Analysis /
    Risk Assessment / Report Generation agents consume — hides internal
    scoring fields they don't need, keeps what they do."""
    payload = []
    for r in ranked:
        payload.append({
            "text": r["text"],
            "source": r["metadata"].get("source"),
            "url": r["metadata"].get("url"),
            "country": r["metadata"].get("country") or r["metadata"].get("country_query"),
            "year": r["metadata"].get("year"),
            "seendate": r["metadata"].get("seendate"),
            "confidence": r["final_score"],
        })
    return payload
