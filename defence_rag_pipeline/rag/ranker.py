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

    # Rank within each source separately, then combine, so one source
    # (e.g. SIPRI's high-reliability numbers) can't fully crowd out
    # another (e.g. GDELT's current events) just by scoring higher overall.
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for cand in candidates:
        source = cand["metadata"].get("source", "unknown")
        by_source.setdefault(source, []).append(cand)

    for source in by_source:
        by_source[source].sort(key=lambda c: c["final_score"], reverse=True)

    half = max(top_n // 2, 1)
    top: List[Dict[str, Any]] = []
    sources = list(by_source.keys())
    per_source_quota = {s: half for s in sources} if len(sources) > 1 else {sources[0]: top_n}

    for source, quota in per_source_quota.items():
        top.extend(by_source[source][:quota])

    if len(top) < top_n:
        remaining = sorted(
            [c for c in candidates if c not in top],
            key=lambda c: c["final_score"], reverse=True,
        )
        top.extend(remaining[: top_n - len(top)])

    top = sorted(top, key=lambda c: c["final_score"], reverse=True)[:top_n]
    logger.info("Ranked %d candidates, returning top %d (source-balanced)", len(candidates), len(top))
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