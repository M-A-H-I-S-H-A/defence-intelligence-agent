"""
Basic verification / data-quality logic.

This is deliberately lightweight (rule-based, explainable) rather than
another LLM call — it's meant to catch obvious junk/contradictions
*before* evidence reaches the ranking + analysis stage, and to produce a
transparent 0-1 verification_score that the ranker can use as one signal.

Checks implemented:
  1. Freshness — is a GDELT article suspiciously old for a "current
     situation" query?
  2. Cross-source corroboration — does more than one distinct domain/
     source report something similar (approximated via keyword overlap)?
  3. Internal consistency — do SIPRI figures for the same country/year
     across candidate chunks agree?
  4. Source reliability — pulled from config.SOURCE_RELIABILITY.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)


def _freshness_score(metadata: Dict[str, Any], now: datetime | None = None) -> float:
    seendate = metadata.get("seendate")
    if not seendate:
        return 0.5  # SIPRI facts (yearly) don't decay the same way; neutral score
    try:
        dt = datetime.fromisoformat(seendate)
    except ValueError:
        return 0.5
    now = now or datetime.now(timezone.utc)
    age_days = max((now - dt).days, 0)
    # Full score if <7 days old, decays to 0.2 floor by 180 days
    if age_days <= 7:
        return 1.0
    if age_days >= 180:
        return 0.2
    return round(1.0 - 0.8 * (age_days - 7) / (180 - 7), 3)


def _corroboration_score(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
    """Rough corroboration: count distinct domains discussing overlapping
    keywords (top 5 non-trivial words of each chunk's text)."""
    def keywords(text: str) -> set[str]:
        words = [w.strip(".,()[]").lower() for w in text.split()]
        stop = {"the", "a", "an", "in", "on", "of", "to", "and", "for", "with", "is", "at"}
        return {w for w in words if len(w) > 3 and w not in stop}

    scores = {}
    for i, cand in enumerate(candidates):
        kw_i = keywords(cand["text"])
        domain_i = cand["metadata"].get("domain")
        corroborating_domains = set()
        for j, other in enumerate(candidates):
            if i == j:
                continue
            other_domain = other["metadata"].get("domain")
            if other_domain and other_domain == domain_i:
                continue  # same outlet doesn't count as independent corroboration
            overlap = kw_i & keywords(other["text"])
            if len(overlap) >= 2:
                corroborating_domains.add(other_domain or other["metadata"].get("source"))
        # 0 corroborators -> 0.4, 1 -> 0.7, 2+ -> 1.0
        n = len(corroborating_domains)
        scores[cand["id"]] = 0.4 if n == 0 else (0.7 if n == 1 else 1.0)
    return scores


def _sipri_consistency_score(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
    """Flag SIPRI chunks whose (country, year) expenditure disagrees with
    another chunk claiming the same (country, year) (shouldn't normally
    happen post-dedup, but this guards against stale/duplicate ingests)."""
    by_key: Dict[tuple, list] = {}
    for cand in candidates:
        meta = cand["metadata"]
        if meta.get("source") != "sipri":
            continue
        key = (meta.get("country"), meta.get("year"))
        by_key.setdefault(key, []).append(cand)

    scores = {}
    for key, group in by_key.items():
        if len(group) == 1:
            scores[group[0]["id"]] = 1.0
            continue
        values = [g["metadata"].get("expenditure_usd") for g in group]
        spread = (max(values) - min(values)) / max(values, key=abs) if max(values) else 0
        score = 1.0 if abs(spread) < 0.02 else 0.5
        for g in group:
            scores[g["id"]] = score
    return scores


def verify_candidates(candidates: List[Dict[str, Any]], source_reliability: Dict[str, float]) -> List[Dict[str, Any]]:
    """Attach verification_score and source_reliability to every candidate
    in place (returns the same list, mutated, for convenience)."""
    corroboration = _corroboration_score(candidates)
    sipri_consistency = _sipri_consistency_score(candidates)

    for cand in candidates:
        meta = cand["metadata"]
        source = meta.get("source", "unknown")
        reliability = source_reliability.get(source, 0.5)

        freshness = _freshness_score(meta) if source == "gdelt_gkg" else 1.0
        corrob = corroboration.get(cand["id"], 0.5) if source == "gdelt_gkg" else 1.0
        consistency = sipri_consistency.get(cand["id"], 1.0) if source == "sipri" else 1.0

        verification_score = round((freshness + corrob + consistency) / 3, 3)

        cand["source_reliability"] = reliability
        cand["freshness"] = freshness
        cand["verification_score"] = verification_score

    logger.info("Verified %d candidates", len(candidates))
    return candidates
