"""
Retrieval Agent (your side of it): takes a natural-language query
("Assess the emerging security situation involving India"), embeds it,
pulls candidate evidence from ChromaDB, and returns a normalized list of
candidates ready for verification + ranking.

This is the function your teammate's Supervisor Agent / LangGraph node
calls: `retrieve_evidence(query, ...)`.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any
from .embedder import embed_texts
from .vector_store import query_collection

logger = logging.getLogger(__name__)


def retrieve_evidence(
    query: str,
    collection,
    embedding_model_name: str,
    top_k: int,
    where: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    query_embedding = embed_texts([query], embedding_model_name)[0]
    results = query_collection(collection, query_embedding, top_k, where)

    candidates = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0] if "ids" in results else [None] * len(docs)

    for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
        candidates.append({
            "id": doc_id,
            "text": text,
            "metadata": meta,
            # Chroma with cosine space returns distance in [0, 2]; convert
            # to a similarity score in [0, 1] for easier downstream use.
            "semantic_similarity": max(0.0, 1 - dist / 2),
        })

    logger.info("Retrieved %d candidates for query=%r", len(candidates), query)
    return candidates
