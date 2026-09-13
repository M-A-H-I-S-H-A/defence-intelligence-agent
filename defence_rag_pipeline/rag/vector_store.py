"""
ChromaDB wrapper: persistent local vector store for the evidence chunks.

This is deliberately a thin wrapper so your teammate's LangGraph
Retrieval/Supervisor agent can import `get_or_create_collection` and
`query_collection` directly without knowing anything about how the
embeddings were built.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Dict, Any
import chromadb

logger = logging.getLogger(__name__)


def get_client(persist_dir: Path):
    return chromadb.PersistentClient(path=str(persist_dir))


def get_or_create_collection(client, name: str):
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_documents(
    collection,
    docs: List[Dict[str, Any]],
    embeddings,
    batch_size: int = 200,
) -> None:
    """docs: list of {"id", "text", "metadata"} as produced by rag/chunker.py"""
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metadatas = [_flatten_metadata(d["metadata"]) for d in docs]

    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=texts[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            metadatas=metadatas[i:i + batch_size],
        )
    logger.info("Upserted %d documents into collection '%s'", len(ids), collection.name)


def _flatten_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool (no None, no nested
    dicts). Sanitize before storing."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def query_collection(
    collection,
    query_embedding,
    top_k: int,
    where: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
