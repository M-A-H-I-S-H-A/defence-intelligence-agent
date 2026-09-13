"""
Retrieval abstraction layer.

RetrievalAgent -> RetrievalService -> Retriever (interface)
                                          |
                                MockRetriever (now)
                                RealRetriever (later, teammate's RAG)

The agent layer only ever talks to RetrievalService. Swapping MockRetriever
for RealRetriever requires no changes above this file.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from schemas.contracts import RetrievedDocument, RetrievalResult

MOCK_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_documents.json"


# ---------------------------------------------------------------------------
# Retriever interface
# ---------------------------------------------------------------------------

class BaseRetriever(ABC):
    """All retrievers (mock or real) must implement this interface."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock retriever
# ---------------------------------------------------------------------------

class MockRetriever(BaseRetriever):
    """
    Simulates semantic retrieval using simple keyword-overlap scoring against
    the static mock knowledge base. Not a substitute for real embeddings —
    just enough signal to exercise ranking, filtering, and downstream agents
    during development.
    """

    def __init__(self, data_path: Path = MOCK_DATA_PATH):
        self.data_path = data_path
        self._documents: list[dict] = self._load_documents()

    def _load_documents(self) -> list[dict]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("documents", [])

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _score(self, query_tokens: set[str], doc: dict) -> float:
        searchable_text = " ".join(
            [
                doc.get("title", ""),
                doc.get("content", ""),
                doc.get("source_type", ""),
                " ".join(str(v) for v in doc.get("metadata", {}).values()),
            ]
        )
        doc_tokens = self._tokenize(searchable_text)

        if not query_tokens or not doc_tokens:
            return 0.0

        overlap = query_tokens & doc_tokens
        # Jaccard-style overlap, weighted toward query coverage rather than
        # penalizing long documents too harshly.
        score = len(overlap) / len(query_tokens)
        return round(min(score, 1.0), 4)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        query_tokens = self._tokenize(query)

        scored_docs = []
        for doc in self._documents:
            score = self._score(query_tokens, doc)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda pair: pair[0], reverse=True)
        top_docs = scored_docs[:top_k]

        results = []
        for score, doc in top_docs:
            results.append(
                RetrievedDocument(
                    document_id=doc["document_id"],
                    title=doc["title"],
                    content=doc["content"],
                    source=doc["source"],
                    source_type=doc["source_type"],
                    publication_date=doc.get("publication_date"),
                    metadata=doc.get("metadata", {}),
                    similarity_score=score,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Real retriever placeholder — teammate wires this up later
# ---------------------------------------------------------------------------

class RealRetriever(BaseRetriever):
    """
    Wraps the Data + RAG teammate's pipeline (defence_rag_pipeline/pipeline.py)
    behind this same BaseRetriever interface. Calls answer_query(), which
    already does retrieval + verification + source-balanced ranking against
    the real SIPRI + GDELT ChromaDB, and maps its output onto our
    RetrievedDocument schema. Nothing above this class (RetrievalAgent,
    RetrievalService) needed any changes to support this.
    """

    def __init__(self, *args, **kwargs):
        import sys
        from pathlib import Path

        pipeline_dir = Path(__file__).resolve().parent.parent / "defence_rag_pipeline"
        if str(pipeline_dir) not in sys.path:
            sys.path.insert(0, str(pipeline_dir))

        from pipeline import answer_query  # imported lazily so app startup
        self._answer_query = answer_query   # doesn't fail if the folder is missing

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
            print(f"[CHECKPOINT] RealRetriever.retrieve called with query: {query}", flush=True)
        raw_results = self._answer_query(query)  # already ranked evidence
        print(f"[CHECKPOINT] answer_query returned {len(raw_results)} results", flush=True)

        documents: list[RetrievedDocument] = []
        for i, item in enumerate(raw_results[:top_k]):
            documents.append(
                RetrievedDocument(
                    document_id=f"{item.get('source', 'unknown')}-{i}",
                    title=item["text"][:80],
                    content=item["text"],
                    source=item.get("source") or "unknown",
                    source_type=(
                        "military_expenditure_data"
                        if item.get("source") == "sipri"
                        else "news_event"
                    ),
                    publication_date=item.get("seendate") or (
                        str(item["year"]) if item.get("year") else None
                    ),
                    metadata={
                        "country": item.get("country"),
                        "url": item.get("url"),
                    },
                    similarity_score=float(item.get("confidence", 0.0)),
                )
            )
        return documents


# ---------------------------------------------------------------------------
# Retrieval service — the only thing agents talk to
# ---------------------------------------------------------------------------

class RetrievalService:
    def __init__(self, retriever: Optional[BaseRetriever] = None):
        self.retriever = retriever or RealRetriever()

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        start = time.perf_counter()
        documents = self.retriever.retrieve(query, top_k=top_k)
        elapsed = round(time.perf_counter() - start, 4)

        return RetrievalResult(
            query=query,
            documents=documents,
            retrieval_time_seconds=elapsed,
            total_documents_found=len(documents),
        )


# Singleton convenience accessor, mirroring llm_service's pattern.
_retrieval_service_instance: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    global _retrieval_service_instance
    if _retrieval_service_instance is None:
        _retrieval_service_instance = RetrievalService()
    return _retrieval_service_instance