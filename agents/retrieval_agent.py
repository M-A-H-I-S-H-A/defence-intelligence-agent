"""
Retrieval Agent — first node in the LangGraph workflow.

Responsibility: given a user query, fetch the most relevant evidence via
RetrievalService, optionally filter low-relevance noise, and return a
structured RetrievalResult. This agent is source-agnostic: it never inspects
or branches on which source/source_type a document came from.
"""

from __future__ import annotations

from typing import Optional

from schemas.contracts import RetrievalResult
from services.retrieval_service import RetrievalService, get_retrieval_service

MIN_RELEVANCE_THRESHOLD = 0.05  # filters out near-zero-overlap noise


class RetrievalAgent:
    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        min_relevance: float = MIN_RELEVANCE_THRESHOLD,
    ):
        self.retrieval_service = retrieval_service or get_retrieval_service()
        self.min_relevance = min_relevance

    def run(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve evidence for a query, filter out documents below the
        minimum relevance threshold, and return the structured result.
        """
        result = self.retrieval_service.search(query, top_k=top_k)

        filtered_docs = [
            doc for doc in result.documents
            if doc.similarity_score >= self.min_relevance
        ]

        return RetrievalResult(
            query=result.query,
            documents=filtered_docs,
            retrieval_time_seconds=result.retrieval_time_seconds,
            total_documents_found=len(filtered_docs),
        )


# Convenience singleton, consistent with the service layer pattern.
_retrieval_agent_instance: Optional[RetrievalAgent] = None


def get_retrieval_agent() -> RetrievalAgent:
    global _retrieval_agent_instance
    if _retrieval_agent_instance is None:
        _retrieval_agent_instance = RetrievalAgent()
    return _retrieval_agent_instance