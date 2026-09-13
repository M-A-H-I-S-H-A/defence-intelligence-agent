"""
Shared LangGraph state for the Defence Intelligence workflow.

Every agent node reads from and writes to this single state object as it
flows through the graph: Retrieval -> Analysis -> Verification ->
(conditional: retry retrieval OR proceed) -> Risk -> Report -> Memory.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from schemas.contracts import (
    AnalysisResult,
    IntelligenceReport,
    RetrievalResult,
    RiskAssessment,
    VerificationResult,
)


class GraphState(TypedDict, total=False):
    # Input
    query: str
    session_id: str

    # Pipeline outputs, populated as the graph progresses
    retrieved_documents: Optional[RetrievalResult]
    analysis_result: Optional[AnalysisResult]
    verification_result: Optional[VerificationResult]
    risk_assessment: Optional[RiskAssessment]
    final_report: Optional[IntelligenceReport]

    # Memory / context carried into the run (e.g. prior session summary)
    memory_context: Optional[str]

    # Confidence used by the conditional routing check
    confidence: float

    # Retry control — prevents infinite retrieval loops
    retry_count: int
    max_retries: int

    # Errors accumulated during the run (non-fatal; agents degrade gracefully)
    errors: list[str]

    # Free-form metadata: timings, agent versions, etc.
    execution_metadata: dict