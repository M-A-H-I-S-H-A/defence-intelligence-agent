"""
Analysis Agent — second node in the LangGraph workflow.

Responsibility: read retrieved evidence and produce structured analytical
findings (key findings, facts, inferences, threats, entities, events).
Must NEVER invent information not grounded in the retrieved documents.
Every fact/inference should reference the document_id(s) that support it.
"""

from __future__ import annotations

from typing import Optional

from schemas.contracts import (
    AnalysisResult,
    Claim,
    ClaimType,
    RetrievalResult,
)
from services.llm_service import LLMService, LLMServiceError, get_llm_service

SYSTEM_PROMPT = """You are a Defence Intelligence Analysis Agent.

You will be given a set of retrieved documents (with document_id, source,
source_type, and content) related to a user's intelligence query.

Your job:
1. Extract key findings relevant to the query.
2. Separate FACTS (directly stated in the documents) from INFERENCES
   (analytical conclusions you draw by connecting evidence across documents).
3. Identify threat indicators, entities (organizations, states, groups),
   events, and any notable patterns or relationships across documents.
4. NEVER invent information that is not supported by the provided documents.
   If evidence is thin, say so rather than fabricating detail.
5. For every fact and inference, list which document_id(s) support it.

Return ONLY a JSON object with this exact structure:
{
  "key_findings": ["string", ...],
  "facts": [{"text": "string", "supporting_document_ids": ["DOC001", ...]}, ...],
  "inferences": [{"text": "string", "supporting_document_ids": ["DOC001", ...]}, ...],
  "threat_indicators": ["string", ...],
  "entities": ["string", ...],
  "events": ["string", ...],
  "evidence": ["string", ...],
  "confidence": 0.0
}

"confidence" is your overall confidence (0.0-1.0) in the analysis given the
quantity and quality of retrieved evidence. Low document count or thin
overlap with the query should lower confidence.
"""


class AnalysisAgent:
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or get_llm_service()

    def run(self, query: str, retrieval_result: RetrievalResult) -> AnalysisResult:
        if not retrieval_result.documents:
            return AnalysisResult(
                key_findings=[],
                facts=[],
                inferences=[],
                threat_indicators=[],
                entities=[],
                events=[],
                evidence=[],
                confidence=0.0,
            )

        documents_block = self._format_documents(retrieval_result)
        user_prompt = (
            f"User query: {query}\n\n"
            f"Retrieved documents:\n{documents_block}"
        )

        try:
            raw = self.llm_service.complete_json(SYSTEM_PROMPT, user_prompt)
        except LLMServiceError:
            # Fail safe: return a low-confidence empty-ish result rather than
            # crashing the workflow. The graph's confidence check will route
            # this toward retry/insufficient-evidence handling.
            return AnalysisResult(confidence=0.0)

        return self._parse_result(raw)

    @staticmethod
    def _format_documents(retrieval_result: RetrievalResult) -> str:
        blocks = []
        for doc in retrieval_result.documents:
            blocks.append(
                f"[{doc.document_id}] Source: {doc.source} ({doc.source_type}) | "
                f"Date: {doc.publication_date}\n"
                f"Title: {doc.title}\n"
                f"Content: {doc.content}\n"
            )
        return "\n".join(blocks)

    @staticmethod
    def _parse_result(raw: dict) -> AnalysisResult:
        def to_claims(items: list, claim_type: ClaimType) -> list[Claim]:
            claims = []
            for item in items or []:
                if isinstance(item, dict):
                    claims.append(
                        Claim(
                            text=item.get("text", ""),
                            claim_type=claim_type,
                            supporting_document_ids=item.get(
                                "supporting_document_ids", []
                            ),
                        )
                    )
                elif isinstance(item, str):
                    # Fallback if the model returns plain strings instead of
                    # the requested object shape.
                    claims.append(
                        Claim(text=item, claim_type=claim_type, supporting_document_ids=[])
                    )
            return claims

        return AnalysisResult(
            key_findings=raw.get("key_findings", []),
            facts=to_claims(raw.get("facts", []), ClaimType.FACT),
            inferences=to_claims(raw.get("inferences", []), ClaimType.INFERENCE),
            threat_indicators=raw.get("threat_indicators", []),
            entities=raw.get("entities", []),
            events=raw.get("events", []),
            evidence=raw.get("evidence", []),
            confidence=float(raw.get("confidence", 0.0)),
        )


_analysis_agent_instance: Optional[AnalysisAgent] = None


def get_analysis_agent() -> AnalysisAgent:
    global _analysis_agent_instance
    if _analysis_agent_instance is None:
        _analysis_agent_instance = AnalysisAgent()
    return _analysis_agent_instance