"""
Verification Agent — third node in the LangGraph workflow.

Responsibility: cross-check the Analysis Agent's facts/inferences against the
retrieved source documents, assess source reliability, detect contradictions
between sources, and decide whether evidence is sufficient to proceed to risk
assessment or whether the graph should retrieve more evidence and re-analyze.
"""

from __future__ import annotations

from typing import Optional

from schemas.contracts import (
    AnalysisResult,
    Contradiction,
    RetrievalResult,
    SourceAssessment,
    SourceReliability,
    VerificationResult,
    VerificationStatus,
)
from services.llm_service import LLMService, LLMServiceError, get_llm_service

MIN_DOCUMENTS_FOR_SUFFICIENCY = 2
MIN_CONFIDENCE_FOR_SUFFICIENCY = 0.4

SYSTEM_PROMPT = """You are a Defence Intelligence Verification Agent.

You will be given:
1. A user query
2. The source documents that were retrieved (with source, source_type, date)
3. The facts and inferences an Analysis Agent extracted from those documents

Your job:
1. Check whether each fact is actually supported by its cited document(s).
   Flag any fact that appears unsupported or overstated relative to the
   source content as an unsupported claim.
2. Assess the reliability of each distinct source using its source_type and
   content quality — do not blindly trust the source_type label alone; use
   judgement. As general guidance: government/NATO/RAND-style strategic
   reports and peer-reviewed research tend toward HIGH/MEDIUM reliability;
   OSINT and news sources tend toward MEDIUM; unverified or single-sourced
   claims with no corroboration tend toward LOW.
3. Compare documents against each other. If two sources make claims that
   directly conflict (e.g. one attributes an event to a state actor, another
   attributes it to a non-state actor; one cites a statistic that another
   contradicts), record it as a contradiction. Do NOT pick a winner — present
   both sides and note that confidence should be reduced.
4. Decide an overall verification status:
   - VERIFIED: claims are well-supported across multiple reliable sources
     with no contradictions.
   - PARTIALLY_VERIFIED: claims are supported but by limited or lower-
     reliability sources, or only partially corroborated.
   - UNVERIFIED: insufficient evidence to support the key claims.
   - CONTRADICTED: sources make directly conflicting claims about the same
     subject.
5. Give an overall confidence score (0.0-1.0).

Return ONLY a JSON object with this exact structure:
{
  "status": "VERIFIED" | "PARTIALLY_VERIFIED" | "UNVERIFIED" | "CONTRADICTED",
  "verified_claims": ["string", ...],
  "unsupported_claims": ["string", ...],
  "contradictions": [
    {"claim_a": "string", "source_a": "string", "claim_b": "string", "source_b": "string", "assessment": "string"}
  ],
  "source_assessments": [
    {"source": "string", "source_type": "string", "reliability": "HIGH" | "MEDIUM" | "LOW", "reasoning": "string"}
  ],
  "confidence": 0.0
}
"""


class VerificationAgent:
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or get_llm_service()

    def run(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        analysis_result: AnalysisResult,
    ) -> VerificationResult:
        if not retrieval_result.documents:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                evidence_sufficient=False,
                confidence=0.0,
            )

        user_prompt = self._build_prompt(query, retrieval_result, analysis_result)

        try:
            raw = self.llm_service.complete_json(SYSTEM_PROMPT, user_prompt)
        except LLMServiceError:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                evidence_sufficient=False,
                confidence=0.0,
            )

        result = self._parse_result(raw)
        result.evidence_sufficient = self._assess_sufficiency(
            result, retrieval_result
        )
        return result

    @staticmethod
    def _build_prompt(
        query: str,
        retrieval_result: RetrievalResult,
        analysis_result: AnalysisResult,
    ) -> str:
        sources_block = "\n".join(
            f"[{doc.document_id}] {doc.source} ({doc.source_type}, {doc.publication_date})\n"
            f"Title: {doc.title}\nContent: {doc.content}\n"
            for doc in retrieval_result.documents
        )
        facts_block = "\n".join(
            f"- {c.text} [supported by: {', '.join(c.supporting_document_ids)}]"
            for c in analysis_result.facts
        )
        inferences_block = "\n".join(
            f"- {c.text} [derived from: {', '.join(c.supporting_document_ids)}]"
            for c in analysis_result.inferences
        )

        return (
            f"User query: {query}\n\n"
            f"SOURCE DOCUMENTS:\n{sources_block}\n\n"
            f"FACTS TO VERIFY:\n{facts_block}\n\n"
            f"INFERENCES TO ASSESS:\n{inferences_block}\n"
        )

    @staticmethod
    def _parse_result(raw: dict) -> VerificationResult:
        status_raw = raw.get("status", "UNVERIFIED")
        try:
            status = VerificationStatus(status_raw)
        except ValueError:
            status = VerificationStatus.UNVERIFIED

        contradictions = [
            Contradiction(
                claim_a=c.get("claim_a", ""),
                source_a=c.get("source_a", ""),
                claim_b=c.get("claim_b", ""),
                source_b=c.get("source_b", ""),
                assessment=c.get("assessment", ""),
            )
            for c in raw.get("contradictions", [])
        ]

        source_assessments = []
        for s in raw.get("source_assessments", []):
            reliability_raw = s.get("reliability", "MEDIUM")
            try:
                reliability = SourceReliability(reliability_raw)
            except ValueError:
                reliability = SourceReliability.MEDIUM
            source_assessments.append(
                SourceAssessment(
                    source=s.get("source", ""),
                    source_type=s.get("source_type", ""),
                    reliability=reliability,
                    reasoning=s.get("reasoning", ""),
                )
            )

        return VerificationResult(
            status=status,
            verified_claims=raw.get("verified_claims", []),
            unsupported_claims=raw.get("unsupported_claims", []),
            contradictions=contradictions,
            source_assessments=source_assessments,
            confidence=float(raw.get("confidence", 0.0)),
            evidence_sufficient=False,  # set by _assess_sufficiency after this
        )

    @staticmethod
    def _assess_sufficiency(
        result: VerificationResult, retrieval_result: RetrievalResult
    ) -> bool:
        """
        Deterministic guardrail on top of the LLM's judgement: even if the
        LLM claims high confidence, don't treat evidence as sufficient if we
        simply don't have enough corroborating documents, or if status is
        UNVERIFIED/CONTRADICTED.
        """
        if result.status in (VerificationStatus.UNVERIFIED,):
            return False
        if len(retrieval_result.documents) < MIN_DOCUMENTS_FOR_SUFFICIENCY:
            return False
        if result.confidence < MIN_CONFIDENCE_FOR_SUFFICIENCY:
            return False
        return True


_verification_agent_instance: Optional[VerificationAgent] = None


def get_verification_agent() -> VerificationAgent:
    global _verification_agent_instance
    if _verification_agent_instance is None:
        _verification_agent_instance = VerificationAgent()
    return _verification_agent_instance