"""
Report Generation Agent — fifth node in the LangGraph workflow.

Responsibility: assemble the outputs of Analysis, Verification, and Risk
into a single structured IntelligenceReport with all required sections.
Every major finding should reference its supporting evidence/source.
Recommendations must be clearly labeled as recommendations, never presented
as established fact.
"""

from __future__ import annotations

from typing import Optional

from schemas.contracts import (
    AnalysisResult,
    EvidenceReference,
    IntelligenceReport,
    RetrievalResult,
    RiskAssessment,
    VerificationResult,
    VerificationStatus,
)
from services.llm_service import LLMService, LLMServiceError, get_llm_service

SYSTEM_PROMPT = """You are a Defence Intelligence Report Generation Agent.

You will be given the full analytical pipeline output for a user query:
retrieved source documents, analysis findings, verification results, and a
risk assessment.

Your job is to write the narrative sections of a structured Defence
Intelligence Report:
1. executive_summary: 3-5 sentences summarizing the situation for a busy
   decision-maker.
2. threat_overview: a concise overview of the threat landscape identified.
3. analytical_assessment: a fuller narrative synthesis of the analysis and
   verification findings, including how contradictions (if any) affect
   confidence.
4. uncertainties: a list of specific things that remain unverified, unclear,
   or contradicted — be concrete, not vague.
5. recommendations: a list of suggested next steps or actions. Every item
   MUST be phrased as a recommendation (e.g. "Consider...", "Recommend
   monitoring...", "Further investigation should...") and must never be
   phrased as an established fact.

Ground everything in the provided findings. Do not introduce new factual
claims not present in the analysis/verification/risk data you were given.

Return ONLY a JSON object with this exact structure:
{
  "executive_summary": "string",
  "threat_overview": "string",
  "analytical_assessment": "string",
  "uncertainties": ["string", ...],
  "recommendations": ["string", ...]
}
"""


class ReportAgent:
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or get_llm_service()

    def run(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        analysis_result: AnalysisResult,
        verification_result: VerificationResult,
        risk_assessment: RiskAssessment,
    ) -> IntelligenceReport:
        user_prompt = self._build_prompt(
            query, retrieval_result, analysis_result, verification_result, risk_assessment
        )

        try:
            raw = self.llm_service.complete_json(SYSTEM_PROMPT, user_prompt)
        except LLMServiceError:
            raw = {
                "executive_summary": (
                    "Report generation failed due to an LLM service error. "
                    "Findings below are drawn directly from prior pipeline "
                    "stages without narrative synthesis."
                ),
                "threat_overview": "",
                "analytical_assessment": "",
                "uncertainties": ["Report narrative could not be generated."],
                "recommendations": ["Retry report generation once the LLM service is available."],
            }

        supporting_evidence = self._build_evidence_references(
            retrieval_result, verification_result
        )
        sources = sorted({doc.source for doc in retrieval_result.documents})

        overall_confidence = min(
            analysis_result.confidence,
            verification_result.confidence,
            risk_assessment.confidence,
        )

        return IntelligenceReport(
            executive_summary=raw.get("executive_summary", ""),
            threat_overview=raw.get("threat_overview", ""),
            key_findings=analysis_result.key_findings,
            verified_facts=verification_result.verified_claims,
            analytical_assessment=raw.get("analytical_assessment", ""),
            risk_assessment=risk_assessment,
            risk_factors=risk_assessment.risk_factors,
            supporting_evidence=supporting_evidence,
            sources=sources,
            confidence=round(overall_confidence, 4),
            uncertainties=raw.get("uncertainties", []),
            recommendations=raw.get("recommendations", []),
        )

    @staticmethod
    def _build_prompt(
        query: str,
        retrieval_result: RetrievalResult,
        analysis_result: AnalysisResult,
        verification_result: VerificationResult,
        risk_assessment: RiskAssessment,
    ) -> str:
        contradictions_block = "\n".join(
            f"- {c.source_a} vs {c.source_b}: {c.assessment}"
            for c in verification_result.contradictions
        )
        return (
            f"User query: {query}\n\n"
            f"KEY FINDINGS:\n" + "\n".join(f"- {f}" for f in analysis_result.key_findings) + "\n\n"
            f"VERIFIED CLAIMS:\n" + "\n".join(f"- {c}" for c in verification_result.verified_claims) + "\n\n"
            f"UNSUPPORTED CLAIMS:\n" + "\n".join(f"- {c}" for c in verification_result.unsupported_claims) + "\n\n"
            f"CONTRADICTIONS:\n{contradictions_block or '(none)'}\n\n"
            f"VERIFICATION STATUS: {verification_result.status.value} "
            f"(confidence {verification_result.confidence})\n\n"
            f"RISK LEVEL: {risk_assessment.risk_level.value} "
            f"(score {risk_assessment.risk_score}, confidence {risk_assessment.confidence})\n"
            f"RISK REASONING: {risk_assessment.reasoning_summary}\n"
        )

    @staticmethod
    def _build_evidence_references(
        retrieval_result: RetrievalResult,
        verification_result: VerificationResult,
    ) -> list[EvidenceReference]:
        """
        Build one evidence reference per retrieved document, linking it back
        to its verification status so the report (and dashboard) can show
        Finding -> Evidence -> Source -> Verified?

        A document only inherits CONTRADICTED if it is actually named as one
        side of a specific contradiction. The Verification Agent's LLM is
        inconsistent about whether it puts a source NAME or a document_id in
        source_a/source_b, so we match against both to be safe. Documents
        not involved in any contradiction never inherit the pipeline's
        overall status blindly — that would mislabel unrelated evidence
        (e.g. a NATO hypersonics report showing as CONTRADICTED just because
        some other document pair disagreed about cyber attribution).
        Unrelated documents default to PARTIALLY_VERIFIED, reflecting that
        they contributed evidence without being individually confirmed or
        refuted.
        """
        contradicted_markers = set()
        for c in verification_result.contradictions:
            contradicted_markers.add(c.source_a)
            contradicted_markers.add(c.source_b)

        references = []
        for doc in retrieval_result.documents:
            is_contradicted = (
                doc.source in contradicted_markers
                or doc.document_id in contradicted_markers
            )
            if is_contradicted:
                doc_status = VerificationStatus.CONTRADICTED
            elif any(doc.source in claim for claim in verification_result.verified_claims):
                doc_status = VerificationStatus.VERIFIED
            else:
                doc_status = VerificationStatus.PARTIALLY_VERIFIED

            references.append(
                EvidenceReference(
                    finding=doc.title,
                    evidence=doc.content[:300],
                    source=doc.source,
                    document_id=doc.document_id,
                    publication_date=doc.publication_date,
                    relevance=doc.similarity_score,
                    verification_status=doc_status,
                )
            )
        return references


_report_agent_instance: Optional[ReportAgent] = None


def get_report_agent() -> ReportAgent:
    global _report_agent_instance
    if _report_agent_instance is None:
        _report_agent_instance = ReportAgent()
    return _report_agent_instance