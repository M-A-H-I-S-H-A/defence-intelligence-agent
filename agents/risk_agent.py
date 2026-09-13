"""
Risk Assessment Agent — fourth node in the LangGraph workflow.

Responsibility: given verified findings (and awareness of contradictions/
confidence from the Verification Agent), assess risk level and score, list
risk factors and threat indicators, and explain the reasoning while clearly
distinguishing evidence from assessment from speculation.
"""

from __future__ import annotations

from typing import Optional

from schemas.contracts import (
    AnalysisResult,
    RiskAssessment,
    RiskLevel,
    VerificationResult,
)
from services.llm_service import LLMService, LLMServiceError, get_llm_service

SYSTEM_PROMPT = """You are a Defence Intelligence Risk Assessment Agent.

You will be given verified analytical findings, threat indicators, and the
verification status/confidence for a defence intelligence query.

Your job:
1. Identify concrete risk factors drawn from the verified findings and threat
   indicators.
2. Assess severity and potential impact of these risk factors.
3. Assign a risk_level: LOW, MEDIUM, HIGH, or CRITICAL.
4. Assign a risk_score from 0-100 consistent with that level (roughly:
   LOW 0-25, MEDIUM 26-50, HIGH 51-75, CRITICAL 76-100).
5. Write a reasoning_summary that clearly separates:
   - EVIDENCE: what the sources directly state
   - ASSESSMENT: your analytical judgement based on that evidence
   - SPECULATION: anything uncertain or forward-looking, explicitly labeled
     as such and not presented as fact
6. If verification status is CONTRADICTED or confidence is low, factor that
   uncertainty into a lower confidence score and note it in your reasoning —
   do not silently ignore contradictions or unresolved evidence.
7. Give an overall confidence score (0.0-1.0) for this risk assessment.

Return ONLY a JSON object with this exact structure:
{
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "risk_score": 0,
  "risk_factors": ["string", ...],
  "threat_indicators": ["string", ...],
  "reasoning_summary": "string",
  "confidence": 0.0
}
"""


class RiskAgent:
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or get_llm_service()

    def run(
        self,
        query: str,
        analysis_result: AnalysisResult,
        verification_result: VerificationResult,
    ) -> RiskAssessment:
        user_prompt = self._build_prompt(query, analysis_result, verification_result)

        try:
            raw = self.llm_service.complete_json(SYSTEM_PROMPT, user_prompt)
        except LLMServiceError:
            return RiskAssessment(
                risk_level=RiskLevel.LOW,
                risk_score=0,
                risk_factors=[],
                threat_indicators=[],
                reasoning_summary=(
                    "Risk assessment could not be completed due to an LLM "
                    "service failure. Defaulting to LOW as a safe fallback; "
                    "this should be manually reviewed."
                ),
                confidence=0.0,
            )

        return self._parse_result(raw)

    @staticmethod
    def _build_prompt(
        query: str,
        analysis_result: AnalysisResult,
        verification_result: VerificationResult,
    ) -> str:
        verified_block = "\n".join(f"- {c}" for c in verification_result.verified_claims)
        unsupported_block = "\n".join(
            f"- {c}" for c in verification_result.unsupported_claims
        )
        contradictions_block = "\n".join(
            f"- {c.source_a} says: {c.claim_a} | {c.source_b} says: {c.claim_b} | "
            f"Assessment: {c.assessment}"
            for c in verification_result.contradictions
        )
        threat_indicators_block = "\n".join(
            f"- {t}" for t in analysis_result.threat_indicators
        )

        return (
            f"User query: {query}\n\n"
            f"VERIFICATION STATUS: {verification_result.status.value}\n"
            f"VERIFICATION CONFIDENCE: {verification_result.confidence}\n\n"
            f"VERIFIED CLAIMS:\n{verified_block or '(none)'}\n\n"
            f"UNSUPPORTED/SPECULATIVE CLAIMS (do not treat as fact):\n"
            f"{unsupported_block or '(none)'}\n\n"
            f"CONTRADICTIONS BETWEEN SOURCES:\n{contradictions_block or '(none)'}\n\n"
            f"THREAT INDICATORS FROM ANALYSIS:\n{threat_indicators_block or '(none)'}\n"
        )

    @staticmethod
    def _parse_result(raw: dict) -> RiskAssessment:
        level_raw = raw.get("risk_level", "LOW")
        try:
            risk_level = RiskLevel(level_raw)
        except ValueError:
            risk_level = RiskLevel.LOW

        score = raw.get("risk_score", 0)
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(100, score))

        return RiskAssessment(
            risk_level=risk_level,
            risk_score=score,
            risk_factors=raw.get("risk_factors", []),
            threat_indicators=raw.get("threat_indicators", []),
            reasoning_summary=raw.get("reasoning_summary", ""),
            confidence=float(raw.get("confidence", 0.0)),
        )


_risk_agent_instance: Optional[RiskAgent] = None


def get_risk_agent() -> RiskAgent:
    global _risk_agent_instance
    if _risk_agent_instance is None:
        _risk_agent_instance = RiskAgent()
    return _risk_agent_instance