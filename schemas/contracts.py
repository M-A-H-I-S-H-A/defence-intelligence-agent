"""
Pydantic data contracts shared across all agents in the Defence Intelligence
multi-agent system. Every agent receives and returns one of these models —
never raw strings or dicts — so the LangGraph workflow stays predictable.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SourceReliability(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ClaimType(str, Enum):
    FACT = "FACT"
    INFERENCE = "INFERENCE"
    UNCERTAINTY = "UNCERTAINTY"


# ---------------------------------------------------------------------------
# 1. UserQuery
# ---------------------------------------------------------------------------

class UserQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 2. RetrievedDocument / RetrievalResult
# ---------------------------------------------------------------------------

class RetrievedDocument(BaseModel):
    document_id: str
    title: str
    content: str
    source: str
    source_type: str
    publication_date: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    similarity_score: float = Field(ge=0.0, le=1.0)


class RetrievalResult(BaseModel):
    query: str
    documents: list[RetrievedDocument] = Field(default_factory=list)
    retrieval_time_seconds: Optional[float] = None
    total_documents_found: int = 0


# ---------------------------------------------------------------------------
# 3. AnalysisResult
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    text: str
    claim_type: ClaimType
    supporting_document_ids: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    key_findings: list[str] = Field(default_factory=list)
    facts: list[Claim] = Field(default_factory=list)
    inferences: list[Claim] = Field(default_factory=list)
    threat_indicators: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


# ---------------------------------------------------------------------------
# 4. VerificationResult
# ---------------------------------------------------------------------------

class SourceAssessment(BaseModel):
    source: str
    source_type: str
    reliability: SourceReliability
    reasoning: str


class Contradiction(BaseModel):
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    assessment: str


class VerificationResult(BaseModel):
    status: VerificationStatus
    verified_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    source_assessments: list[SourceAssessment] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence_sufficient: bool = False


# ---------------------------------------------------------------------------
# 5. RiskAssessment
# ---------------------------------------------------------------------------

class RiskAssessment(BaseModel):
    risk_level: RiskLevel
    risk_score: int = Field(ge=0, le=100)
    risk_factors: list[str] = Field(default_factory=list)
    threat_indicators: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


# ---------------------------------------------------------------------------
# 6. IntelligenceReport
# ---------------------------------------------------------------------------

class EvidenceReference(BaseModel):
    finding: str
    evidence: str
    source: str
    document_id: str
    publication_date: Optional[str] = None
    relevance: Optional[float] = None
    verification_status: VerificationStatus


class IntelligenceReport(BaseModel):
    executive_summary: str
    threat_overview: str
    key_findings: list[str] = Field(default_factory=list)
    verified_facts: list[str] = Field(default_factory=list)
    analytical_assessment: str = ""
    risk_assessment: RiskAssessment
    risk_factors: list[str] = Field(default_factory=list)
    supporting_evidence: list[EvidenceReference] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    uncertainties: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 7. MemoryRecord
# ---------------------------------------------------------------------------

class MemoryRecord(BaseModel):
    session_id: str
    query: str
    report: Optional[IntelligenceReport] = None
    findings_summary: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 8. AgentError
# ---------------------------------------------------------------------------

class AgentError(BaseModel):
    agent_name: str
    error_type: str
    message: str
    recoverable: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)