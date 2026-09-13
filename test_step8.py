"""
Manual smoke test for Retrieval -> Analysis -> Verification.
Targets the DOC001/DOC006 contradiction (state-linked vs non-state actor
attribution for the same cyber reconnaissance activity) to check whether
the Verification Agent correctly flags it.
"""

from agents.retrieval_agent import get_retrieval_agent
from agents.analysis_agent import get_analysis_agent
from agents.verification_agent import get_verification_agent

query = "What are the emerging cybersecurity threats to critical infrastructure?"

retrieval_agent = get_retrieval_agent()
analysis_agent = get_analysis_agent()
verification_agent = get_verification_agent()

retrieval_result = retrieval_agent.run(query, top_k=5)
print(f"Retrieved {retrieval_result.total_documents_found} documents\n")

analysis_result = analysis_agent.run(query, retrieval_result)
print(f"Analysis confidence: {analysis_result.confidence}\n")

verification_result = verification_agent.run(query, retrieval_result, analysis_result)

print(f"STATUS: {verification_result.status}")
print(f"CONFIDENCE: {verification_result.confidence}")
print(f"EVIDENCE SUFFICIENT: {verification_result.evidence_sufficient}\n")

print("VERIFIED CLAIMS:")
for c in verification_result.verified_claims:
    print(f"  - {c}")

print("\nUNSUPPORTED CLAIMS:")
for c in verification_result.unsupported_claims:
    print(f"  - {c}")

print("\nCONTRADICTIONS:")
for c in verification_result.contradictions:
    print(f"  Claim A ({c.source_a}): {c.claim_a}")
    print(f"  Claim B ({c.source_b}): {c.claim_b}")
    print(f"  Assessment: {c.assessment}\n")

print("SOURCE ASSESSMENTS:")
for s in verification_result.source_assessments:
    print(f"  {s.source} ({s.source_type}) -> {s.reliability}: {s.reasoning}")