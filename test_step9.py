"""
Manual smoke test for Retrieval -> Analysis -> Verification -> Risk.
Full pipeline test on the cybersecurity query, including the known
DOC001/DOC006 contradiction, to see how Risk Agent handles it.
"""

from agents.retrieval_agent import get_retrieval_agent
from agents.analysis_agent import get_analysis_agent
from agents.verification_agent import get_verification_agent
from agents.risk_agent import get_risk_agent

query = "What are the emerging cybersecurity threats to critical infrastructure?"

retrieval_agent = get_retrieval_agent()
analysis_agent = get_analysis_agent()
verification_agent = get_verification_agent()
risk_agent = get_risk_agent()

retrieval_result = retrieval_agent.run(query, top_k=5)
print(f"Retrieved {retrieval_result.total_documents_found} documents")

analysis_result = analysis_agent.run(query, retrieval_result)
print(f"Analysis confidence: {analysis_result.confidence}")

verification_result = verification_agent.run(query, retrieval_result, analysis_result)
print(f"Verification status: {verification_result.status}, confidence: {verification_result.confidence}")
print(f"Contradictions found: {len(verification_result.contradictions)}\n")

risk_result = risk_agent.run(query, analysis_result, verification_result)

print("=" * 60)
print(f"RISK LEVEL: {risk_result.risk_level}")
print(f"RISK SCORE: {risk_result.risk_score}")
print(f"CONFIDENCE: {risk_result.confidence}")
print("=" * 60)

print("\nRISK FACTORS:")
for f in risk_result.risk_factors:
    print(f"  - {f}")

print("\nTHREAT INDICATORS:")
for t in risk_result.threat_indicators:
    print(f"  - {t}")

print("\nREASONING SUMMARY:")
print(risk_result.reasoning_summary)