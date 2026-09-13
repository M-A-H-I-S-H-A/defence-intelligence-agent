"""
Manual smoke test for Retrieval Agent -> Analysis Agent.
Not part of the permanent tests/ suite (that comes in Phase 6) —
just a quick way to eyeball the output shape right now.
"""

from agents.retrieval_agent import get_retrieval_agent
from agents.analysis_agent import get_analysis_agent

query = "What are the emerging cybersecurity threats to critical infrastructure?"

retrieval_agent = get_retrieval_agent()
analysis_agent = get_analysis_agent()

retrieval_result = retrieval_agent.run(query, top_k=5)
print(f"Retrieved {retrieval_result.total_documents_found} documents:")
for doc in retrieval_result.documents:
    print(f"  {doc.document_id} ({doc.similarity_score}) - {doc.title}")

print("\nRunning analysis...\n")
analysis_result = analysis_agent.run(query, retrieval_result)

print("KEY FINDINGS:")
for f in analysis_result.key_findings:
    print(f"  - {f}")

print("\nFACTS:")
for fact in analysis_result.facts:
    print(f"  - {fact.text}  [{fact.supporting_document_ids}]")

print("\nINFERENCES:")
for inf in analysis_result.inferences:
    print(f"  - {inf.text}  [{inf.supporting_document_ids}]")

print(f"\nTHREAT INDICATORS: {analysis_result.threat_indicators}")
print(f"ENTITIES: {analysis_result.entities}")
print(f"CONFIDENCE: {analysis_result.confidence}")