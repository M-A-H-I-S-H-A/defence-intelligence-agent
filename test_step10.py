"""
Manual smoke test for the FULL pipeline:
Retrieval -> Analysis -> Verification -> Risk -> Report

This is the first time we see a complete IntelligenceReport assembled.
"""

from agents.retrieval_agent import get_retrieval_agent
from agents.analysis_agent import get_analysis_agent
from agents.verification_agent import get_verification_agent
from agents.risk_agent import get_risk_agent
from agents.report_agent import get_report_agent

query = "What are the emerging cybersecurity threats to critical infrastructure?"

retrieval_agent = get_retrieval_agent()
analysis_agent = get_analysis_agent()
verification_agent = get_verification_agent()
risk_agent = get_risk_agent()
report_agent = get_report_agent()

retrieval_result = retrieval_agent.run(query, top_k=5)
analysis_result = analysis_agent.run(query, retrieval_result)
verification_result = verification_agent.run(query, retrieval_result, analysis_result)
risk_result = risk_agent.run(query, analysis_result, verification_result)

report = report_agent.run(
    query, retrieval_result, analysis_result, verification_result, risk_result
)

print("=" * 70)
print("DEFENCE INTELLIGENCE REPORT")
print("=" * 70)

print("\nEXECUTIVE SUMMARY:")
print(report.executive_summary)

print("\nTHREAT OVERVIEW:")
print(report.threat_overview)

print("\nKEY FINDINGS:")
for f in report.key_findings:
    print(f"  - {f}")

print("\nVERIFIED FACTS:")
for f in report.verified_facts:
    print(f"  - {f}")

print("\nANALYTICAL ASSESSMENT:")
print(report.analytical_assessment)

print(f"\nRISK: {report.risk_assessment.risk_level.value} (score {report.risk_assessment.risk_score})")

print("\nRISK FACTORS:")
for f in report.risk_factors:
    print(f"  - {f}")

print(f"\nSUPPORTING EVIDENCE ({len(report.supporting_evidence)} references):")
for ev in report.supporting_evidence:
    print(f"  [{ev.document_id}] {ev.source} ({ev.publication_date}) -> {ev.verification_status.value}")
    print(f"      Finding: {ev.finding}")

print(f"\nSOURCES: {report.sources}")

print(f"\nOVERALL CONFIDENCE: {report.confidence}")

print("\nUNCERTAINTIES:")
for u in report.uncertainties:
    print(f"  - {u}")

print("\nRECOMMENDATIONS:")
for r in report.recommendations:
    print(f"  - {r}")

print(f"\nGenerated at: {report.generated_at}")

# Sanity check: confirm the model is a fully valid Pydantic object
print("\n--- Pydantic validation check ---")
print(f"Report is valid IntelligenceReport instance: {type(report).__name__ == 'IntelligenceReport'}")