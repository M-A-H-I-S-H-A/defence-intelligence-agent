"""
Full end-to-end test of the compiled LangGraph workflow.
Runs a single query through: Retrieval -> Analysis -> Verification ->
(conditional retry loop) -> Risk -> Report -> Memory.
"""

from graph.workflow import run_workflow
from agents.memory_agent import get_memory_agent

query = "What are the emerging cybersecurity threats to critical infrastructure?"
session_id = "test_session_001"

print("Running full workflow...\n")
final_state = run_workflow(query, session_id=session_id)

print("=" * 70)
print("WORKFLOW COMPLETE")
print("=" * 70)

print(f"\nRetry count: {final_state.get('retry_count', 0)}")
print(f"Errors: {final_state.get('errors', [])}")

report = final_state.get("final_report")
if report:
    print(f"\nExecutive Summary:\n{report.executive_summary}")
    print(f"\nRisk: {report.risk_assessment.risk_level.value} (score {report.risk_assessment.risk_score})")
    print(f"Overall confidence: {report.confidence}")
    print(f"Sources: {report.sources}")
else:
    print("\nNo final report was generated — check errors above.")

# Confirm memory actually got the record
memory_agent = get_memory_agent()
history = memory_agent.get_history(session_id)
print(f"\nMemory records for session '{session_id}': {len(history)}")
if history:
    print(f"Last saved query: {history[-1].query}")