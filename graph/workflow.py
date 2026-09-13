"""
LangGraph workflow wiring together all six Defence Intelligence agents.

Retrieval -> Analysis -> Verification -> Confidence Check
                                            |
                            NO (retry available)  YES
                                    |                |
                            Retrieve Again      Risk Agent
                            (loop back to             |
                             Analysis)          Report Agent
                                                       |
                                                 Memory Agent
                                                       |
                                                      END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.analysis_agent import get_analysis_agent
from agents.memory_agent import get_memory_agent
from agents.report_agent import get_report_agent
from agents.retrieval_agent import get_retrieval_agent
from agents.risk_agent import get_risk_agent
from agents.verification_agent import get_verification_agent
from graph.state import GraphState

DEFAULT_MAX_RETRIES = 2
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Node functions — each wraps one agent and updates the shared state
# ---------------------------------------------------------------------------

def retrieval_node(state: GraphState) -> dict:
    agent = get_retrieval_agent()
    try:
        result = agent.run(state["query"], top_k=DEFAULT_TOP_K)
        return {"retrieved_documents": result}
    except Exception as exc:  # noqa: BLE001
        errors = state.get("errors", [])
        errors.append(f"RetrievalAgent error: {exc}")
        return {"errors": errors}


def analysis_node(state: GraphState) -> dict:
    agent = get_analysis_agent()
    retrieval_result = state.get("retrieved_documents")
    try:
        result = agent.run(state["query"], retrieval_result)
        return {"analysis_result": result}
    except Exception as exc:  # noqa: BLE001
        errors = state.get("errors", [])
        errors.append(f"AnalysisAgent error: {exc}")
        return {"errors": errors}


def verification_node(state: GraphState) -> dict:
    agent = get_verification_agent()
    retrieval_result = state.get("retrieved_documents")
    analysis_result = state.get("analysis_result")
    try:
        result = agent.run(state["query"], retrieval_result, analysis_result)
        return {
            "verification_result": result,
            "confidence": result.confidence,
        }
    except Exception as exc:  # noqa: BLE001
        errors = state.get("errors", [])
        errors.append(f"VerificationAgent error: {exc}")
        return {"errors": errors, "confidence": 0.0}


def retry_retrieval_node(state: GraphState) -> dict:
    """
    Bump the retry counter and widen retrieval (larger top_k) before looping
    back to Analysis. Keeps the same query — a more advanced version could
    reformulate the query here, but that's out of scope for v1.
    """
    retry_count = state.get("retry_count", 0) + 1
    agent = get_retrieval_agent()
    try:
        wider_top_k = DEFAULT_TOP_K + (retry_count * 3)
        result = agent.run(state["query"], top_k=wider_top_k)
        return {"retrieved_documents": result, "retry_count": retry_count}
    except Exception as exc:  # noqa: BLE001
        errors = state.get("errors", [])
        errors.append(f"RetryRetrieval error: {exc}")
        return {"errors": errors, "retry_count": retry_count}


def risk_node(state: GraphState) -> dict:
    agent = get_risk_agent()
    analysis_result = state.get("analysis_result")
    verification_result = state.get("verification_result")
    try:
        result = agent.run(state["query"], analysis_result, verification_result)
        return {"risk_assessment": result}
    except Exception as exc:  # noqa: BLE001
        errors = state.get("errors", [])
        errors.append(f"RiskAgent error: {exc}")
        return {"errors": errors}


def report_node(state: GraphState) -> dict:
    agent = get_report_agent()
    try:
        result = agent.run(
            state["query"],
            state.get("retrieved_documents"),
            state.get("analysis_result"),
            state.get("verification_result"),
            state.get("risk_assessment"),
        )
        return {"final_report": result}
    except Exception as exc:  # noqa: BLE001
        errors = state.get("errors", [])
        errors.append(f"ReportAgent error: {exc}")
        return {"errors": errors}


def memory_node(state: GraphState) -> dict:
    agent = get_memory_agent()
    try:
        agent.save(
            session_id=state.get("session_id", "default"),
            query=state["query"],
            report=state.get("final_report"),
        )
    except Exception as exc:  # noqa: BLE001
        errors = state.get("errors", [])
        errors.append(f"MemoryAgent error: {exc}")
        return {"errors": errors}
    return {}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_verification(state: GraphState) -> str:
    """
    Decide whether evidence is sufficient to proceed to Risk Assessment, or
    whether we should retrieve more evidence and re-analyze. Bounded by
    max_retries to prevent an infinite loop (spec rule #11).
    """
    verification_result = state.get("verification_result")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    evidence_sufficient = bool(
        verification_result and verification_result.evidence_sufficient
    )

    if evidence_sufficient:
        return "sufficient"

    if retry_count >= max_retries:
        # Out of retries — proceed anyway with whatever evidence we have,
        # rather than looping forever or dead-ending the graph.
        return "sufficient"

    return "insufficient"


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

def build_workflow():
    builder = StateGraph(GraphState)

    builder.add_node("retrieval", retrieval_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("verification", verification_node)
    builder.add_node("retry_retrieval", retry_retrieval_node)
    builder.add_node("risk", risk_node)
    builder.add_node("report", report_node)
    builder.add_node("memory", memory_node)

    builder.add_edge(START, "retrieval")
    builder.add_edge("retrieval", "analysis")
    builder.add_edge("analysis", "verification")

    builder.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "sufficient": "risk",
            "insufficient": "retry_retrieval",
        },
    )

    # Loop back: retry_retrieval -> analysis -> verification -> (check again)
    builder.add_edge("retry_retrieval", "analysis")

    builder.add_edge("risk", "report")
    builder.add_edge("report", "memory")
    builder.add_edge("memory", END)

    return builder.compile()


_compiled_workflow = None


def get_workflow():
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = build_workflow()
    return _compiled_workflow


def run_workflow(
    query: str,
    session_id: str = "default",
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> GraphState:
    """
    Convenience entrypoint: run the full workflow for a single query and
    return the final state.
    """
    workflow = get_workflow()
    initial_state: GraphState = {
        "query": query,
        "session_id": session_id,
        "retry_count": 0,
        "max_retries": max_retries,
        "errors": [],
        "execution_metadata": {},
    }
    final_state = workflow.invoke(initial_state)
    return final_state