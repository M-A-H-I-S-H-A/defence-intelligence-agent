"""
Memory Agent — sixth node in the LangGraph workflow.

Responsibility: store user queries, session history, and previous
intelligence reports so future queries in the same session can be given
relevant context. In-memory for the first working version (per spec, this
is NOT mandatory to be persistent yet); structured so SQLite/Redis can be
swapped in later without changing the agent's public interface.
"""

from __future__ import annotations

from typing import Optional

from schemas.contracts import IntelligenceReport, MemoryRecord


class MemoryAgent:
    def __init__(self):
        # session_id -> list of MemoryRecord, most recent last
        self._store: dict[str, list[MemoryRecord]] = {}

    def save(
        self,
        session_id: str,
        query: str,
        report: Optional[IntelligenceReport] = None,
    ) -> MemoryRecord:
        """
        Save a completed query + report into session history.
        """
        findings_summary = None
        if report is not None:
            findings_summary = report.executive_summary

        record = MemoryRecord(
            session_id=session_id,
            query=query,
            report=report,
            findings_summary=findings_summary,
        )

        self._store.setdefault(session_id, []).append(record)
        return record

    def get_history(self, session_id: str) -> list[MemoryRecord]:
        """
        Return all records for a session, oldest first.
        """
        return list(self._store.get(session_id, []))

    def get_recent_context(self, session_id: str, limit: int = 3) -> str:
        """
        Return a compact text summary of the most recent findings in this
        session, useful for giving agents context on follow-up queries.
        Returns an empty string if there's no history yet.
        """
        history = self.get_history(session_id)
        if not history:
            return ""

        recent = history[-limit:]
        lines = []
        for record in recent:
            summary = record.findings_summary or "(no report generated)"
            lines.append(f"- Query: {record.query}\n  Summary: {summary}")
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        """Remove all history for a session."""
        self._store.pop(session_id, None)


_memory_agent_instance: Optional[MemoryAgent] = None


def get_memory_agent() -> MemoryAgent:
    global _memory_agent_instance
    if _memory_agent_instance is None:
        _memory_agent_instance = MemoryAgent()
    return _memory_agent_instance