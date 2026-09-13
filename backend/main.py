"""
FastAPI backend for the Defence Intelligence multi-agent system.

Exposes the LangGraph workflow (graph/workflow.py) as a REST API so a
frontend (or curl, or Postman) can submit a question and get back a
structured IntelligenceReport as JSON.

Endpoints:
    GET  /health           - liveness check, no auth required
    POST /assess           - run the full agent workflow for a question
                              (requires the X-API-Key header)

Run with:
    uvicorn backend.main:app --reload --port 8000
(from the project root, so the `agents`, `graph`, `schemas` etc.
 packages are importable)
"""

from __future__ import annotations

import os
import secrets
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from graph.workflow import run_workflow

load_dotenv()

# ---------------------------------------------------------------------------
# Basic authentication
# ---------------------------------------------------------------------------
# Simple shared-secret API key, read from the environment. This is
# intentionally lightweight (item #11 in the brief just asks for "basic
# authentication", not a full user/login system) -- good enough to stop the
# dashboard being wide open to anyone who finds the URL, not meant to be
# production-grade security.
API_KEY = os.getenv("APP_API_KEY")
if not API_KEY:
    # Generate one at startup so the app still runs locally without any
    # extra setup, and print it once so whoever started the server can
    # copy it into the frontend / their API client.
    API_KEY = secrets.token_urlsafe(24)
    print(f"\n[auth] No APP_API_KEY set in .env -- generated one for this run:\n"
          f"[auth] {API_KEY}\n"
          f"[auth] Set APP_API_KEY in your .env to keep a stable key across restarts.\n")


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Defence Intelligence & Decision Support API",
    description="Multi-agent RAG system for defence/geopolitical evidence analysis.",
    version="1.0.0",
)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
# Allow the plain HTML/JS frontend (served separately, e.g. via file:// or
# a simple static server on a different port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to a specific origin before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class AssessRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness check -- no auth, so uptime monitors / load balancers can hit it freely."""
    return {"status": "ok"}


@app.post("/assess")
def assess(request: AssessRequest, x_api_key: str | None = Header(default=None)):
    """
    Run the full Retrieval -> Analysis -> Verification -> Risk -> Report
    -> Memory workflow for a single question and return the final report.
    """
    verify_api_key(x_api_key)

    session_id = request.session_id or str(uuid.uuid4())

    try:
        final_state = run_workflow(query=request.query, session_id=session_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Workflow failed: {exc}") from exc

    report = final_state.get("final_report")
    if report is None:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Workflow completed without producing a report.",
                "errors": final_state.get("errors", []),
            },
        )

    return {
        "session_id": session_id,
        "report": report.model_dump(mode="json"),
        "errors": final_state.get("errors", []),
    }
