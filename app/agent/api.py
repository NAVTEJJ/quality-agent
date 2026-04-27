"""
FastAPI facade for the Quality Agent (Phase 3, Step 3).

Endpoints
---------
* ``POST /ask``                          -- run the full agentic loop.
* ``GET  /suggested-questions``          -- 5 demo-ready showcase prompts.
* ``GET  /health``                       -- liveness + demo-story status.
* ``GET  /session/{session_id}/summary`` -- rollup of a chat session.
* ``GET  /lot/{lot_no}/risk``            -- direct KPI lookup (no LLM).
* ``GET  /supplier/{supplier_id}/profile`` -- direct supplier profile.
* ``GET  /process-drift``                -- all drift signals.

Design
------
* The FastAPI lifespan handler wires the engine, registry, and agent onto
  ``app.state`` exactly once per process.
* Direct endpoints (``/lot/...``, ``/supplier/...``, ``/process-drift``)
  bypass the LLM so the dashboard can render fast, deterministic data
  even when no ``ANTHROPIC_API_KEY`` is present.
* The ``/ask`` endpoint returns HTTP 503 if the agent could not be built
  (missing API key), so the UI can detect this cleanly.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent.agent_core import QualityAgent
from app.agent.tool_executor import ToolExecutor
from app.models.schema import get_engine
from app.services.service_registry import clear_registry_cache, get_registry
from configs import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question.")
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional session id. Used to scope the audit log and allow "
            "get_session_summary to roll up multiple turns."
        ),
    )


# ---------------------------------------------------------------------------
# Lifespan -- initialise engine, registry, agent; validate demo stories
# ---------------------------------------------------------------------------

_DEMO_STORY_L778 = "L-778 HIGH risk"
_DEMO_STORY_LINE2 = "LINE-2 Night drift"
_DEMO_STORY_PREMIUM = "Premium suppliers identified"


def _validate_demo_stories(registry) -> Dict[str, bool]:
    """Quickly reassert the three Phase 2 marquee stories are still green."""
    results: Dict[str, bool] = {}

    try:
        risk = registry.kpi.get_lot_risk_scores()
        l778 = risk[risk["lot_no"] == "L-778"]
        results[_DEMO_STORY_L778] = (
            not l778.empty and l778.iloc[0]["risk_tier"] == "HIGH"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("L-778 check failed: %s", exc)
        results[_DEMO_STORY_L778] = False

    try:
        drift = registry.kpi.get_process_drift_by_line_shift()
        line2 = drift[(drift["line"] == "LINE-2") & (drift["shift"] == "Night")]
        results[_DEMO_STORY_LINE2] = (
            not line2.empty and float(line2.iloc[0]["torque_fail_rate"]) > 0.1
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LINE-2 Night check failed: %s", exc)
        results[_DEMO_STORY_LINE2] = False

    try:
        premium = registry.kpi.get_premium_suppliers()
        results[_DEMO_STORY_PREMIUM] = len(premium) >= 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Premium supplier check failed: %s", exc)
        results[_DEMO_STORY_PREMIUM] = False

    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the agent on startup, tear nothing down on shutdown."""
    logger.info("Quality Agent API -- starting ...")

    clear_registry_cache()
    app.state.engine = get_engine()
    app.state.registry = get_registry(app.state.engine)
    app.state.tool_executor = ToolExecutor(app.state.registry, app.state.engine)

    # Agent (LLM layer) is optional -- fall back to direct endpoints only
    # if the API key is missing, so the UI still works for read-only data.
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            app.state.agent = QualityAgent(
                registry=app.state.registry,
                engine=app.state.engine,
            )
            logger.info("QualityAgent initialised with model=%s", app.state.agent.model)
        except Exception as exc:  # noqa: BLE001
            logger.error("QualityAgent init failed: %s", exc)
            app.state.agent = None
    else:
        logger.warning(
            "ANTHROPIC_API_KEY not set -- /ask disabled; direct endpoints still served."
        )
        app.state.agent = None

    app.state.demo_stories = _validate_demo_stories(app.state.registry)
    logger.info("Demo stories: %s", app.state.demo_stories)

    if all(app.state.demo_stories.values()):
        logger.info("Quality Agent API ready -- all systems operational")
    else:
        failed = [k for k, v in app.state.demo_stories.items() if not v]
        logger.error("Quality Agent API started with FAILED demo stories: %s", failed)

    yield
    logger.info("Quality Agent API -- shutting down")


# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Quality Agent API",
    version="1.0",
    description=(
        "AI Quality Management Copilot for mechanical / automotive "
        "manufacturing. Ask natural-language questions and receive "
        "evidence-linked insights, recommended actions, and SAP touchpoints."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One-line access log with request duration."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "http  %s %s -> %d  (%.1fms)",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


# ---------------------------------------------------------------------------
# Health + metadata
# ---------------------------------------------------------------------------

@app.get("/health")
def health(request: Request) -> Dict[str, Any]:
    """Returns service status, table count, insight count, and demo stories."""
    registry = request.app.state.registry
    engine = request.app.state.engine

    # Table count -- one query to sqlite_master (or pg_catalog in future).
    from sqlalchemy import inspect
    insp = inspect(engine)
    table_count = len(insp.get_table_names())

    # Insight count -- use the on-disk artefact to avoid reruns.
    insights_path = settings.PROCESSED_DIR / "insights.json"
    insight_count = 0
    if insights_path.exists():
        try:
            import json
            payload = json.loads(insights_path.read_text(encoding="utf-8"))
            insight_count = int(payload.get("count", 0))
        except Exception:  # noqa: BLE001
            insight_count = 0

    demo_stories = getattr(request.app.state, "demo_stories", {})
    demo_story_codes = []
    if demo_stories.get(_DEMO_STORY_L778):
        demo_story_codes.append("L778_HIGH")
    if demo_stories.get(_DEMO_STORY_LINE2):
        demo_story_codes.append("LINE2_NIGHT_CONFIRMED")
    if demo_stories.get(_DEMO_STORY_PREMIUM):
        demo_story_codes.append("PREMIUM_SUPPLIERS")

    return {
        "status":       "ok" if demo_story_codes else "degraded",
        "tables":       table_count,
        "insights":     insight_count,
        "demo_stories": demo_story_codes,
        "agent_available": request.app.state.agent is not None,
    }


@app.get("/suggested-questions")
def suggested_questions(request: Request) -> List[str]:
    """The five showcase demo prompts."""
    agent = request.app.state.agent
    if agent is not None:
        return agent.get_suggested_questions()
    # Fallback when the LLM agent is disabled -- static list.
    return [
        "What is the risk level of lot L-778 and what actions should I take?",
        "Are there any process drift issues on our production lines?",
        "Compare SUP-A and SUP-B for a new safety-critical program",
        "Why did serial SR20260008 fail in the field?",
        "Where should I focus incoming inspection effort this week?",
    ]


# ---------------------------------------------------------------------------
# Conversational endpoints
# ---------------------------------------------------------------------------

@app.post("/ask")
def ask(payload: AskRequest, request: Request) -> Dict[str, Any]:
    """Run the full agentic loop for a user question."""
    agent: Optional[QualityAgent] = request.app.state.agent
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "QualityAgent is not initialised -- set ANTHROPIC_API_KEY "
                "and restart the server. Direct endpoints (/lot, /supplier, "
                "/process-drift) still work without a key."
            ),
        )
    try:
        response = agent.ask(payload.question, session_id=payload.session_id)
        return response.to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.exception("ask() failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/session/{session_id}/summary")
def session_summary(session_id: str, request: Request) -> Dict[str, Any]:
    agent: Optional[QualityAgent] = request.app.state.agent
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="QualityAgent not initialised -- no session data available.",
        )
    return agent.get_session_summary(session_id)


# ---------------------------------------------------------------------------
# Direct endpoints (no LLM)
# ---------------------------------------------------------------------------

@app.get("/lot/{lot_no}/risk")
def lot_risk(lot_no: str, request: Request) -> Dict[str, Any]:
    """Direct lot-risk payload -- bypasses the agent loop."""
    tx: ToolExecutor = request.app.state.tool_executor
    result = tx.execute("get_lot_risk", {"lot_no": lot_no, "include_warranty": True})
    if result.error:
        raise HTTPException(status_code=500, detail=result.error)
    if not result.result_data.get("found", True):
        raise HTTPException(
            status_code=404,
            detail=f"Lot {lot_no!r} not found in lot_risk_scores.",
        )
    return result.to_dict()


@app.get("/supplier/{supplier_id}/profile")
def supplier_profile(supplier_id: str, request: Request) -> Dict[str, Any]:
    """Direct supplier-profile payload."""
    tx: ToolExecutor = request.app.state.tool_executor
    result = tx.execute(
        "get_supplier_profile",
        {"supplier_id": supplier_id, "include_coo_decomposition": True},
    )
    if result.error:
        raise HTTPException(status_code=500, detail=result.error)
    if not result.result_data.get("found", True):
        raise HTTPException(
            status_code=404,
            detail=f"Supplier {supplier_id!r} not found.",
        )
    return result.to_dict()


@app.get("/process-drift")
def process_drift(
    request: Request,
    line: Optional[str] = None,
    shift: Optional[str] = None,
    only_flagged: bool = False,
) -> Dict[str, Any]:
    """Direct drift-signals payload (optionally filtered to flagged rows)."""
    tx: ToolExecutor = request.app.state.tool_executor
    result = tx.execute(
        "get_process_drift",
        {"line": line, "shift": shift, "only_flagged": only_flagged},
    )
    if result.error:
        raise HTTPException(status_code=500, detail=result.error)
    return result.to_dict()


__all__ = ["app"]
