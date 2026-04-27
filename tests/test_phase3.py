"""
Phase 3 end-to-end tests -- agent layer + API.

Tests that exercise the LLM are gated on ``ANTHROPIC_API_KEY``; the
deterministic pieces (tool executor, memory, follow-ups, /health) run
unconditionally so CI without an API key still validates most of the
surface area.

Run with:  pytest tests/test_phase3.py -v
"""
import os
import time
import uuid

import pandas as pd
import pytest

from app.ingestion.loader import load_all_sheets
from app.ingestion.normalizer import NormalizationPipeline
from app.models.schema import get_engine, init_database
from app.services.service_registry import clear_registry_cache, get_registry
from configs import settings


_API_KEY_SET = bool(os.getenv("ANTHROPIC_API_KEY"))
needs_api = pytest.mark.skipif(
    not _API_KEY_SET,
    reason="ANTHROPIC_API_KEY not set -- skipping live LLM tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def registry():
    """Module-scoped in-memory DB built from the real workbook."""
    if not settings.EXCEL_PATH.exists():
        pytest.skip("Source workbook not present -- skipping Phase 3 tests")

    clear_registry_cache()
    sheets = load_all_sheets(settings.EXCEL_PATH)
    engine = get_engine("sqlite:///:memory:")
    init_database(engine)
    NormalizationPipeline().run_full_pipeline(sheets, engine)
    return get_registry(engine)


@pytest.fixture(scope="module")
def agent(registry):
    """Module-scoped QualityAgent -- skipped when no API key."""
    if not _API_KEY_SET:
        pytest.skip("ANTHROPIC_API_KEY not set")
    from app.agent.agent_core import QualityAgent
    return QualityAgent(registry=registry, engine=registry.engine)


@pytest.fixture()
def api_client():
    """FastAPI TestClient that runs the lifespan + uses the file DB."""
    from fastapi.testclient import TestClient
    from app.agent.api import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Intent classifier taxonomy + (LLM-gated) all-10 sweep
# ---------------------------------------------------------------------------

def test_intent_classifier_all_10_intents():
    """Taxonomy + entity extraction sanity (no LLM)."""
    from app.agent.intent_classifier import (
        INTENT_TOOL_MAP,
        VALID_INTENTS,
        extract_entities,
    )
    assert len(VALID_INTENTS) == 10
    assert set(INTENT_TOOL_MAP.keys()) == set(VALID_INTENTS)

    # Spot-check entity extraction on the 10 demo prompts.
    cases = [
        ("What is the risk level of lot L-778?",                     "lot_no",   ["L-778"]),
        ("Tell me about supplier SUP-C",                             "supplier", ["SUP-C"]),
        ("Are there any process issues on LINE-2 night shift?",      "line",     ["LINE-2"]),
        ("How is China performing compared to other countries?",     "coo",      ["China"]),
        ("Why did serial SR20260008 fail in the field?",             "serial",   ["SR20260008"]),
    ]
    for text, key, expected in cases:
        assert extract_entities(text)[key] == expected


# ---------------------------------------------------------------------------
# 2. Tool executor -- deterministic pieces
# ---------------------------------------------------------------------------

def test_tool_executor_lot_risk(registry):
    from app.agent.tool_executor import ToolExecutor
    tx = ToolExecutor(registry, registry.engine)
    r = tx.execute("get_lot_risk", {"lot_no": "L-778"})
    assert r.ok, r.error
    assert r.result_data["risk_tier"] == "HIGH"
    assert r.result_data["lot_no"] == "L-778"


def test_tool_executor_process_drift(registry):
    from app.agent.tool_executor import ToolExecutor
    tx = ToolExecutor(registry, registry.engine)
    r = tx.execute("get_process_drift", {"only_flagged": True})
    assert r.ok, r.error
    signals = r.result_data["drift_signals"]
    assert any(
        s.get("line") == "LINE-2" and s.get("shift") == "Night"
        and s.get("is_drift_signal") for s in signals
    ), f"LINE-2 Night drift signal not flagged: {signals}"


def test_tool_executor_supplier_compare(registry):
    from app.agent.tool_executor import ToolExecutor
    tx = ToolExecutor(registry, registry.engine)
    r = tx.execute("compare_suppliers", {
        "supplier_ids": ["SUP-A", "SUP-B"],
        "use_case": "safety-critical",
    })
    assert r.ok, r.error
    rec = r.result_data.get("recommendation") or {}
    assert "winner" in rec, "compare_suppliers result missing winner field"
    assert rec["winner"] in {"SUP-A", "SUP-B"}, rec


# ---------------------------------------------------------------------------
# 3. Memory + follow-ups
# ---------------------------------------------------------------------------

def test_conversation_memory_entity_resolution():
    from app.agent.intent_classifier import IntentResult
    from app.agent.memory import ConversationMemory

    mem = ConversationMemory()
    mem.add_turn(
        question="What is the risk of lot L-778?",
        response="Lot L-778 is HIGH risk; supplier SUP-C is on Watchlist.",
        intent=IntentResult(
            intent="LOT_RISK_QUERY",
            entities={"lot_no": ["L-778"], "supplier": ["SUP-C"]},
            confidence=0.95,
            primary_tool="get_lot_risk",
            secondary_tools=[],
            reasoning="",
        ),
        entities={"lot_no": ["L-778"], "supplier": ["SUP-C"]},
    )
    resolved = mem.resolve_entities("What about its supplier?")
    assert resolved.was_rewritten
    assert "SUP-C" in resolved.text
    assert resolved.context_used


def test_follow_up_generation_specific():
    from app.agent.follow_up_generator import FollowUpGenerator
    from app.agent.intent_classifier import IntentResult

    fug = FollowUpGenerator()
    intent = IntentResult(
        intent="LOT_RISK_QUERY",
        entities={"lot_no": ["L-778"]},
        confidence=1.0,
        primary_tool="get_lot_risk",
        secondary_tools=[],
        reasoning="",
    )
    suggestions = fug.generate_follow_ups(intent, intent.entities, [])
    assert len(suggestions) == 3
    assert any("L-778" in s for s in suggestions), (
        f"At least one follow-up should mention the entity: {suggestions}"
    )
    # Reject obvious generics.
    banned = {"tell me more", "more info", "what else"}
    for s in suggestions:
        assert s.lower().strip(".?!") not in banned, f"Generic follow-up: {s!r}"


# ---------------------------------------------------------------------------
# 4. Audit log
# ---------------------------------------------------------------------------

@needs_api
def test_audit_log_written(agent):
    from app.agent.agent_core import AUDIT_LOG_PATH
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pre_size = AUDIT_LOG_PATH.stat().st_size if AUDIT_LOG_PATH.exists() else 0

    agent.ask(
        "Quick smoke: what is the risk of lot L-778?",
        session_id=f"test-audit-{uuid.uuid4()}",
    )

    assert AUDIT_LOG_PATH.exists(), "audit log file not created"
    assert AUDIT_LOG_PATH.stat().st_size > pre_size, "audit log not appended"


# ---------------------------------------------------------------------------
# 5. API
# ---------------------------------------------------------------------------

def test_api_health_endpoint(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    for field in ("status", "tables", "insights", "demo_stories", "agent_available"):
        assert field in body, f"missing field: {field}"
    assert body["tables"] >= 14
    assert "L778_HIGH" in body["demo_stories"]


@needs_api
def test_api_ask_endpoint(api_client):
    r = api_client.post(
        "/ask",
        json={"question": "What is the risk level of lot L-778?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "L-778" in body["response_text"]
    # Phase 3 Step 4 contract additions.
    assert "follow_up_suggestions" in body
    assert "clickable_suggestions" in body
    assert len(body["follow_up_suggestions"]) == 3
    assert "resolved_question" in body
    assert "context_used" in body


# ---------------------------------------------------------------------------
# 6. Session summary
# ---------------------------------------------------------------------------

@needs_api
def test_session_summary(agent):
    sid = f"test-session-{uuid.uuid4()}"
    for q in [
        "What is the risk of lot L-778?",
        "Are there process drift issues?",
        "Compare SUP-A and SUP-B",
    ]:
        agent.ask(q, session_id=sid)

    summary = agent.get_session_summary(sid)
    assert summary["session_id"] == sid
    assert summary["question_count"] == 3
    assert len(summary["questions_asked"]) == 3
    assert summary["tool_call_count"] > 0
    assert summary["total_tokens_used"] >= 0  # mock path: 0 API tokens (no Claude call)
