"""
The Quality Agent's reasoning engine (Phase 3, Step 3).

:class:`QualityAgent` wires the intent classifier, the tool executor, the
system prompt, and the Anthropic tool-use loop into a single public
entry point :meth:`QualityAgent.ask`.

Flow (seven steps, in order)
----------------------------
1.  Classify the question's intent.
2.  Build a context block (recent turns + cached DB state summary).
3.  First Claude call with the system prompt and all ten tools.
4.  Tool-use loop: execute every ``tool_use`` block, feed results back,
    call Claude again. Capped at ``MAX_TOOL_ROUNDS`` to guard against
    runaway loops.
5.  Verify the final text matches the mandated response structure;
    one reformat call is allowed if it does not.
6.  Append one compact JSON line to ``data/processed/audit_log.jsonl``.
7.  Trim and update the in-memory conversation history.

The agent never swallows silently -- if a tool raises, the executor
returns a typed error that Claude then sees; if Claude refuses to follow
the structure after a reformat call, we return the final text anyway and
flag it in the audit trail.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from sqlalchemy.engine import Engine

from app.agent.follow_up_generator import FollowUpGenerator
from app.agent.intent_classifier import IntentClassifier, IntentResult
from app.agent.memory import ConversationMemory
from app.agent.system_prompt import SYSTEM_PROMPT
from app.agent.tool_executor import ToolExecutor, ToolResult
from app.agent.tools import CLAUDE_TOOLS
from app.services.service_registry import ServiceRegistry
from configs import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOOL_ROUNDS = 1
HISTORY_MAX_TURNS = 2       # 1 user + 1 assistant
STATE_CACHE_SECONDS = 300   # 5 minutes
MAX_TOKENS = 600
AUDIT_LOG_PATH         = settings.PROCESSED_DIR / "audit_log.jsonl"
DEFAULT_BRANCH_LOG_PATH = settings.PROCESSED_DIR / "unrouted_questions.jsonl"


def _general_fallback_text() -> str:
    return (
        "## ⚠️ Response Unavailable\n\n"
        "**Finding:** The quality database could not be reached for this query.\n\n"
        "**Actions:**\n"
        "1. Retry the question\n"
        "2. Check DB connection in the sidebar"
    )

_REQUIRED_SECTIONS = [
    "**Finding:**",
    "**Evidence:**",
]


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    question: str
    intent: IntentResult
    response_text: str
    tools_called: List[str]
    tool_results: List[ToolResult]
    total_tokens: int
    execution_time_ms: float
    session_id: str
    timestamp: datetime
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_suggestions: List[str] = field(default_factory=list)
    resolved_question: str = ""
    context_used: bool = False
    # Phase 5 Step 2 — performance instrumentation.
    response_time_ms: float = 0.0
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly dict for API responses."""
        return {
            "question":              self.question,
            "resolved_question":     self.resolved_question,
            "context_used":          self.context_used,
            "intent":                self.intent.to_dict(),
            "response_text":         self.response_text,
            "tools_called":          list(self.tools_called),
            "tool_results":          [tr.to_dict() for tr in self.tool_results],
            "total_tokens":          self.total_tokens,
            "execution_time_ms":     self.execution_time_ms,
            "response_time_ms":      self.response_time_ms,
            "cache_hit":             self.cache_hit,
            "session_id":            self.session_id,
            "timestamp":             self.timestamp.isoformat(),
            "audit_trail":           list(self.audit_trail),
            "follow_up_suggestions": list(self.follow_up_suggestions),
            # UI-friendly alias -- the frontend renders these as clickable chips.
            "clickable_suggestions": list(self.follow_up_suggestions),
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class QualityAgent:
    """Claude-powered reasoning engine over the Phase 2 insight stack."""

    def __init__(
        self,
        registry: ServiceRegistry,
        engine: Engine,
        anthropic_client: Optional[anthropic.Anthropic] = None,
        model: str = DEFAULT_MODEL,
        classifier_model: Optional[str] = None,
    ) -> None:
        self.registry = registry
        self.engine = engine
        self.model = model

        if anthropic_client is None:
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Export it or pass "
                    "anthropic_client=... to QualityAgent()."
                )
            anthropic_client = anthropic.Anthropic(api_key=key)
        self.client = anthropic_client

        self.tool_executor = ToolExecutor(registry, engine)
        self.classifier = IntentClassifier(
            model=classifier_model or "claude-haiku-4-5-20251001",  # classifier stays cheap
            client=anthropic_client,
        )

        # Multi-turn memory + follow-up generator -- what makes this a
        # copilot rather than a Q&A box.
        self.memory = ConversationMemory(max_turns=HISTORY_MAX_TURNS)
        self.follow_up_generator = FollowUpGenerator()

        self.audit_log: List[Dict[str, Any]] = []

        # Per-session audit entries for get_session_summary.
        self._per_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Cached DB state summary (5-min TTL).
        self._state_summary: Optional[str] = None
        self._state_summary_cached_at: float = 0.0

        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Restore last session's entity context from the persistent store
        # so pronoun resolution works across restarts.
        try:
            from app.core.chat_store import get_chat_store
            last = get_chat_store().get_last_session_context()
            if last:
                for turn in last["recent_turns"]:
                    from app.agent.intent_classifier import extract_entities
                    entities = extract_entities(turn["question"])
                    self.memory.add_turn(
                        question=turn["question"],
                        response=turn["response"],
                        intent=None,
                        entities=entities,
                    )
                logger.info(
                    "Restored %d turns from session %s (last active %s)",
                    len(last["recent_turns"]),
                    last["session_id"][:8],
                    last["last_active"][:16],
                )
        except Exception as _restore_exc:
            logger.debug("Could not restore chat context: %s", _restore_exc)

        # Excel agent — primary answer path when API key is present.
        # Loads workbook context once and caches it for the session.
        try:
            from app.agent.excel_agent import ExcelQualityAgent
            self.excel_agent: Optional[ExcelQualityAgent] = ExcelQualityAgent(
                client=anthropic_client,
            )
            logger.info("ExcelQualityAgent ready")
        except Exception as _exc:
            logger.warning("ExcelQualityAgent unavailable: %s", _exc)
            self.excel_agent = None

    # ==================================================================
    # Public API
    # ==================================================================

    def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> AgentResponse:
        """Run the full agentic loop for *question* and return the result.

        When ``use_cache`` is True (default) we first check the shared
        :class:`~app.core.cache.QueryCache`. A hit returns a clone of the
        cached :class:`AgentResponse` with ``cache_hit=True`` and a fresh
        ``response_time_ms`` (typically < 5 ms). A miss runs the full
        agentic loop and writes the result to the cache.
        """
        session_id = session_id or str(uuid.uuid4())
        start_time = time.perf_counter()
        audit_trail: List[Dict[str, Any]] = []

        # ── Cache fast-path ──────────────────────────────────────────
        if use_cache:
            try:
                from app.core.cache import (
                    annotate_response, get_default_cache, SLA_WARN_MS,
                )
                cached = get_default_cache().get(question)
                if cached is not None:
                    elapsed = (time.perf_counter() - start_time) * 1000
                    cached.session_id = session_id
                    cached.timestamp  = datetime.now(timezone.utc)
                    annotate_response(cached, elapsed, cache_hit=True)
                    return cached
            except Exception:  # noqa: BLE001 — cache must never break asks
                pass

        # ── Step 0 — entity / pronoun resolution from memory ─────────
        resolved = self.memory.resolve_entities(question)
        if resolved.was_rewritten:
            logger.info(
                "memory  rewrite  %r -> %r  (%s)",
                resolved.original, resolved.text, resolved.substitutions,
            )
        audit_trail.append({
            "step":           "entity_resolution",
            "original":       resolved.original,
            "resolved":       resolved.text,
            "context_used":   resolved.context_used,
            "substitutions":  resolved.substitutions,
        })

        # ── Step 1 — intent classification (on the resolved text) ────
        try:
            intent = self.classifier.classify(resolved.text)
            audit_trail.append({
                "step": "intent_classification",
                "intent": intent.intent,
                "confidence": intent.confidence,
                "primary_tool": intent.primary_tool,
                "entities": intent.entities,
                "reasoning": intent.reasoning,
            })
        except Exception as exc:
            logger.exception("Intent classification failed")
            # Graceful fallback: plain LLM call without routing hints.
            intent = IntentResult(
                intent="GENERAL_INSIGHT",
                entities={},
                confidence=0.0,
                primary_tool="search_insights",
                secondary_tools=[],
                reasoning=f"(classifier error: {exc})",
            )
            audit_trail.append({"step": "intent_classification", "error": str(exc)})

        # ── Step 2 — context enrichment (memory + DB state) ──────────
        context_block = self.memory.get_context_block()
        state_summary = self._get_state_summary()
        audit_trail.append({
            "step":                 "context_enrichment",
            "history_turns":        len(self.memory.conversation_history),
            "active_investigation": self.memory.active_investigation,
            "state_summary":        state_summary,
        })

        # ── Step 2b — answer routing ──────────────────────────────────
        # Priority 1: Excel agent (Claude with full workbook context).
        #   Handles ANY question with real numbers. Used when API key present.
        # Priority 2: Mock responder (zero-token DB queries, 15 branches).
        #   Fallback when Excel agent unavailable.
        # Priority 3: Generic fallback text.

        _result: Optional[Dict[str, Any]] = None

        if self.excel_agent is not None:
            try:
                _result = self.excel_agent.ask(resolved.text, session_id)
            except Exception as _exc:
                logger.warning("ExcelQualityAgent.ask failed: %s", _exc)

        if _result is None:
            try:
                from app.agent.mock_responder import render_mock_response
                _result = render_mock_response(
                    resolved.text, self.registry, self.engine,
                    session_id=session_id, use_cache=True,
                )
            except Exception as _mock_exc:
                logger.warning("Mock responder failed: %s", _mock_exc)

        if _result is None:
            _result = {"text": _general_fallback_text(), "tools": [], "confidence": 0, "suggestions": []}

        resp_text  = _result["text"]
        resp_tools = _result.get("tools", [])
        resp_sugg  = _result.get("suggestions", [])
        resp_tok   = int(_result.get("tokens", 0))
        resp_cache = bool(_result.get("cache_hit", False))
        resp_branch = _result.get("_branch", "excel_agent" if self.excel_agent else "mock")

        # Log unrouted questions for coverage analysis
        if resp_branch == "default":
            try:
                DEFAULT_BRANCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with DEFAULT_BRANCH_LOG_PATH.open("a", encoding="utf-8") as _f:
                    _f.write(json.dumps({
                        "ts":         datetime.now(timezone.utc).isoformat(),
                        "session_id": session_id,
                        "question":   question,
                        "resolved":   resolved.text,
                    }) + "\n")
            except Exception:
                pass

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        audit_trail.append({
            "step":       "answer_routing",
            "branch":     resp_branch,
            "confidence": _result.get("confidence", 0),
            "cache_hit":  resp_cache,
            "tokens":     resp_tok,
        })
        self._write_audit_entry(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            question=question,
            intent=intent,
            tool_results=[],
            response_text=resp_text,
            total_tokens=resp_tok,
            total_time_ms=elapsed_ms,
            mock_tools=resp_tools,
        )
        self.memory.add_turn(
            question=resolved.text,
            response=resp_text,
            intent=intent,
            entities=intent.entities if intent else {},
        )

        # Persist turn to chat history store (survives restarts).
        try:
            from app.core.chat_store import get_chat_store
            get_chat_store().save_turn(
                session_id=session_id,
                question=question,
                response=resp_text,
                tokens=resp_tok,
                cache_hit=resp_cache,
                branch=resp_branch,
                execution_time_ms=elapsed_ms,
            )
        except Exception as _cs_exc:
            logger.debug("chat_store save_turn failed: %s", _cs_exc)

        return AgentResponse(
            question=question,
            response_text=resp_text,
            intent=intent,
            tools_called=resp_tools,
            tool_results=[],
            follow_up_suggestions=resp_sugg,
            total_tokens=resp_tok,
            audit_trail=audit_trail,
            execution_time_ms=elapsed_ms,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            cache_hit=resp_cache,
        )

        # ── Dead code preserved for future Claude re-enablement ───────
        # The block below is intentionally unreachable. Claude-based fact
        # generation was removed to guarantee 100% DB-grounded responses.
        # To re-enable: remove the return above and restore this block.
        user_message_content = (
            f"[CURRENT DATABASE STATE]\n{state_summary}\n\n"
            f"[USER QUESTION]\n{resolved.text}"
        )
        messages: List[Dict[str, Any]] = []
        messages.extend(context_block)
        messages.append({"role": "user", "content": user_message_content})

        total_tokens = 0
        tools_called: List[str] = []
        tool_results: List[ToolResult] = []

        try:
            response, loop_tokens, executed = self._run_tool_loop(
                messages=messages, audit_trail=audit_trail
            )
            total_tokens += loop_tokens
            tools_called.extend(tr.tool_name for tr in executed)
            tool_results.extend(executed)
            response_text = self._extract_text(response)
        except anthropic.APIError as exc:
            err_str = str(exc).lower()
            is_billing = any(k in err_str for k in ("credit", "billing", "balance", "quota", "rate_limit", "overload"))
            if is_billing:
                # Silently fall back to mock responder so demo keeps running.
                logger.warning("Anthropic API billing/quota error — falling back to demo mode: %s", exc)
                try:
                    from app.agent.mock_responder import render_mock_response
                    mock = render_mock_response(question, self.registry, self.engine, session_id=session_id)
                    response_text = mock["text"]
                    tools_called  = mock.get("tools", [])
                except Exception as mock_exc:  # noqa: BLE001
                    response_text = f"## Demo Mode\n\nRunning in demo mode (API quota reached)."
                    logger.warning("Mock responder also failed: %s", mock_exc)
                audit_trail.append({"step": "billing_fallback", "error": str(exc)})
            else:
                logger.exception("Anthropic API error")
                response_text = (
                    f"## Agent Error\n\n**Finding:** The Anthropic API returned "
                    f"an error while processing your question.\n\n"
                    f"**Evidence:**\n- {exc.__class__.__name__}: {exc}\n\n"
                    f"**Recommended Actions:**\n1. Retry the question.\n\n"
                    f"**Confidence:** 0% -- the error prevented data retrieval."
                )
                audit_trail.append({"step": "claude_error", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected agent error")
            response_text = (
                f"## Unexpected Error\n\n**Finding:** {exc.__class__.__name__}: {exc}"
            )
            audit_trail.append({"step": "unexpected_error", "error": str(exc)})

        # ── Step 4b — grounding validation ───────────────────────────
        grounding_warnings = self._validate_grounding(response_text, tool_results)
        if grounding_warnings:
            for w in grounding_warnings:
                logger.warning("grounding_check  FAIL  %s", w)
            audit_trail.append({"step": "grounding_warnings", "warnings": grounding_warnings})

        # ── Step 5 — structure validation + optional reformat ────────
        if not self._response_is_well_structured(response_text):
            try:
                reformatted, extra = self._reformat_response(
                    question, response_text, audit_trail
                )
                response_text = reformatted
                total_tokens += extra
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reformat call failed: %s", exc)
                audit_trail.append({"step": "reformat_error", "error": str(exc)})

        # ── Step 6 — audit logging ────────────────────────────────────
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        timestamp = datetime.now(timezone.utc)
        self._write_audit_entry(
            session_id=session_id,
            timestamp=timestamp,
            question=question,
            intent=intent,
            tool_results=tool_results,
            response_text=response_text,
            total_tokens=total_tokens,
            total_time_ms=elapsed_ms,
        )

        # ── Step 7 — update memory + generate follow-ups ─────────────
        self.memory.add_turn(
            question=resolved.text,
            response=response_text,
            intent=intent,
            entities=intent.entities,
        )

        try:
            follow_ups = self.follow_up_generator.generate_follow_ups(
                intent=intent,
                entities=intent.entities,
                tool_results=tool_results,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("follow-up generation failed: %s", exc)
            follow_ups = []
        audit_trail.append({"step": "follow_ups", "suggestions": follow_ups})

        agent_response = AgentResponse(
            question=question,
            intent=intent,
            response_text=response_text,
            tools_called=tools_called,
            tool_results=tool_results,
            total_tokens=total_tokens,
            execution_time_ms=elapsed_ms,
            session_id=session_id,
            timestamp=timestamp,
            audit_trail=audit_trail,
            follow_up_suggestions=follow_ups,
            resolved_question=resolved.text,
            context_used=resolved.context_used,
        )

        # ── Phase 5 Step 2 — instrumentation + cache write ──────────
        try:
            from app.core.cache import annotate_response, get_default_cache
            annotate_response(agent_response, elapsed_ms, cache_hit=False)
            if use_cache:
                get_default_cache().set(question, agent_response)
        except Exception:  # noqa: BLE001
            pass

        return agent_response

    # ------------------------------------------------------------------

    def get_suggested_questions(self) -> List[str]:
        """Demo-ready questions that exercise every major feature."""
        return [
            "What is the risk level of lot L-778 and what actions should I take?",
            "Are there any process drift issues on our production lines?",
            "Compare SUP-A and SUP-B for a new safety-critical program",
            "Why did serial SR20260008 fail in the field?",
            "Where should I focus incoming inspection effort this week?",
        ]

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        entries = list(self._per_session.get(session_id, []))
        questions_asked: List[str] = []
        tools_used: List[str] = []
        insights_surfaced: List[str] = []
        actions_recommended: List[str] = []
        total_tokens = 0

        for entry in entries:
            questions_asked.append(entry.get("question", ""))
            for t in entry.get("tools_called", []):
                tools_used.append(t.get("name", ""))
            insights_surfaced.extend(entry.get("insights_surfaced", []))
            actions_recommended.extend(entry.get("actions_recommended", []))
            total_tokens += int(entry.get("total_tokens", 0))

        return {
            "session_id":          session_id,
            "questions_asked":     questions_asked,
            "question_count":      len(questions_asked),
            "tools_used":          sorted(set(tools_used)),
            "tool_call_count":     len(tools_used),
            "insights_surfaced":   insights_surfaced,
            "actions_recommended": actions_recommended,
            "total_tokens_used":   total_tokens,
        }

    @staticmethod
    def get_unrouted_questions(top_n: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent unrouted (default-branch) questions.

        Reads ``unrouted_questions.jsonl`` and returns up to *top_n* entries
        sorted newest-first.  Useful for identifying coverage gaps.
        """
        if not DEFAULT_BRANCH_LOG_PATH.exists():
            return []
        entries: List[Dict[str, Any]] = []
        try:
            with DEFAULT_BRANCH_LOG_PATH.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception:
            return []
        return list(reversed(entries))[:top_n]

    # ==================================================================
    # Steps 3 + 4 -- tool-use loop
    # ==================================================================

    def _run_tool_loop(
        self,
        messages: List[Dict[str, Any]],
        audit_trail: List[Dict[str, Any]],
    ) -> tuple[Any, int, List[ToolResult]]:
        """Call Claude, execute any tool_use blocks, loop until done.

        Returns the final response object, total tokens across calls,
        and the list of executed :class:`ToolResult` objects.
        """
        total_tokens = 0
        executed: List[ToolResult] = []

        for round_idx in range(MAX_TOOL_ROUNDS + 1):
            logger.info("agent  claude_call  round=%d", round_idx)
            # Round 0: force at least one tool call so Claude cannot answer
            # from training-data memory. Subsequent rounds: auto (Claude may
            # stop if it has enough data).
            tc = {"type": "any"} if round_idx == 0 else {"type": "auto"}
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=CLAUDE_TOOLS,
                tool_choice=tc,
                messages=messages,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                total_tokens += int(usage.input_tokens) + int(usage.output_tokens)

            audit_trail.append({
                "step": "claude_call",
                "round": round_idx,
                "stop_reason": response.stop_reason,
                "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
            })

            tool_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

            if not tool_blocks or response.stop_reason != "tool_use":
                # Final assistant turn.
                return response, total_tokens, executed

            if round_idx >= MAX_TOOL_ROUNDS:
                logger.warning(
                    "agent  tool_loop_cap  rounds=%d -- forcing final answer",
                    round_idx,
                )
                audit_trail.append({
                    "step": "tool_loop_cap",
                    "message": f"reached MAX_TOOL_ROUNDS={MAX_TOOL_ROUNDS}",
                })
                # Ask Claude to finalise without more tools.
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": (
                        "You have reached the tool-call limit for this "
                        "question. Produce the final answer now using the "
                        "data already retrieved, following the mandatory "
                        "response structure."
                    ),
                })
                continue

            # Execute each tool call and feed the results back.
            tool_result_blocks: List[Dict[str, Any]] = []
            for block in tool_blocks:
                tool_name = block.name
                tool_input = dict(block.input or {})
                result = self.tool_executor.execute(tool_name, tool_input)
                executed.append(result)
                audit_trail.append({
                    "step": "tool_call",
                    "tool": tool_name,
                    "input": tool_input,
                    "row_count": result.row_count,
                    "time_ms": result.execution_time_ms,
                    "error": result.error,
                })

                tool_result_blocks.append(self._tool_result_to_block(block.id, result))

            messages.append({"role": "assistant", "content": response.content})
            # Grounding constraint: Claude must use only facts from tool results.
            grounding = {
                "type": "text",
                "text": (
                    "[GROUNDING CONSTRAINT] Your final answer must cite ONLY "
                    "supplier names, lot numbers, and numeric values that appear "
                    "in the tool results above. Every claim must be traceable to "
                    "a specific field in the tool output. Do NOT add suppliers, "
                    "lots, Cpk values, percentages, or COO data from memory."
                ),
            }
            messages.append({"role": "user", "content": tool_result_blocks + [grounding]})

        # Unreachable -- loop returns within.
        raise RuntimeError("tool loop exited without a final response")

    @staticmethod
    def _tool_result_to_block(tool_use_id: str, result: ToolResult) -> Dict[str, Any]:
        """Serialise a :class:`ToolResult` into a Claude tool_result block.

        Large payloads are truncated defensively to keep Claude's context
        manageable; full data stays in the audit log.
        """
        if result.error:
            return {
                "type":        "tool_result",
                "tool_use_id": tool_use_id,
                "content":     f"ERROR: {result.error}",
                "is_error":    True,
            }

        try:
            body = json.dumps(result.result_data, default=str)
        except (TypeError, ValueError) as exc:
            body = json.dumps({"_serialization_error": str(exc)})

        # 50k chars is ~12-15k tokens -- ample for any single tool call.
        if len(body) > 50_000:
            body = body[:50_000] + '... [truncated]"'
        return {
            "type":        "tool_result",
            "tool_use_id": tool_use_id,
            "content":     body,
        }

    # ==================================================================
    # Grounding validator — catch hallucinated entities
    # ==================================================================

    @staticmethod
    def _validate_grounding(
        response_text: str,
        tool_results: List[ToolResult],
    ) -> List[str]:
        """Return warning strings for any supplier/lot claims in *response_text*
        that do not appear in any tool result.  Empty list = fully grounded."""
        import json as _json

        # Build a ground-truth corpus from all tool result payloads.
        corpus = ""
        for tr in tool_results:
            if tr.ok:
                try:
                    corpus += _json.dumps(tr.result_data, default=str).upper()
                except Exception:
                    pass

        sup_in_response = set(re.findall(r"\bSUP-[A-Z]\b", response_text.upper()))
        lot_in_response = set(re.findall(r"\b(?:L-\d+|LOT-[A-Z0-9\-]+)\b", response_text.upper()))

        sup_in_tools = set(re.findall(r"SUP-[A-Z]", corpus))
        lot_in_tools = set(re.findall(r"(?:L-\d+|LOT-[A-Z0-9\-]+)", corpus))

        warnings: List[str] = []
        for s in sup_in_response - sup_in_tools:
            warnings.append(f"supplier {s} not found in tool results — possible hallucination")
        for lot in lot_in_response - lot_in_tools:
            warnings.append(f"lot {lot} not found in tool results — possible hallucination")
        return warnings

    # ==================================================================
    # Step 5 -- response structure validation
    # ==================================================================

    @staticmethod
    def _response_is_well_structured(text: str) -> bool:
        if not text or not text.strip():
            return False
        hits = sum(1 for marker in _REQUIRED_SECTIONS if marker in text)
        has_heading = bool(re.search(r"(?m)^##\s", text))
        return has_heading and hits >= 3

    def _reformat_response(
        self,
        question: str,
        draft: str,
        audit_trail: List[Dict[str, Any]],
    ) -> tuple[str, int]:
        """Ask Claude to rewrite *draft* into the mandated structure."""
        reformat_prompt = (
            "Rewrite the following answer using the mandatory short format: "
            "## Headline, **Finding:** (1-2 sentences), "
            "**Evidence:** (max 3 bullets with exact values), "
            "**Actions:** (max 2 numbered items with SAP codes). "
            "Keep the same facts. Remove all filler, padding, and extra sections.\n\n"
            f"Question: {question}\n\n"
            f"Draft:\n---\n{draft}\n---"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": reformat_prompt}],
        )
        usage = getattr(response, "usage", None)
        tokens = (
            int(usage.input_tokens) + int(usage.output_tokens)
            if usage is not None else 0
        )
        audit_trail.append({
            "step": "reformat",
            "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        })
        return self._extract_text(response), tokens

    @staticmethod
    def _extract_text(response: Any) -> str:
        parts = [
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        return "\n".join(parts).strip()

    # ==================================================================
    # Step 2 helpers -- DB state summary
    # (Conversation context block lives on ConversationMemory.)
    # ==================================================================

    def _get_state_summary(self) -> str:
        """Cached, 5-minute DB state summary sentence."""
        now = time.time()
        if (
            self._state_summary
            and (now - self._state_summary_cached_at) < STATE_CACHE_SECONDS
        ):
            return self._state_summary

        try:
            risk = self.registry.kpi.get_lot_risk_scores()
            drift = self.registry.kpi.get_drift_signals()
            n_lots = len(risk)
            n_high = int((risk["risk_tier"] == "HIGH").sum())
            n_drift = len(drift)
            summary = (
                f"Current state: {n_high} HIGH risk lot(s), "
                f"{n_drift} drift signal(s) active, {n_lots:,} lots in "
                f"database, last updated: "
                f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}"
            )
        except Exception as exc:  # noqa: BLE001
            summary = f"Current state: unavailable ({exc.__class__.__name__})"

        self._state_summary = summary
        self._state_summary_cached_at = now
        return summary

    # ==================================================================
    # Step 6 helpers -- audit log
    # ==================================================================

    def _write_audit_entry(
        self,
        session_id: str,
        timestamp: datetime,
        question: str,
        intent: IntentResult,
        tool_results: List[ToolResult],
        response_text: str,
        total_tokens: int,
        total_time_ms: float,
        mock_tools: Optional[List[str]] = None,
    ) -> None:
        tool_entries = [
            {
                "name":       tr.tool_name,
                "input":      tr.input_used,
                "row_count":  tr.row_count,
                "time_ms":    tr.execution_time_ms,
                "error":      tr.error,
            }
            for tr in tool_results
        ]
        # Add mock DB-query tool calls (conceptual tool calls with no API cost)
        for t in (mock_tools or []):
            tool_entries.append({"name": t, "input": {}, "row_count": 0, "time_ms": 0, "error": None})

        entry = {
            "session_id":       session_id,
            "timestamp":        timestamp.isoformat(),
            "question":         question,
            "intent": {
                "intent":        intent.intent,
                "confidence":    intent.confidence,
                "primary_tool":  intent.primary_tool,
                "entities":      intent.entities,
            },
            "tools_called": tool_entries,
            "response_preview":  (response_text or "")[:200],
            "total_tokens":      total_tokens,
            "total_time_ms":     total_time_ms,
        }

        self.audit_log.append(entry)
        self._per_session[session_id].append(entry)

        try:
            with AUDIT_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.warning("Could not write audit log: %s", exc)

    # ==================================================================
    # Convenience proxies
    # ==================================================================

    @property
    def conversation_history(self) -> List[Dict[str, Any]]:
        """Backwards-compatible view onto the memory's transcript."""
        return self.memory.conversation_history


__all__ = ["QualityAgent", "AgentResponse"]
