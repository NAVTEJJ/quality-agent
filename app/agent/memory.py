"""
Conversation memory for the Quality Agent (Phase 3, Step 4).

:class:`ConversationMemory` is what turns a stateless LLM call into a
multi-turn copilot: it keeps the last few turns, tracks which entity is
currently under investigation, and rewrites pronoun-laden follow-up
questions into fully-qualified questions before they hit the classifier.

Typical flow inside :class:`QualityAgent`::

    resolved = memory.resolve_entities(user_question)
    intent   = classifier.classify(resolved.text)
    ...
    memory.add_turn(user_question, response_text, intent, intent.entities)

Design notes
------------
* Memory is scoped to a single :class:`QualityAgent` instance -- the API
  layer creates one agent per process today; per-session memory is handled
  by the audit log's ``session_id`` partitioning.
* ``resolve_entities`` is deliberately conservative: it only rewrites when
  it is confident (explicit pronoun + known last entity, or a bare
  action phrase + an active investigation). A question that already
  names an entity is never rewritten.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.agent.intent_classifier import extract_entities

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolved-question envelope
# ---------------------------------------------------------------------------

@dataclass
class ResolvedQuestion:
    """Output of :meth:`ConversationMemory.resolve_entities`."""
    original: str
    text: str
    context_used: bool = False
    substitutions: List[str] = field(default_factory=list)

    @property
    def was_rewritten(self) -> bool:
        return self.text != self.original


# ---------------------------------------------------------------------------
# Pattern banks
# ---------------------------------------------------------------------------

# Pronouns the rewriter listens for when deciding whether to inject a
# remembered entity. Intentionally conservative -- false positives are
# much more costly than false negatives.
_LOT_PRONOUN_PATTERNS = [
    r"\bthis lot\b",
    r"\bthat lot\b",
    r"\bthe lot\b",
    r"\bthe same lot\b",
]

_SUPPLIER_PRONOUN_PATTERNS = [
    r"\bthis supplier\b",
    r"\bthat supplier\b",
    r"\bthe supplier\b",
    r"\bsame supplier\b",
    r"\bits supplier\b",
]

_LINE_PRONOUN_PATTERNS = [
    r"\bthis line\b",
    r"\bthat line\b",
    r"\bthe line\b",
    r"\bsame line\b",
]

# Phrases that indicate a bare follow-up with no entity of its own.
_BARE_FOLLOWUP_PATTERNS = [
    r"\bwhat (should|can|could|would|do) (i|we) (do|take|try)\b",
    r"\bwhat (are|is) the next step",
    r"\bwhat actions\b",
    r"\brecommend\b",
    r"\bsuggest(ions?)?\b",
    r"\bshow (me )?(more|full|all) details\b",
    r"\bshow (me )?details\b",
    r"\bdrill.?down\b",
    r"\btell me more\b",
    r"\bmore info\b",
    r"\bwhat else\b",
    r"\bnext\b",
    r"\banything else\b",
    r"\bwhy\??$",
]

# Tokens for "compare it with X" style rewrites -- only apply when there
# is clearly a second entity in play but the first is missing.
_COMPARISON_TOKENS = ("compare", " vs ", " vs.", "versus")


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------

class ConversationMemory:
    """Rolling-window memory with entity tracking and pronoun resolution."""

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns

        # Each entry: {"role": "user"|"assistant", "content": str,
        #              "intent": Optional[str], "entities": Dict[str, List[str]]}
        self.conversation_history: List[Dict[str, Any]] = []

        # Running set of every entity mentioned this session (most recent first).
        self.entity_context: Dict[str, List[str]] = defaultdict(list)

        # Last-seen convenience pointers (used for pronoun resolution).
        self.last_lot: Optional[str] = None
        self.last_supplier: Optional[str] = None
        self.last_line: Optional[str] = None

        # "lot:L-778", "supplier:SUP-C", "line:LINE-2" or None.
        self.active_investigation: Optional[str] = None

        # Internal counters for active_investigation detection.
        self._turn_history_by_entity: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Turn bookkeeping
    # ------------------------------------------------------------------

    def add_turn(
        self,
        question: str,
        response: str,
        intent: Any,
        entities: Dict[str, List[str]],
    ) -> None:
        """Record a single user/assistant exchange and refresh state."""
        intent_name = getattr(intent, "intent", None) or (
            intent.get("intent") if isinstance(intent, dict) else None
        )

        self.conversation_history.append({
            "role":     "user",
            "content":  question,
            "intent":   intent_name,
            "entities": {k: list(v) for k, v in (entities or {}).items() if v},
        })
        self.conversation_history.append({
            "role":     "assistant",
            "content":  response,
            "intent":   intent_name,
            "entities": {},
        })

        # Window the history.
        if len(self.conversation_history) > self.max_turns * 2:
            self.conversation_history = self.conversation_history[-self.max_turns * 2:]

        # Update entity_context (most-recent-first, dedup).
        for key, values in (entities or {}).items():
            for v in values:
                bucket = self.entity_context[key]
                if v in bucket:
                    bucket.remove(v)
                bucket.insert(0, v)

        # Convenience pointers.
        if entities:
            if entities.get("lot_no"):
                self.last_lot = entities["lot_no"][0]
            if entities.get("supplier"):
                self.last_supplier = entities["supplier"][0]
            if entities.get("line"):
                self.last_line = entities["line"][0]

        # Active-investigation heuristic:
        #   3+ turns about the same lot      -> lot:<id>
        #   2+ turns about the same supplier -> supplier:<id>
        #   2+ turns about the same line     -> line:<id>
        self._update_active_investigation(entities or {})

    def _update_active_investigation(self, entities: Dict[str, List[str]]) -> None:
        # Recompute counts over the last few user turns.
        counts: Dict[str, int] = defaultdict(int)
        for turn in self.conversation_history[-self.max_turns * 2:]:
            if turn["role"] != "user":
                continue
            for e in turn["entities"].get("lot_no", []):
                counts[f"lot:{e}"] += 1
            for e in turn["entities"].get("supplier", []):
                counts[f"supplier:{e}"] += 1
            for e in turn["entities"].get("line", []):
                counts[f"line:{e}"] += 1

        # Thresholds per entity family.
        best: Optional[str] = None
        for key, n in counts.items():
            if key.startswith("lot:") and n >= 3:
                if best is None or n > counts.get(best, 0):
                    best = key
            elif key.startswith("supplier:") and n >= 2:
                if best is None or n > counts.get(best, 0):
                    best = key
            elif key.startswith("line:") and n >= 2:
                if best is None or n > counts.get(best, 0):
                    best = key

        self.active_investigation = best

    # ------------------------------------------------------------------
    # Pronoun resolution
    # ------------------------------------------------------------------

    def resolve_entities(self, question: str) -> ResolvedQuestion:
        """Rewrite *question* to fill in entities referenced by pronoun.

        Returns a :class:`ResolvedQuestion` whose ``text`` is either the
        original question (no rewrite needed) or an enriched form.
        """
        original = question
        existing = extract_entities(question)
        q_lower = question.lower()
        substitutions: List[str] = []
        enriched = question

        has_lot = bool(existing.get("lot_no"))
        has_supplier = bool(existing.get("supplier"))
        has_line = bool(existing.get("line"))
        has_any_entity = any(existing.values())

        # ── 1. Supplier pronouns: "its supplier", "that supplier", ... ─────
        if not has_supplier and self.last_supplier:
            for pat in _SUPPLIER_PRONOUN_PATTERNS:
                if re.search(pat, q_lower):
                    enriched, n = re.subn(
                        pat, f"supplier {self.last_supplier}",
                        enriched, count=1, flags=re.IGNORECASE,
                    )
                    if n:
                        substitutions.append(f"supplier pronoun -> {self.last_supplier}")
                        has_supplier = True
                        break

        # ── 2. Lot pronouns ─────────────────────────────────────────────────
        if not has_lot and self.last_lot:
            for pat in _LOT_PRONOUN_PATTERNS:
                if re.search(pat, q_lower):
                    enriched, n = re.subn(
                        pat, f"lot {self.last_lot}",
                        enriched, count=1, flags=re.IGNORECASE,
                    )
                    if n:
                        substitutions.append(f"lot pronoun -> {self.last_lot}")
                        has_lot = True
                        break

        # ── 3. Line pronouns ────────────────────────────────────────────────
        if not has_line and self.last_line:
            for pat in _LINE_PRONOUN_PATTERNS:
                if re.search(pat, q_lower):
                    enriched, n = re.subn(
                        pat, f"line {self.last_line}",
                        enriched, count=1, flags=re.IGNORECASE,
                    )
                    if n:
                        substitutions.append(f"line pronoun -> {self.last_line}")
                        has_line = True
                        break

        # ── 4. Bare "it" / "its" + comparison context ─────────────────────
        # "Compare it with SUP-A" -> "Compare <last_supplier> with SUP-A"
        compare_mode = any(tok in q_lower for tok in _COMPARISON_TOKENS)
        if (
            compare_mode
            and self.last_supplier
            and re.search(r"\bit\b", q_lower)
            and self.last_supplier not in enriched
        ):
            enriched, n = re.subn(
                r"\bit\b", self.last_supplier, enriched, count=1, flags=re.IGNORECASE
            )
            if n:
                substitutions.append(f"'it' -> {self.last_supplier} (comparison)")
                has_supplier = True

        # ── 5. Bare "its" modifier (e.g. "What about its supplier?") ──────
        # If we still have nothing and "its" is present, choose the most
        # recently-anchored entity family.
        if not has_lot and not has_supplier and not has_line:
            if re.search(r"\bits\b", q_lower) and self.last_lot:
                # "Its" typically refers to the lot in our domain.
                enriched = re.sub(
                    r"\bits\b",
                    f"lot {self.last_lot}'s",
                    enriched,
                    count=1,
                    flags=re.IGNORECASE,
                )
                substitutions.append(f"'its' -> lot {self.last_lot}'s")
                has_lot = True

        # ── 6. Bare follow-up with no entity + active investigation ───────
        # e.g. "What should I do?" when actively investigating lot L-778.
        if (
            not has_any_entity
            and not substitutions
            and self.active_investigation
        ):
            if any(re.search(p, q_lower) for p in _BARE_FOLLOWUP_PATTERNS):
                kind, _, ent_id = self.active_investigation.partition(":")
                tag = {"lot": "lot", "supplier": "supplier", "line": "line"}.get(kind, kind)
                suffix = f" for {tag} {ent_id}"
                # Insert the suffix before the trailing '?' if present.
                if enriched.rstrip().endswith("?"):
                    enriched = enriched.rstrip("?").rstrip() + suffix + "?"
                else:
                    enriched = enriched.rstrip(".") + suffix
                substitutions.append(f"bare follow-up + active_investigation -> {self.active_investigation}")

        return ResolvedQuestion(
            original=original,
            text=enriched,
            context_used=bool(substitutions),
            substitutions=substitutions,
        )

    # ------------------------------------------------------------------
    # Context block for Claude messages
    # ------------------------------------------------------------------

    def get_context_block(self) -> List[Dict[str, Any]]:
        """Return recent turns + a context-summary message for Claude.

        The summary is a synthetic user turn that tells the model what we
        are currently investigating; keeping it as the first item in the
        window gives the LLM a persistent anchor without burning tokens
        on full transcripts.
        """
        block: List[Dict[str, Any]] = []
        summary = self._summary_line()
        if summary:
            # Anthropic doesn't support 'system' inside messages -- fold
            # the memory summary into a user-role note.
            block.append({
                "role": "user",
                "content": f"[CONVERSATION CONTEXT]\n{summary}",
            })

        # Windowed transcript -- user/assistant alternation preserved.
        for turn in self.conversation_history[-self.max_turns * 2:]:
            block.append({
                "role":    turn["role"],
                "content": turn["content"],
            })
        return block

    def _summary_line(self) -> str:
        if not self.conversation_history:
            return ""
        bits: List[str] = []
        if self.active_investigation:
            bits.append(f"Currently investigating {self.active_investigation}.")
        if self.entity_context:
            flat: List[str] = []
            for key, values in self.entity_context.items():
                if values:
                    flat.append(f"{key}={','.join(values[:3])}")
            if flat:
                bits.append(f"Entities discussed: {'; '.join(flat)}.")
        last_assistant = next(
            (t for t in reversed(self.conversation_history) if t["role"] == "assistant"),
            None,
        )
        if last_assistant:
            preview = last_assistant["content"].strip().replace("\n", " ")[:180]
            if preview:
                bits.append(f"Previous findings (preview): {preview}")
        return " ".join(bits)


__all__ = ["ConversationMemory", "ResolvedQuestion"]
