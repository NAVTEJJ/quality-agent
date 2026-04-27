"""
Intent classifier for the Quality Agent (Phase 3, Step 1).

Splits a user question into:

* **intent** -- one of ten fixed business intents, decided by Claude via a
  tool-use call (deterministic JSON output, not free-form text).
* **entities** -- lot, supplier, serial, line, shift, COO, component,
  extracted deterministically via regex so we never round-trip a lot
  number or serial through the LLM.
* **primary_tool / secondary_tools** -- the tool dispatch plan for the
  downstream orchestrator, looked up from a static intent -> tools map.

Cost / latency
--------------
Each classification is a single Claude message call with forced tool use
and ``max_tokens=400``. Sonnet-4-5 is the default model; pass a different
model (or a custom client) in the constructor for tests / fallbacks.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

VALID_INTENTS: List[str] = [
    "LOT_RISK_QUERY",
    "SUPPLIER_PROFILE",
    "PROCESS_DRIFT",
    "COO_ANALYSIS",
    "DRILL_DOWN",
    "INSPECTION_STRATEGY",
    "SUPPLIER_COMPARE",
    "WARRANTY_TRACE",
    "GENERAL_INSIGHT",
    "ACTION_REQUEST",
]

# Human-readable guide that gets embedded in the classifier system prompt.
_INTENT_GUIDE: Dict[str, str] = {
    "LOT_RISK_QUERY":
        'Risk / status of a specific lot. Examples: "is lot X risky", '
        '"what\'s wrong with lot L-778", "how bad is L-778".',
    "SUPPLIER_PROFILE":
        'Overall profile of a supplier. Examples: "tell me about SUP-C", '
        '"which suppliers are reliable", "is SUP-A any good".',
    "PROCESS_DRIFT":
        'Production line / shift issues. Examples: "any line issues", '
        '"LINE-2 night shift problems", "why is torque failing".',
    "COO_ANALYSIS":
        'Country-of-origin quality trends. Examples: "China quality trends", '
        '"country performance", "how is Germany doing".',
    "DRILL_DOWN":
        'Requests for details / traceability / evidence for something the '
        'user already knows about. Examples: "show me details for lot X", '
        '"trace serial SR20260008", "full drill-down for L-778".',
    "INSPECTION_STRATEGY":
        'Where to focus inspection effort. Examples: "where to focus '
        'inspection", "which lots need sampling", "can we reduce inspection".',
    "SUPPLIER_COMPARE":
        'Side-by-side of two or more suppliers. Examples: "compare SUP-A '
        'vs SUP-B", "best supplier for critical builds", "SUP-B or SUP-C".',
    "WARRANTY_TRACE":
        'Field warranty failures and their root cause. Examples: "why did '
        'this unit fail in the field", "warranty root cause for SR...".',
    "GENERAL_INSIGHT":
        'Broad questions about current risk posture. Examples: "what are '
        'the biggest quality risks right now", "give me a summary".',
    "ACTION_REQUEST":
        'Requests for actions to take. Examples: "what should I do about X", '
        '"recommend actions for Y", "next steps for L-778".',
}

# Intent -> tool dispatch plan. Primary tool runs first; secondary tools
# enrich the response. Kept as a pure map so tests can assert it.
INTENT_TOOL_MAP: Dict[str, Dict[str, Any]] = {
    "LOT_RISK_QUERY":      {"primary": "get_lot_risk",           "secondary": ["get_drill_down", "get_supplier_profile"]},
    "SUPPLIER_PROFILE":    {"primary": "get_supplier_profile",   "secondary": ["get_coo_trend", "compare_suppliers"]},
    "PROCESS_DRIFT":       {"primary": "get_process_drift",      "secondary": ["get_drill_down"]},
    "COO_ANALYSIS":        {"primary": "get_coo_trend",          "secondary": ["compare_suppliers", "get_supplier_profile"]},
    "DRILL_DOWN":          {"primary": "get_drill_down",         "secondary": ["get_lot_risk", "get_warranty_trace"]},
    "INSPECTION_STRATEGY": {"primary": "get_inspection_strategy", "secondary": ["get_lot_risk"]},
    "SUPPLIER_COMPARE":    {"primary": "compare_suppliers",      "secondary": ["get_coo_trend", "get_supplier_profile"]},
    "WARRANTY_TRACE":      {"primary": "get_warranty_trace",     "secondary": ["get_drill_down", "get_lot_risk"]},
    "GENERAL_INSIGHT":     {"primary": "search_insights",        "secondary": ["get_inspection_strategy", "get_lot_risk"]},
    "ACTION_REQUEST":      {"primary": "get_action_playbook",    "secondary": ["get_lot_risk", "get_supplier_profile"]},
}


# ---------------------------------------------------------------------------
# Entity extraction -- regex, no LLM
# ---------------------------------------------------------------------------

_KNOWN_COUNTRIES = ("China", "Japan", "Germany", "India", "Mexico")
_KNOWN_COMPONENTS = ("SENSOR-HALL", "BEARING-SET", "SEAL-KIT", "HOUSING")

_PATTERNS: Dict[str, re.Pattern[str]] = {
    # L-778 or LOT-SEAL-215 style
    "lot_no":    re.compile(r"\b(?:LOT-[A-Z]+-\d+|L-\d+)\b", re.IGNORECASE),
    "supplier":  re.compile(r"\bSUP-[A-Z]\b", re.IGNORECASE),
    "serial":    re.compile(r"\bSR\d+\b", re.IGNORECASE),
    "line":      re.compile(r"\bLINE-\d+\b", re.IGNORECASE),
    "shift":     re.compile(r"\b(Day|Night)\b", re.IGNORECASE),
    "coo":       re.compile(
        r"\b(" + "|".join(_KNOWN_COUNTRIES) + r")\b", re.IGNORECASE
    ),
    "component": re.compile(
        r"\b(" + "|".join(_KNOWN_COMPONENTS) + r")\b", re.IGNORECASE
    ),
}


def _canonicalise(entity_type: str, raw: str) -> str:
    """Normalise each entity to the form stored in the warehouse."""
    s = raw.strip()
    if entity_type == "shift":
        return s.capitalize()            # day -> Day, NIGHT -> Night
    if entity_type == "coo":
        return s.title()                 # china -> China
    return s.upper()                     # L-778, SUP-A, LINE-2, SR20260008


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Pull structured entities from free text.

    Always returns all seven entity keys so downstream code can rely on
    the shape; values are ordered lists of unique canonical strings.
    """
    result: Dict[str, List[str]] = {}
    for name, pattern in _PATTERNS.items():
        matches: List[str] = []
        seen: set[str] = set()
        for m in pattern.finditer(text):
            raw = m.group(1) if m.groups() else m.group(0)
            canon = _canonicalise(name, raw)
            if canon and canon not in seen:
                seen.add(canon)
                matches.append(canon)
        result[name] = matches
    return result


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    intent: str
    entities: Dict[str, List[str]]
    confidence: float
    primary_tool: str
    secondary_tools: List[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Classifier system prompt -- note this is DIFFERENT from the main agent
# system prompt. Scope = single-turn classification, structured output.
# ---------------------------------------------------------------------------

def _build_classifier_system_prompt() -> str:
    lines = [
        "You are an intent classifier for a quality-management AI copilot.",
        "Given a user's question, classify it into EXACTLY ONE of these ten intents:",
        "",
    ]
    for intent in VALID_INTENTS:
        lines.append(f"- {intent} -- {_INTENT_GUIDE[intent]}")
    lines += [
        "",
        "Tie-breaking guidance:",
        "- A question mentioning a specific lot number defaults to LOT_RISK_QUERY,",
        "  UNLESS the user asks for details / traceability (DRILL_DOWN),",
        "  comparison (SUPPLIER_COMPARE), or an action plan (ACTION_REQUEST).",
        "- A question about a production line or shift defaults to PROCESS_DRIFT.",
        "- A question comparing two or more suppliers defaults to SUPPLIER_COMPARE",
        "  even if it also asks for a recommendation.",
        "- A broad 'what should I worry about' question is GENERAL_INSIGHT.",
        "- A 'what should I do' question about a known entity is ACTION_REQUEST.",
        "",
        "Return your answer via the classify_intent tool. Choose the single",
        "best intent -- do not hedge, do not return multiple intents.",
    ]
    return "\n".join(lines)


_CLASSIFIER_SYSTEM_PROMPT = _build_classifier_system_prompt()


_CLASSIFY_TOOL = {
    "name": "classify_intent",
    "description": (
        "Record the single classified intent for the user's question, "
        "together with a confidence score and a one-sentence reason."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": VALID_INTENTS,
                "description": "The one intent that best matches the question.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Classifier confidence in [0, 1].",
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence justifying the classification.",
            },
        },
        "required": ["intent", "confidence", "reasoning"],
    },
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-5"


class IntentClassifier:
    """LLM-backed intent classifier with deterministic entity extraction."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: Optional[anthropic.Anthropic] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 400,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Export the variable or "
                    "pass api_key=... to IntentClassifier()."
                )
            self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------

    def classify(self, question: str) -> IntentResult:
        """Classify *question* and return a populated :class:`IntentResult`."""
        entities = extract_entities(question)

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_CLASSIFIER_SYSTEM_PROMPT,
            tools=[_CLASSIFY_TOOL],
            tool_choice={"type": "tool", "name": "classify_intent"},
            messages=[{"role": "user", "content": question}],
        )

        tool_use = next(
            (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_use is None:
            raise RuntimeError(
                f"Classifier returned no tool_use block. Question: {question!r}; "
                f"response: {resp.content!r}"
            )

        args = tool_use.input
        intent = args["intent"]
        if intent not in INTENT_TOOL_MAP:
            raise RuntimeError(f"Classifier emitted unknown intent: {intent!r}")

        tools_plan = INTENT_TOOL_MAP[intent]
        return IntentResult(
            intent=intent,
            entities=entities,
            confidence=float(args["confidence"]),
            primary_tool=tools_plan["primary"],
            secondary_tools=list(tools_plan["secondary"]),
            reasoning=str(args.get("reasoning", "")),
        )


__all__ = [
    "IntentClassifier",
    "IntentResult",
    "VALID_INTENTS",
    "INTENT_TOOL_MAP",
    "extract_entities",
]
