"""
Tests for the Phase 3 intent classifier.

* Entity-extraction tests run unconditionally (regex, no API).
* LLM classification tests hit the Anthropic API -- automatically skipped
  when ANTHROPIC_API_KEY is not set.

Run with:  pytest tests/test_intent_classifier.py -v
"""
import os

import pytest

from app.agent.intent_classifier import (
    INTENT_TOOL_MAP,
    VALID_INTENTS,
    extract_entities,
)


# ---------------------------------------------------------------------------
# Taxonomy sanity -- runs unconditionally
# ---------------------------------------------------------------------------

def test_valid_intents_count():
    assert len(VALID_INTENTS) == 10


def test_intent_tool_map_covers_every_intent():
    assert set(INTENT_TOOL_MAP.keys()) == set(VALID_INTENTS)
    for intent, plan in INTENT_TOOL_MAP.items():
        assert plan["primary"], f"{intent}: primary tool is empty"
        assert isinstance(plan["secondary"], list)


# ---------------------------------------------------------------------------
# Deterministic entity extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, entity_type, expected",
    [
        ("What is the risk level of lot L-778?",              "lot_no",    ["L-778"]),
        ("Drill down on LOT-SEAL-215 today",                  "lot_no",    ["LOT-SEAL-215"]),
        ("Tell me about supplier SUP-C",                      "supplier",  ["SUP-C"]),
        ("Compare SUP-A and SUP-B for a safety-critical",     "supplier",  ["SUP-A", "SUP-B"]),
        ("Any issues on LINE-2 night shift?",                 "line",      ["LINE-2"]),
        ("Any issues on LINE-2 night shift?",                 "shift",     ["Night"]),
        ("How is China performing vs Germany?",               "coo",       ["China", "Germany"]),
        ("Why did serial SR20260008 fail?",                   "serial",    ["SR20260008"]),
        ("Status of SENSOR-HALL components",                  "component", ["SENSOR-HALL"]),
        ("Which lots should we prioritize for sampling?",     "lot_no",    []),
    ],
)
def test_extract_entities(text, entity_type, expected):
    result = extract_entities(text)
    assert entity_type in result
    assert result[entity_type] == expected


def test_extract_entities_returns_all_keys():
    out = extract_entities("hello world")
    assert set(out.keys()) == {
        "lot_no", "supplier", "serial", "line", "shift", "coo", "component"
    }
    assert all(v == [] for v in out.values())


def test_extract_entities_deduplicates_and_canonicalises():
    out = extract_entities("lot l-778 mentioned, then L-778 again; shift: NIGHT")
    assert out["lot_no"] == ["L-778"]
    assert out["shift"] == ["Night"]


# ---------------------------------------------------------------------------
# LLM classification -- needs ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

_API_KEY_SET = bool(os.getenv("ANTHROPIC_API_KEY"))

needs_api = pytest.mark.skipif(
    not _API_KEY_SET,
    reason="ANTHROPIC_API_KEY not set -- skipping live LLM classifier tests",
)


@pytest.fixture(scope="module")
def classifier():
    if not _API_KEY_SET:
        pytest.skip("ANTHROPIC_API_KEY not set")
    from app.agent.intent_classifier import IntentClassifier
    return IntentClassifier()


_GOLDEN_SET = [
    ("What is the risk level of lot L-778?",                       "LOT_RISK_QUERY"),
    ("Tell me about supplier SUP-C",                               "SUPPLIER_PROFILE"),
    ("Are there any process issues on LINE-2 night shift?",        "PROCESS_DRIFT"),
    ("How is China performing compared to other countries?",       "COO_ANALYSIS"),
    ("Show me the full traceability for lot L-778",                "DRILL_DOWN"),
    ("Which lots should we prioritize for incoming inspection?",   "INSPECTION_STRATEGY"),
    ("Compare SUP-A and SUP-B for a safety-critical program",      "SUPPLIER_COMPARE"),
    ("Why did serial SR20260008 fail in the field?",               "WARRANTY_TRACE"),
    ("What are the top quality risks right now?",                  "GENERAL_INSIGHT"),
    ("What actions should I take for lot L-778?",                  "ACTION_REQUEST"),
]


@needs_api
@pytest.mark.parametrize("question, expected_intent", _GOLDEN_SET)
def test_classify_intent(classifier, question, expected_intent):
    result = classifier.classify(question)
    assert result.intent == expected_intent, (
        f"Question: {question!r}\n"
        f"Expected: {expected_intent}\n"
        f"Got:      {result.intent}  (reason: {result.reasoning})"
    )
    # Every classified intent carries a usable tool plan.
    assert result.primary_tool
    assert 0.0 <= result.confidence <= 1.0
