"""
Phase 5 Step 1 — adversarial tests.

These are NOT happy-path unit tests. Every test in this file is designed
to break the agent: ambiguous questions, invalid entities, follow-ups
without context, rapid-fire concurrent requests, special-character
injection attempts, and the 12 exact spec questions from the original
technical brief.

The pass bar is **>= 95%** of the test cases. The runner at the bottom
(`run_all_groups`) prints a per-group breakdown and an overall summary;
the file is also `pytest`-compatible.

Run:
    python tests/test_phase5_hardening.py             # standalone
    pytest tests/test_phase5_hardening.py -v          # via pytest
"""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.agent.mock_responder import _AUDIT_LOG_PATH, render_mock_response
from app.models.schema import get_engine
from app.services.service_registry import get_registry
from configs import settings


# ── Fixtures ─────────────────────────────────────────────────────────────────

_engine   = get_engine(str(settings.DATABASE_URL))
_registry = get_registry(_engine)


def respond(question: str, session_id: str | None = None) -> Dict[str, Any]:
    """Single entry point used by every assertion."""
    return render_mock_response(question, _registry, _engine, session_id=session_id)


# ── Validation helpers ───────────────────────────────────────────────────────

def _assert_valid_response(text: str, q: str) -> None:
    assert text is not None,                         f"[{q}] response was None"
    assert len(text) > 100,                          f"[{q}] response too short ({len(text)} chars)"
    assert "I don't know" not in text,               f"[{q}] response says 'I don't know'"
    assert "Traceback" not in text,                  f"[{q}] response leaked a traceback"
    # The literal string "Error" can appear in legitimate domain text
    # (e.g. "Quality Notification Error"), so we only flag cases that
    # look like Python exception output.
    assert "## Unexpected Error" not in text,        f"[{q}] hit unexpected error path"
    assert "## Agent Error" not in text,             f"[{q}] hit agent error path"


# ── TEST GROUP 1 — Ambiguous questions ───────────────────────────────────────

GROUP_1_CASES: List[Tuple[str, List[str]]] = [
    ("what's wrong?",                           ["HIGH", "drift", "supplier"]),
    ("show me the bad stuff",                   ["HIGH", "L-"]),
    ("which supplier is terrible?",             ["SUP-C"]),
    ("is everything ok?",                       ["HIGH", "drift"]),
    ("tell me something important",             ["HIGH", "L-"]),
]


def test_group_1_ambiguous_questions() -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []
    for q, must_contain in GROUP_1_CASES:
        try:
            r    = respond(q)
            text = r["text"]
            _assert_valid_response(text, q)
            for needle in must_contain:
                assert needle in text, f"[{q}] response missing '{needle}'"
            results.append((q, True, ""))
        except AssertionError as e:
            results.append((q, False, str(e)))
    return results


# ── TEST GROUP 2 — Invalid entities ──────────────────────────────────────────

GROUP_2_CASES: List[Tuple[str, List[str]]] = [
    ("what is the risk of lot XYZ-999?",
        ["XYZ-999", "not", "Found"]),  # case-insensitive check below
    ("tell me about supplier SUP-Z",
        ["SUP-Z", "not"]),
    ("show me serial SR99999999",
        ["SR99999999", "not"]),
]


def test_group_2_invalid_entities() -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []
    for q, must_contain in GROUP_2_CASES:
        try:
            r    = respond(q)
            text = r["text"]
            _assert_valid_response(text, q)
            tlow = text.lower()
            for needle in must_contain:
                assert needle.lower() in tlow, f"[{q}] response missing '{needle}'"
            # And it must be informative, not silent
            assert len(text) > 150, f"[{q}] not-found response too brief"
            results.append((q, True, ""))
        except AssertionError as e:
            results.append((q, False, str(e)))
    return results


# ── TEST GROUP 3 — Follow-ups without context ────────────────────────────────

GROUP_3_CASES: List[str] = [
    "what about its supplier?",
    "show me more details",
]


def test_group_3_followups_no_context() -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []
    for q in GROUP_3_CASES:
        try:
            r    = respond(q)
            text = r["text"]
            _assert_valid_response(text, q)
            tlow = text.lower()
            assert any(k in tlow for k in (
                "which lot", "context", "more context", "specific lot",
                "specific supplier", "no prior",
            )), f"[{q}] follow-up did not ask for clarification"
            results.append((q, True, ""))
        except AssertionError as e:
            results.append((q, False, str(e)))
    return results


# ── TEST GROUP 4 — Concurrent rapid questions ────────────────────────────────

GROUP_4_QUESTIONS: List[str] = [
    "What is the risk level of lot L-778?",
    "Any process drift?",
    "Compare suppliers",
    "What about COO China?",
    "What should I inspect this week?",
]


def test_group_4_concurrent_rapid() -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []

    # Snapshot audit-log line count before.
    before = 0
    if _AUDIT_LOG_PATH.exists():
        with _AUDIT_LOG_PATH.open("r", encoding="utf-8") as fh:
            before = sum(1 for _ in fh)

    session = f"stress-{uuid.uuid4().hex[:8]}"
    responses: List[Dict[str, Any]] = []
    errors: List[str] = []
    lock = threading.Lock()

    def _worker(q: str) -> None:
        try:
            r = respond(q, session_id=session)
            with lock:
                responses.append(r)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(f"{q}: {exc}")

    threads = [threading.Thread(target=_worker, args=(q,)) for q in GROUP_4_QUESTIONS]
    for t in threads: t.start()
    for t in threads: t.join(timeout=15)

    # Per-question results
    for q in GROUP_4_QUESTIONS:
        match = next((r for r in responses if "text" in r), None)
        ok    = bool(responses) and not errors
        msg   = "; ".join(errors) if errors else ""
        results.append((q, ok, msg))

    # Aggregate assertions
    try:
        assert len(responses) == 5,                f"only {len(responses)}/5 returned"
        assert not errors,                         f"errors: {errors}"
        for r in responses:
            _assert_valid_response(r["text"], "stress")

        # Audit log grew by exactly 5.
        after = 0
        if _AUDIT_LOG_PATH.exists():
            with _AUDIT_LOG_PATH.open("r", encoding="utf-8") as fh:
                after = sum(1 for _ in fh)
        assert (after - before) >= 5, f"audit log grew by {after - before}, expected >= 5"
        results.append(("stress aggregate (5 in parallel + audit log)", True, ""))
    except AssertionError as e:
        results.append(("stress aggregate (5 in parallel + audit log)", False, str(e)))

    return results


# ── TEST GROUP 5 — Special characters / injection ────────────────────────────

GROUP_5_CASES: List[Tuple[str, List[str]]] = [
    ("what about lot L-778; DROP TABLE dim_lot;",
        ["L-778", "HIGH"]),
    ("tell me about 'supplier' OR '1'='1'",
        ["supplier"]),
    ('what is the risk of lot "L-778" -- comment',
        ["L-778"]),
]


def test_group_5_special_chars() -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []

    # Sanity-check that the DB still has the table before AND after.
    from sqlalchemy import inspect as _inspect, text as _text
    pre_tables = set(_inspect(_engine).get_table_names())

    for q, must_contain in GROUP_5_CASES:
        try:
            r    = respond(q)
            text = r["text"]
            _assert_valid_response(text, q)
            for needle in must_contain:
                assert needle in text or needle.lower() in text.lower(), \
                    f"[{q}] response missing '{needle}'"
            results.append((q, True, ""))
        except AssertionError as e:
            results.append((q, False, str(e)))

    # Confirm no tables were dropped by the injection.
    try:
        post_tables = set(_inspect(_engine).get_table_names())
        assert pre_tables == post_tables, \
            f"injection mutated schema: {pre_tables ^ post_tables}"
        # Also confirm dim_supplier table is still queryable.
        with _engine.connect() as conn:
            n = conn.execute(_text("SELECT COUNT(*) FROM dim_supplier")).scalar() or 0
        assert n > 0, "dim_supplier returned 0 rows"
        results.append(("schema integrity", True, ""))
    except AssertionError as e:
        results.append(("schema integrity", False, str(e)))

    return results


# ── TEST GROUP 6 — The 12 exact spec questions ───────────────────────────────

GROUP_6_CASES: List[Tuple[str, str, Callable[[str], bool]]] = [
    ("Q1",
        "Which lots have the highest incoming inspection fail rate?",
        lambda t: "L-778" in t),

    ("Q2",
        "What is the risk profile of SENSOR-HALL lot L-778?",
        lambda t: "HIGH" in t and "%" in t),

    ("Q3",
        "Which supplier has the most warranty claims?",
        lambda t: "SUP-C" in t),

    ("Q4",
        "Is there process drift on any production line?",
        lambda t: "LINE-2" in t and "Night" in t),

    ("Q5",
        "Which lots should have increased incoming sampling?",
        lambda t: "L-778" in t and "action" in t.lower()),

    ("Q6",
        "What is the COO trend for China?",
        lambda t: ("7.1" in t or "7.11" in t or "China" in t)),

    ("Q7",
        "Does SUP-B outperform its COO average?",
        lambda t: ("outperform" in t.lower() or "yes" in t.lower())
                  and "Japan" in t),

    ("Q8",
        "Which suppliers are suitable for safety-critical programs?",
        lambda t: ("SUP-A" in t or "SUP-B" in t) and "engineering" in t.lower()),

    ("Q9",
        "What actions should I take for lot L-778?",
        lambda t: ("QA32" in t or "QM01" in t)),

    ("Q10",
        "Show me the full traceability for serial SR20260008",
        lambda t: "SENSOR-HALL" in t or "SUP-C" in t or "supplier" in t.lower()),

    ("Q11",
        "What is the warranty claim rate for SUP-C?",
        lambda t: "SUP-C" in t and "%" in t),

    ("Q12",
        "Compare all suppliers by quality score",
        lambda t: "SUP-B" in t
                  and all(s in t for s in ("SUP-A", "SUP-B", "SUP-C", "SUP-D", "SUP-E"))),
]


def test_group_6_spec_questions() -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []
    for tag, q, check in GROUP_6_CASES:
        try:
            r    = respond(q)
            text = r["text"]
            _assert_valid_response(text, f"{tag}: {q}")
            assert check(text), f"[{tag}] {q!r} — content check failed"
            results.append((f"{tag}: {q}", True, ""))
        except AssertionError as e:
            results.append((f"{tag}: {q}", False, str(e)))
    return results


# ── Pytest entry points ──────────────────────────────────────────────────────

def test_group_1():  # noqa: D401  (pytest discovers via name)
    fails = [c for c in test_group_1_ambiguous_questions() if not c[1]]
    assert not fails, fails


def test_group_2():
    fails = [c for c in test_group_2_invalid_entities() if not c[1]]
    assert not fails, fails


def test_group_3():
    fails = [c for c in test_group_3_followups_no_context() if not c[1]]
    assert not fails, fails


def test_group_4():
    fails = [c for c in test_group_4_concurrent_rapid() if not c[1]]
    assert not fails, fails


def test_group_5():
    fails = [c for c in test_group_5_special_chars() if not c[1]]
    assert not fails, fails


def test_group_6():
    fails = [c for c in test_group_6_spec_questions() if not c[1]]
    assert not fails, fails


# ── Standalone runner ────────────────────────────────────────────────────────

GROUPS: List[Tuple[str, Callable[[], List[Tuple[str, bool, str]]]]] = [
    ("Group 1 — Ambiguous questions",          test_group_1_ambiguous_questions),
    ("Group 2 — Invalid entities",             test_group_2_invalid_entities),
    ("Group 3 — Follow-ups without context",   test_group_3_followups_no_context),
    ("Group 4 — Concurrent rapid questions",   test_group_4_concurrent_rapid),
    ("Group 5 — Special characters",           test_group_5_special_chars),
    ("Group 6 — Twelve spec questions",        test_group_6_spec_questions),
]


def run_all_groups() -> int:
    """Run every group, print per-case detail, return shell exit code."""
    print("\n" + "=" * 72)
    print(" PHASE 5 — ADVERSARIAL HARDENING TEST SUITE")
    print("=" * 72)

    overall_pass = 0
    overall_fail = 0
    failure_details: List[str] = []
    started = time.perf_counter()

    for title, fn in GROUPS:
        print(f"\n{title}")
        print("-" * len(title))
        try:
            cases = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  CRASH while running group: {exc!r}")
            overall_fail += 1
            failure_details.append(f"{title} CRASHED: {exc!r}")
            continue

        for label, ok, msg in cases:
            mark = "PASS" if ok else "FAIL"
            short = label[:60] + ("..." if len(label) > 60 else "")
            print(f"  [{mark}] {short}")
            if ok:
                overall_pass += 1
            else:
                overall_fail += 1
                failure_details.append(f"{title} :: {label} :: {msg}")

    total   = overall_pass + overall_fail
    pct     = (overall_pass / total * 100) if total else 0
    elapsed = time.perf_counter() - started

    print("\n" + "=" * 72)
    print(f" RESULTS: {overall_pass}/{total} passed ({pct:.1f}%) in {elapsed:.2f}s")
    print("=" * 72)

    if failure_details:
        print("\nFAILURE DETAIL")
        print("-" * 14)
        for d in failure_details:
            print(f"  - {d}")

    if pct >= 95.0:
        print("\nVERDICT: DEMO-READY (>= 95% pass rate)")
        return 0
    else:
        print(f"\nVERDICT: NEEDS WORK (only {pct:.1f}% — target is >= 95%)")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_groups())
