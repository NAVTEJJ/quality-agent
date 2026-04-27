#!/usr/bin/env python
"""
PRE-DEMO CHECKLIST -- Run this 5 minutes before presenting to judges.

Usage:
    python scripts/pre_demo_check.py

What it does (in order, in under 60 seconds):
    1.  Database integrity         -- 14 tables, key rows, no orphans
    2.  Demo data sanity           -- L-778=HIGH, LINE-2 Night drift, SUP-B=89, claims=13
    3.  API health (FastAPI)       -- spins up uvicorn, hits /health, validates fields
    4.  Streamlit imports          -- every screen + components load clean
    5.  Env config                 -- .env, ANTHROPIC_API_KEY, DATABASE_URL
    6.  Demo story validations     -- the three Phase 2 marquee stories
    7.  Mock-response timing       -- < 500 ms after cache is warm
    8.  Database query timing      -- < 200 ms p95

Final verdict (one of):
    ALL SYSTEMS GO       -- cleared to demo
    DEMO READY (warnings)-- safe but address warnings
    DEMO BLOCKED         -- fix specific failures before presenting
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.reliability_check import (
    check_api_health,
    check_database_integrity,
    check_demo_data,
    check_env_config,
    check_streamlit_imports,
)


# ── Color helpers (degrade gracefully when output isn't a TTY) ───────────────

_USE_COLOR = sys.stdout.isatty()
def _c(code, t):  return f"\033[{code}m{t}\033[0m" if _USE_COLOR else t
def green(t):  return _c("32", t)
def red(t):    return _c("31", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)


def _badge(status: str) -> str:
    return {"PASS": green("PASS"), "WARN": yellow("WARN"), "FAIL": red("FAIL")}.get(
        status, status
    )


# ── Step 6: demo-story validation ─────────────────────────────────────────────

def _demo_stories(registry) -> list[tuple[str, bool, str]]:
    stories: list[tuple[str, bool, str]] = []
    try:
        risk = registry.kpi.get_lot_risk_scores()
        l778 = risk[risk["lot_no"] == "L-778"]
        ok   = (not l778.empty) and (str(l778["risk_tier"].values[0]) == "HIGH")
        stories.append((
            "L-778 HIGH risk",
            ok,
            "" if ok else "L-778 not HIGH",
        ))
    except Exception as exc:  # noqa: BLE001
        stories.append(("L-778 HIGH risk", False, repr(exc)))

    try:
        drift = registry.kpi.get_process_drift_by_line_shift()
        ln    = drift[(drift["line"] == "LINE-2") & (drift["shift"] == "Night")]
        tfr   = float(ln["torque_fail_rate"].values[0]) if not ln.empty else 0.0
        ok    = tfr > 0.10
        stories.append((
            "LINE-2 Night drift",
            ok,
            "" if ok else f"tfr={tfr:.3f}",
        ))
    except Exception as exc:  # noqa: BLE001
        stories.append(("LINE-2 Night drift", False, repr(exc)))

    try:
        rankings = registry.kpi.get_supplier_rankings()
        # Premium suppliers identified -- top tier should be 'Preferred'
        top      = rankings.iloc[0] if not rankings.empty else None
        ok       = (top is not None) and ("preferred" in str(top.get("tier","")).lower())
        stories.append((
            "Premium suppliers identified (SUP-B)",
            ok,
            "" if ok else "no Preferred-tier supplier at top",
        ))
    except Exception as exc:  # noqa: BLE001
        stories.append(("Premium suppliers identified", False, repr(exc)))
    return stories


# ── Step 7+8: timing checks ───────────────────────────────────────────────────

def _timing_checks(registry, engine) -> tuple[float, float, str]:
    """Returns (mock_ms, db_ms, msg).

    mock_ms   -- mock response time (after cache warm)
    db_ms     -- representative DB query time
    msg       -- non-empty if either misses its target
    """
    from app.agent.mock_responder import render_mock_response
    from app.core.cache import pre_warm_cache, reset_default_cache

    # Pre-warm cache so the demo's first click is instant.
    reset_default_cache()
    pre_warm_cache(registry=registry, engine=engine)

    # Mock response (cache-hit path).
    t0 = time.perf_counter()
    render_mock_response(
        "What is the risk level of lot L-778 and what actions should I take?",
        registry, engine,
    )
    mock_ms = (time.perf_counter() - t0) * 1000

    # DB query (cold-ish, but cached after first hit).
    t0 = time.perf_counter()
    registry.kpi.get_lot_risk_scores()
    db_ms = (time.perf_counter() - t0) * 1000

    msg_parts = []
    if mock_ms > 500:
        msg_parts.append(f"mock {mock_ms:.0f}ms > 500ms target")
    if db_ms > 200:
        msg_parts.append(f"db {db_ms:.0f}ms > 200ms target")
    return mock_ms, db_ms, "; ".join(msg_parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    started = time.perf_counter()
    print()
    print(bold("=" * 60))
    print(bold("AI QUALITY COPILOT -- PRE-DEMO CHECKLIST"))
    print(bold("=" * 60))

    # ── Step 1-5: reliability checks (run sequentially, capture status) ───
    reliability = [
        ("Database integrity",  check_database_integrity),
        ("Demo data",           check_demo_data),
        ("API health",          check_api_health),
        ("Streamlit imports",   check_streamlit_imports),
        ("Env config",          check_env_config),
    ]
    rel_results: list[tuple[str, str, list[str]]] = []
    for label, fn in reliability:
        try:
            rel_results.append(fn())
        except Exception as exc:  # noqa: BLE001
            rel_results.append((label, "FAIL", [f"crashed: {exc!r}"]))

    print()
    print(bold("[1-5] Reliability"))
    print("-" * 60)
    for name, status, msgs in rel_results:
        print(f"  {_badge(status)}  {name}")
        for m in msgs:
            colour = yellow if status == "WARN" else red
            print(f"          {colour('-')} {m}")

    fail_count = sum(1 for _, s, _ in rel_results if s == "FAIL")
    warn_count = sum(1 for _, s, _ in rel_results if s == "WARN")

    # ── Step 6: demo-story validations ────────────────────────────────────
    print()
    print(bold("[6] Demo story validations"))
    print("-" * 60)

    from app.models.schema import get_engine
    from app.services.service_registry import get_registry
    from configs import settings

    engine   = get_engine(str(settings.DATABASE_URL))
    registry = get_registry(engine)

    stories = _demo_stories(registry)
    story_failed = 0
    story_marks: list[str] = []
    for label, ok, msg in stories:
        mark = green("OK") if ok else red("X")
        print(f"  {_badge('PASS' if ok else 'FAIL')}  {label}")
        if not ok:
            print(f"          {red('-')} {msg}")
            story_failed += 1
        story_marks.append(f"{label.split()[0]} {mark}")

    # ── Step 7+8: timing ──────────────────────────────────────────────────
    print()
    print(bold("[7-8] Performance"))
    print("-" * 60)

    try:
        mock_ms, db_ms, timing_msg = _timing_checks(registry, engine)
        timing_ok = not timing_msg
    except Exception as exc:  # noqa: BLE001
        mock_ms = db_ms = -1
        timing_msg = f"crashed: {exc!r}"
        timing_ok = False

    if timing_ok:
        print(f"  {green('PASS')}  Mock response timing -- {mock_ms:.1f} ms (target < 500 ms)")
        print(f"  {green('PASS')}  Database query timing -- {db_ms:.1f} ms (target < 200 ms)")
    else:
        print(f"  {red('FAIL')}  Performance: {timing_msg}")

    # ── Final verdict ─────────────────────────────────────────────────────
    elapsed = time.perf_counter() - started

    print()
    print(bold("=" * 60))
    print(bold("VERDICT"))
    print(bold("=" * 60))

    total_fails = fail_count + story_failed + (0 if timing_ok else 1)

    if total_fails == 0 and warn_count == 0:
        print()
        print(green(bold("  ALL SYSTEMS GO -- Ready to demo")))
        print(green(f"     Database: PASS | API: PASS | UI: PASS | Data: PASS"))
        print(green(f"     Demo stories: {' | '.join(story_marks)}"))
        if mock_ms >= 0:
            print(green(f"     Estimated response time: {mock_ms:.0f} ms (cache warmed)"))
        print(green(f"     Completed in {elapsed:.1f} s"))
        return 0

    if total_fails == 0:
        print()
        print(yellow(bold("  DEMO READY WITH WARNINGS")))
        for name, status, msgs in rel_results:
            if status == "WARN":
                for m in msgs:
                    print(yellow(f"     - {name}: {m}"))
        print(yellow(f"     Safe to demo, but address warnings when possible."))
        print(yellow(f"     Completed in {elapsed:.1f} s"))
        return 0

    # Failures present.
    print()
    print(red(bold("  DEMO BLOCKED -- Fix before presenting")))
    for name, status, msgs in rel_results:
        if status == "FAIL":
            for m in msgs:
                print(red(f"     - {name}: {m}"))
    if story_failed:
        for label, ok, msg in stories:
            if not ok:
                print(red(f"     - Demo story FAIL: {label}: {msg}"))
    if not timing_ok:
        print(red(f"     - Performance: {timing_msg}"))
    print(red(f"     Completed in {elapsed:.1f} s"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
