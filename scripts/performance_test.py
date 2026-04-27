"""
Performance test harness -- Phase 5 Step 2.

Measures p50 / p95 / p99 wall-clock latency for every hot path the demo
hits. Anything that misses its target SLA gets flagged in the summary.

Usage
-----
    python scripts/performance_test.py
    python scripts/performance_test.py --skip-api    # skip FastAPI block
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Tiny ANSI-color helper ────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()
def _c(code: str, t: str) -> str:
    return f"\033[{code}m{t}\033[0m" if _USE_COLOR else t
def green(t):  return _c("32", t)
def red(t):    return _c("31", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)


# ── Stats ─────────────────────────────────────────────────────────────────────

def _percentiles(samples_ms: List[float]) -> Dict[str, float]:
    if not samples_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(samples_ms)
    def _pct(p: float) -> float:
        idx = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return s[idx]
    return {
        "p50":  _pct(0.50),
        "p95":  _pct(0.95),
        "p99":  _pct(0.99),
        "min":  s[0],
        "max":  s[-1],
        "mean": statistics.fmean(s),
        "n":    len(s),
    }


def _time_n(fn: Callable[[], Any], n: int) -> List[float]:
    """Run fn() n times, returning per-call ms timings."""
    out: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def _verdict(stats: Dict[str, float], target_p95_ms: float) -> Tuple[str, bool]:
    p95 = stats["p95"]
    ok  = p95 <= target_p95_ms
    mark = green("PASS") if ok else red("FAIL")
    return f"{mark}  p50 {stats['p50']:6.1f}ms  p95 {p95:6.1f}ms  " \
           f"p99 {stats['p99']:6.1f}ms  (target p95 <= {target_p95_ms:.0f}ms)", ok


# ── Suite 1: database queries ─────────────────────────────────────────────────

def test_database_queries() -> List[Tuple[str, bool, str]]:
    print(bold("\n[1/3] Database queries (10 runs each)"))
    print("-" * 60)
    from app.models.schema import get_engine
    from app.services.service_registry import get_registry
    from configs import settings

    engine   = get_engine(str(settings.DATABASE_URL))
    registry = get_registry(engine)

    cases: List[Tuple[str, Callable[[], Any], float]] = [
        ("get_lot_risk_scores",                lambda: registry.kpi.get_lot_risk_scores(),                  200.0),
        ("get_process_drift_by_line_shift",    lambda: registry.kpi.get_process_drift_by_line_shift(),     100.0),
        ("get_supplier_rankings",              lambda: registry.kpi.get_supplier_rankings(),               100.0),
        ("get_full_drill_down_chain('L-778')", lambda: registry.drill_down.get_full_drill_down_chain('L-778'), 500.0),
    ]

    results: List[Tuple[str, bool, str]] = []
    for label, fn, target in cases:
        try:
            # One warm-up call so the timing reflects steady-state — this
            # is the path judges actually hit after Streamlit's pre-warm.
            fn()
            timings = _time_n(fn, n=10)
            stats   = _percentiles(timings)
            verdict, ok = _verdict(stats, target)
            print(f"  {label:42s} {verdict}")
            results.append((label, ok, "" if ok else f"p95 {stats['p95']:.1f}ms exceeds {target}ms"))
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:42s} {red('CRASH')}  {exc!r}")
            results.append((label, False, repr(exc)))
    return results


# ── Suite 2: FastAPI endpoints ────────────────────────────────────────────────

_API_HOST = "127.0.0.1"
_API_PORT = 8765   # avoid clashing with the user's normal 8000


def _port_open(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) == 0


def _start_api() -> subprocess.Popen | None:
    """Boot uvicorn against app.agent.api:app on a fresh port."""
    cmd = [
        sys.executable, "-m", "uvicorn", "app.agent.api:app",
        "--host", _API_HOST, "--port", str(_API_PORT), "--log-level", "warning",
    ]
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(_ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  {red('FAIL')}  could not launch uvicorn: {exc}")
        return None

    # Wait up to 10s for the port.
    for _ in range(40):
        if _port_open(_API_HOST, _API_PORT):
            return proc
        time.sleep(0.25)
    proc.terminate()
    return None


def _http_get(path: str) -> Tuple[float, int]:
    url = f"http://{_API_HOST}:{_API_PORT}{path}"
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=5) as resp:
        resp.read()
        return (time.perf_counter() - t0) * 1000, resp.status


def test_api_endpoints() -> List[Tuple[str, bool, str]]:
    print(bold("\n[2/3] FastAPI endpoints (5 runs each)"))
    print("-" * 60)

    proc = _start_api()
    if proc is None:
        print(f"  {yellow('SKIP')}  uvicorn not available -- skipping endpoint timings")
        return [("api boot", False, "uvicorn failed to start")]

    try:
        cases: List[Tuple[str, str, float]] = [
            ("GET /health",                 "/health",                  50.0),
            ("GET /lot/L-778/risk",         "/lot/L-778/risk",         300.0),
            ("GET /process-drift",          "/process-drift",          200.0),
            ("GET /supplier/SUP-B/profile", "/supplier/SUP-B/profile", 200.0),
        ]
        # Warm the worker once so first-call cold start doesn't dominate.
        try:
            _http_get("/health")
        except Exception:
            pass

        results: List[Tuple[str, bool, str]] = []
        for label, path, target in cases:
            timings: List[float] = []
            error = ""
            try:
                for _ in range(5):
                    ms, status = _http_get(path)
                    if status != 200:
                        raise RuntimeError(f"status {status}")
                    timings.append(ms)
                stats = _percentiles(timings)
                verdict, ok = _verdict(stats, target)
                print(f"  {label:38s} {verdict}")
                results.append((label, ok, "" if ok else f"p95 {stats['p95']:.1f}ms exceeds {target}ms"))
            except (urllib.error.URLError, RuntimeError) as exc:
                print(f"  {label:38s} {red('FAIL')}  {exc}")
                results.append((label, False, str(exc)))
        return results
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


# ── Suite 3: Streamlit data load ──────────────────────────────────────────────

def test_streamlit_data_load() -> List[Tuple[str, bool, str]]:
    print(bold("\n[3/3] Streamlit data load"))
    print("-" * 60)

    from app.models.schema import get_engine
    from app.services.service_registry import clear_registry_cache, get_registry
    from configs import settings

    results: List[Tuple[str, bool, str]] = []

    # 3a -- registry init (cold).
    clear_registry_cache()
    t0 = time.perf_counter()
    engine = get_engine(str(settings.DATABASE_URL))
    registry = get_registry(engine)
    init_ms = (time.perf_counter() - t0) * 1000
    ok = init_ms <= 1000
    mark = green("PASS") if ok else red("FAIL")
    print(f"  {'registry init (cold)':38s} {mark}  {init_ms:6.1f}ms  (target <= 1000ms)")
    results.append(("registry init (cold)", ok, "" if ok else f"{init_ms:.0f}ms exceeds 1000ms"))

    # 3b -- per-screen data loads (the queries each screen actually executes).
    screen_loads: List[Tuple[str, Callable[[], Any]]] = [
        ("Screen B: dashboard data", lambda: (
            registry.kpi.get_lot_risk_scores(),
            registry.kpi.get_process_drift_by_line_shift(),
            registry.kpi.get_supplier_rankings(),
            registry.kpi.get_coo_performance(),
            registry.recommendations.get_inspection_strategy(),
        )),
        ("Screen C: drilldown L-778",  lambda: registry.drill_down.get_full_drill_down_chain('L-778')),
        ("Screen D: analytics data",   lambda: (
            registry.kpi.get_lot_risk_scores(),
            registry.kpi.get_coo_vs_supplier_decomposition(),
            registry.kpi.get_inspection_focus(),
        )),
    ]

    for label, fn in screen_loads:
        try:
            timings = _time_n(fn, n=3)
            stats   = _percentiles(timings)
            ok      = stats["p95"] <= 1000
            mark    = green("PASS") if ok else red("FAIL")
            print(f"  {label:38s} {mark}  p95 {stats['p95']:6.1f}ms  (target <= 1000ms)")
            results.append((label, ok, "" if ok else f"p95 {stats['p95']:.0f}ms exceeds 1000ms"))
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:38s} {red('CRASH')}  {exc!r}")
            results.append((label, False, repr(exc)))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip FastAPI block (uvicorn boot can be slow on cold runs)")
    args = parser.parse_args()

    started = time.perf_counter()
    print()
    print(bold(cyan("=" * 72)))
    print(bold(cyan(" PERFORMANCE TEST SUITE -- AI Quality Inspection Copilot")))
    print(bold(cyan("=" * 72)))

    all_results: List[Tuple[str, bool, str]] = []
    all_results += test_database_queries()
    if not args.skip_api:
        all_results += test_api_endpoints()
    all_results += test_streamlit_data_load()

    elapsed = time.perf_counter() - started
    passed = sum(1 for _, ok, _ in all_results if ok)
    failed = len(all_results) - passed

    print()
    print(bold(cyan("=" * 72)))
    print(f" {bold(str(passed))}/{len(all_results)} cases pass | "
          f"{elapsed:.1f}s wall-clock")
    print(bold(cyan("=" * 72)))

    if failed:
        print()
        print(yellow(bold("Slow paths to investigate:")))
        for label, ok, msg in all_results:
            if not ok:
                print(f"  - {label}: {msg}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
