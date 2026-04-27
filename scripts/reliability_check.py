"""
Reliability checks — Phase 5 Step 2.

Five short, focused health checks:

  * check_database_integrity   -- 14 tables, key rows, no orphans
  * check_demo_data            -- L-778 HIGH, LINE-2 Night drift, SUP-B 89, claims=13
  * check_streamlit_imports    -- every screen + components import clean
  * check_api_health           -- spin up uvicorn, hit /health, validate fields
  * check_env_config           -- .env present, ANTHROPIC_API_KEY, DATABASE_URL

Each returns a (name, status, list_of_messages) tuple. Status is one of:
    "PASS" | "WARN" | "FAIL"

Run standalone:
    python scripts/reliability_check.py
"""
from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Color helpers ─────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()
def _c(code, t):  return f"\033[{code}m{t}\033[0m" if _USE_COLOR else t
def green(t):  return _c("32", t)
def red(t):    return _c("31", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)


CheckResult = Tuple[str, str, List[str]]   # (name, status, messages)


# ── 1) Database integrity ─────────────────────────────────────────────────────

EXPECTED_TABLES = [
    "dim_supplier", "dim_component", "dim_lot", "dim_serial", "dim_material",
    "fact_incoming_qm", "fact_process_measurements", "fact_warranty_claims",
    "fact_constituent_bom",
    "agg_supplier_scorecard", "agg_coo_trends", "agg_coo_vs_supplier",
    "ref_action_playbook", "ref_ai_insights",
]


def check_database_integrity() -> CheckResult:
    msgs: List[str] = []
    try:
        from sqlalchemy import inspect as _inspect, text as _text
        from app.models.schema import get_engine
        from configs import settings

        engine = get_engine(str(settings.DATABASE_URL))
        tables = set(_inspect(engine).get_table_names())

        # 1a. Required tables exist.
        missing = [t for t in EXPECTED_TABLES if t not in tables]
        if missing:
            msgs.append(f"missing tables: {missing}")

        # 1b. Key rows exist.
        with engine.connect() as conn:
            n_lots = conn.execute(_text("SELECT COUNT(*) FROM dim_lot")).scalar() or 0
            if n_lots < 100:
                msgs.append(f"dim_lot only has {n_lots} rows (expected 100+)")

            l778 = conn.execute(
                _text("SELECT lot_id FROM dim_lot WHERE lot_no = :lot"),
                {"lot": "L-778"},
            ).fetchone()
            if l778 is None:
                msgs.append("L-778 missing from dim_lot")

            # 1c. LINE-2 Night drift signal — torque_fail_rate above the 10% threshold.
            from app.services.service_registry import get_registry
            registry = get_registry(engine)
            drift = registry.kpi.get_process_drift_by_line_shift()
            ln    = drift[(drift["line"] == "LINE-2") & (drift["shift"] == "Night")]
            if ln.empty:
                msgs.append("LINE-2 Night row missing from drift table")
            else:
                tfr = float(ln["torque_fail_rate"].values[0])
                if tfr <= 0.1:
                    msgs.append(
                        f"LINE-2 Night torque_fail_rate = {tfr:.3f} (expected > 0.1)"
                    )

            # 1d. No orphan facts (foreign keys resolve).
            orphan_q = (
                "SELECT COUNT(*) FROM fact_incoming_qm fq "
                "LEFT JOIN dim_lot l ON fq.lot_id = l.lot_id "
                "WHERE l.lot_id IS NULL"
            )
            orphans = conn.execute(_text(orphan_q)).scalar() or 0
            if orphans > 0:
                msgs.append(f"{orphans} orphan inspection rows (no matching dim_lot)")

    except Exception as exc:  # noqa: BLE001
        msgs.append(f"crashed: {exc!r}")

    return ("Database integrity", "PASS" if not msgs else "FAIL", msgs)


# ── 2) Demo data ──────────────────────────────────────────────────────────────

def check_demo_data() -> CheckResult:
    msgs: List[str] = []
    try:
        from sqlalchemy import text as _text
        from app.models.schema import get_engine
        from app.services.service_registry import get_registry
        from configs import settings

        engine   = get_engine(str(settings.DATABASE_URL))
        registry = get_registry(engine)

        # 2a. L-778 risk_tier == HIGH.
        risk = registry.kpi.get_lot_risk_scores()
        l778 = risk[risk["lot_no"] == "L-778"]
        if l778.empty:
            msgs.append("L-778 not in risk scores")
        elif str(l778["risk_tier"].values[0]) != "HIGH":
            msgs.append(f"L-778 tier is {l778['risk_tier'].values[0]} (expected HIGH)")

        # 2b. LINE-2 Night torque_fail_rate > 0.1.
        drift = registry.kpi.get_process_drift_by_line_shift()
        ln = drift[(drift["line"] == "LINE-2") & (drift["shift"] == "Night")]
        if ln.empty:
            msgs.append("LINE-2 Night row missing")
        else:
            tfr = float(ln["torque_fail_rate"].values[0])
            if tfr <= 0.1:
                msgs.append(f"LINE-2 Night torque_fail_rate = {tfr:.3f} (expected > 0.1)")

        # 2c. SUP-B quality_score == 89.
        rankings = registry.kpi.get_supplier_rankings()
        supb = rankings[rankings["supplier"] == "SUP-B"]
        if supb.empty:
            msgs.append("SUP-B missing from supplier_rankings")
        else:
            qs = int(supb["quality_score"].values[0])
            if qs != 89:
                msgs.append(f"SUP-B quality_score = {qs} (expected 89)")

        # 2d. ~13 warranty claims (allow some flex if dataset shifts slightly).
        with engine.connect() as conn:
            claims = conn.execute(_text("SELECT COUNT(*) FROM fact_warranty_claims")).scalar() or 0
        if claims < 8:
            msgs.append(f"only {claims} warranty claims (expected ~13)")

    except Exception as exc:  # noqa: BLE001
        msgs.append(f"crashed: {exc!r}")

    return ("Demo data", "PASS" if not msgs else "FAIL", msgs)


# ── 3) Streamlit imports ──────────────────────────────────────────────────────

def check_streamlit_imports() -> CheckResult:
    """Importing streamlit_app at module level invokes st.set_page_config
    which emits 'bare mode' warnings -- suppress them so the check output
    stays clean."""
    msgs: List[str] = []
    modules = [
        "app.frontend.theme",
        "app.frontend.components.screen_a_copilot",
        "app.frontend.components.screen_b_dashboard",
        "app.frontend.components.screen_c_drilldown",
        "app.frontend.components.screen_d_analytics",
        "app.frontend.components.kpi_cards",
        "app.frontend.components.export",
        "app.agent.mock_responder",
        "app.core.cache",
    ]
    # Streamlit emits warnings to stderr when imported in bare mode.
    devnull = open(os.devnull, "w")
    saved_stderr = sys.stderr
    try:
        sys.stderr = devnull
        for m in modules:
            try:
                __import__(m)
            except Exception as exc:  # noqa: BLE001
                msgs.append(f"{m}: {exc!r}")
        # Streamlit-app entry point can crash on st.set_page_config in bare
        # mode; do a syntax-only compile to catch import-level issues.
        try:
            import py_compile
            py_compile.compile(
                str(_ROOT / "app" / "frontend" / "streamlit_app.py"),
                doraise=True,
            )
        except Exception as exc:  # noqa: BLE001
            msgs.append(f"app.frontend.streamlit_app: compile error {exc!r}")
    finally:
        sys.stderr = saved_stderr
        devnull.close()
    return ("Streamlit imports", "PASS" if not msgs else "FAIL", msgs)


# ── 4) API health ─────────────────────────────────────────────────────────────

_API_HOST = "127.0.0.1"


def _free_port() -> int:
    """Return an OS-assigned free TCP port."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _port_open(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) == 0


def check_api_health() -> CheckResult:
    msgs: List[str] = []
    proc = None
    api_port = _free_port()
    try:
        cmd = [
            sys.executable, "-m", "uvicorn", "app.agent.api:app",
            "--host", _API_HOST, "--port", str(api_port), "--log-level", "warning",
        ]
        proc = subprocess.Popen(
            cmd, cwd=str(_ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Wait up to 15s (Windows startup can be slow)
        for _ in range(60):
            if _port_open(_API_HOST, api_port):
                break
            time.sleep(0.25)
        else:
            msgs.append("uvicorn did not bind within 15s")
            return ("API health", "FAIL", msgs)

        # Hit /health.
        url = f"http://{_API_HOST}:{api_port}/health"
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status != 200:
                msgs.append(f"/health returned status {resp.status}")
                return ("API health", "FAIL", msgs)
            import json as _json
            payload = _json.loads(resp.read())

        for f in ("status", "demo_stories", "tables"):
            if f not in payload:
                msgs.append(f"/health response missing field {f!r}")
        if payload.get("status") != "ok":
            msgs.append(f"/health status field is {payload.get('status')!r} (expected 'ok')")

    except (urllib.error.URLError, Exception) as exc:  # noqa: BLE001
        msgs.append(f"crashed: {exc!r}")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()

    return ("API health", "PASS" if not msgs else "FAIL", msgs)


# ── 5) Env config ─────────────────────────────────────────────────────────────

def check_env_config() -> CheckResult:
    """Returns PASS / WARN / FAIL — missing API key is a WARN, not a FAIL."""
    msgs: List[str] = []
    status = "PASS"

    env_path = _ROOT / ".env"
    if not env_path.exists():
        msgs.append(".env file not found at project root (demo mode will be active)")
        status = "WARN"

    if not os.getenv("ANTHROPIC_API_KEY"):
        # Try parsing .env manually to catch the case where it's set there
        # but the shell didn't load it.
        loaded_in_env = False
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("ANTHROPIC_API_KEY"):
                        loaded_in_env = True
                        break
            except Exception:
                pass
        if not loaded_in_env:
            msgs.append("ANTHROPIC_API_KEY not set -- agent will run in demo mode")
        else:
            msgs.append("ANTHROPIC_API_KEY in .env but not exported -- run.bat / run.sh will load it")
        if status != "FAIL":
            status = "WARN"

    try:
        from configs import settings
        db_url = str(settings.DATABASE_URL)
        if "sqlite" not in db_url and "postgresql" not in db_url:
            msgs.append(f"DATABASE_URL looks unusual: {db_url[:60]}...")
            status = "WARN"
    except Exception as exc:  # noqa: BLE001
        msgs.append(f"could not read DATABASE_URL: {exc!r}")
        status = "FAIL"

    return ("Env config", status, msgs)


# ── Composer ──────────────────────────────────────────────────────────────────

ALL_CHECKS = [
    check_database_integrity,
    check_demo_data,
    check_streamlit_imports,
    check_api_health,
    check_env_config,
]


def run_all() -> List[CheckResult]:
    results: List[CheckResult] = []
    for fn in ALL_CHECKS:
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001
            results.append((fn.__name__, "FAIL", [f"crashed: {exc!r}"]))
    return results


def print_results(results: List[CheckResult]) -> None:
    for name, status, msgs in results:
        if status == "PASS":
            print(f"  {green('PASS')}  {name}")
        elif status == "WARN":
            print(f"  {yellow('WARN')}  {name}")
            for m in msgs:
                print(f"          {yellow('-')} {m}")
        else:
            print(f"  {red('FAIL')}  {name}")
            for m in msgs:
                print(f"          {red('-')} {m}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print()
    print(bold(cyan("=" * 72)))
    print(bold(cyan(" RELIABILITY CHECK -- AI Quality Inspection Copilot")))
    print(bold(cyan("=" * 72)))
    print()

    started = time.perf_counter()
    results = run_all()
    print_results(results)

    elapsed = time.perf_counter() - started
    fails = sum(1 for _, s, _ in results if s == "FAIL")
    warns = sum(1 for _, s, _ in results if s == "WARN")

    print()
    print(bold(cyan("=" * 72)))
    print(f" {len(results) - fails - warns}/{len(results)} pass | "
          f"{warns} warn | {fails} fail | {elapsed:.1f}s")
    print(bold(cyan("=" * 72)))

    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
