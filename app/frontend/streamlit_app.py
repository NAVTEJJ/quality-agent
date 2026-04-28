"""
AI Quality Inspection Copilot — Streamlit frontend entry point.

Run with:
    streamlit run app/frontend/streamlit_app.py
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.frontend.theme import CSS, COLORS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Quality Inspection Copilot",
    page_icon="mag",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS, unsafe_allow_html=True)


# ── Session-state defaults ────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "chat_history":     [],
    "current_lot":      None,
    "current_supplier": None,
    "active_screen":    "AI Copilot",
    "db_engine":        None,
    "registry":         None,
    "agent":            None,
    "api_key_present":  bool(os.getenv("ANTHROPIC_API_KEY")),
    "session_id":       None,
    "pending_question": None,
    "sidebar_open":     True,   # owned by us, not Streamlit
    "tokens_used":      0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── DB / agent bootstrap ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to database...")
def _load_registry():
    from configs import settings
    from app.models.schema import get_engine, init_database
    from app.ingestion.loader import load_all_sheets
    from app.ingestion.normalizer import NormalizationPipeline
    from app.services.service_registry import get_registry

    engine = get_engine(str(settings.DATABASE_URL))
    from sqlalchemy import inspect as _inspect
    if not _inspect(engine).get_table_names():
        init_database(engine)
        sheets = load_all_sheets(settings.EXCEL_PATH)
        NormalizationPipeline().run_full_pipeline(sheets, engine)
    return engine, get_registry(engine)


@st.cache_resource(show_spinner="Initialising AI agent...")
def _load_agent(_engine, _registry):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        from app.agent.agent_core import QualityAgent
        return QualityAgent(registry=_registry, engine=_engine)
    except Exception:
        return None


def _ensure_loaded():
    if st.session_state.db_engine is None:
        engine, registry = _load_registry()
        st.session_state.db_engine = engine
        st.session_state.registry  = registry
    if st.session_state.agent is None:
        st.session_state.agent = _load_agent(
            st.session_state.db_engine,
            st.session_state.registry,
        )
    if st.session_state.session_id is None:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        # ── Collapse button (top-right of sidebar) ─────────────────────────
        _, btn_col = st.columns([5, 1])
        with btn_col:
            if st.button("◀", key="collapse_sidebar", help="Hide sidebar"):
                st.session_state.sidebar_open = False
                st.rerun()

        # ── Pulsing alert dot ──────────────────────────────────────────────
        _reg     = st.session_state.registry
        _risk_df = None
        if _reg is not None:
            try:
                _risk_df = _reg.kpi.get_lot_risk_scores()
                _n_high  = int((_risk_df["risk_tier"] == "HIGH").sum())
            except Exception:
                _n_high = 0
        else:
            _n_high = 0

        if _n_high > 0:
            st.markdown(
                f"""
                <style>
                @keyframes pulse {{
                  0%   {{ opacity:1; }}
                  50%  {{ opacity:0.35; }}
                  100% {{ opacity:1; }}
                }}
                .pulse-dot {{
                  display:inline-block;width:9px;height:9px;border-radius:50%;
                  background:#F85149;margin-right:6px;
                  animation:pulse 1.4s ease-in-out infinite;
                }}
                </style>
                <div style="background:rgba(248,81,73,0.1);border:1px solid #F85149;
                            border-radius:6px;padding:0.4rem 0.6rem;margin-bottom:0.75rem;">
                  <span class="pulse-dot"></span>
                  <span style="font-size:0.8rem;color:#F85149;font-weight:600;">
                    {_n_high} Active Alert{"s" if _n_high != 1 else ""}
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Demo mode banner ───────────────────────────────────────────────
        if not st.session_state.api_key_present:
            st.markdown(
                f"""
                <div style="background:rgba(88,166,255,0.08);border:1px solid #58A6FF44;
                            border-radius:6px;padding:0.5rem 0.6rem;margin-bottom:0.75rem;">
                  <div style="font-size:0.78rem;color:#58A6FF;font-weight:600;">
                    Demo Mode
                  </div>
                  <div style="font-size:0.72rem;color:{COLORS['text_secondary']};margin-top:2px;line-height:1.4;">
                    Real data, simulated AI responses.<br>
                    Set <code>ANTHROPIC_API_KEY</code> in <code>.env</code>
                    to enable live Claude AI.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Brand header
        st.markdown(
            f"""
            <div style="padding:0 0 1rem 0;">
              <div style="font-size:1.15rem;font-weight:700;color:{COLORS['text_primary']};">
                Quality Copilot
              </div>
              <div style="font-size:0.72rem;color:{COLORS['text_muted']};margin-top:2px;">
                AI Quality Inspection Agent
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)

        screens = [
            ("mag_right",                    "AI Copilot"),
            ("bar_chart",                    "Quality Dashboard"),
            ("microscope",                   "Lot Drill-Down"),
            ("chart_with_upwards_trend",     "Analytics"),
        ]
        for icon, label in screens:
            is_active = st.session_state.active_screen == label
            if st.button(
                f":{icon}: {label}",
                key=f"nav_{label}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.active_screen = label
                st.rerun()

        st.markdown("---")

        # ── Live status panel ──────────────────────────────────────────────
        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)

        reg = st.session_state.registry
        if reg is not None:
            try:
                from sqlalchemy import text
                with st.session_state.db_engine.connect() as conn:
                    lot_count = conn.execute(
                        text("SELECT COUNT(*) FROM fact_incoming_qm")
                    ).scalar() or 0

                st.markdown(
                    f'<span class="status-dot-green"></span>'
                    f'<span style="font-size:0.8rem;color:{COLORS["accent_green"]};">DB Connected</span>',
                    unsafe_allow_html=True,
                )
                st.metric("Inspections", f"{lot_count:,}")

            except Exception as exc:
                st.markdown(
                    f'<span class="status-dot-red"></span>'
                    f'<span style="font-size:0.8rem;color:{COLORS["accent_red"]};">DB Error</span>',
                    unsafe_allow_html=True,
                )
                st.caption(str(exc)[:120])
        else:
            st.markdown(
                f'<span class="status-dot-yellow"></span>'
                f'<span style="font-size:0.8rem;color:{COLORS["accent_yellow"]};">Loading...</span>',
                unsafe_allow_html=True,
            )

        if st.session_state.api_key_present:
            st.markdown(
                f'<span class="status-dot-green"></span>'
                f'<span style="font-size:0.8rem;color:{COLORS["accent_green"]};">Claude API Ready</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="status-dot-yellow"></span>'
                f'<span style="font-size:0.8rem;color:{COLORS["accent_yellow"]};">'
                f'No API Key — demo mode</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Chat History ───────────────────────────────────────────────────
        try:
            from app.core.chat_store import get_chat_store
            _store   = get_chat_store()
            _recent  = _store.get_recent_questions(n=15)
            _total_q = _store.total_questions()
            _total_s = _store.total_sessions()
        except Exception:
            _recent = []; _total_q = 0; _total_s = 0

        with st.expander(
            f"Chat History  ({_total_q} questions · {_total_s} sessions)",
            expanded=False,
        ):
            if not _recent:
                st.caption("No history yet — start asking questions.")
            else:
                # Search box
                _search = st.text_input(
                    "Search history", placeholder="e.g. L-778, SUP-C ...",
                    key="history_search", label_visibility="collapsed",
                )
                _hits = (
                    [r for r in _recent if _search.lower() in r["question"].lower()]
                    if _search else _recent
                )
                for item in _hits[:10]:
                    _ts = item["ts"][:16].replace("T", " ")
                    _q  = item["question"]
                    _label = (_q[:55] + "...") if len(_q) > 55 else _q
                    col_q, col_re = st.columns([5, 1])
                    with col_q:
                        st.markdown(
                            f"<div style='font-size:0.72rem;color:#8B949E;'>{_ts}</div>"
                            f"<div style='font-size:0.78rem;color:#E6EDF3;"
                            f"margin-bottom:4px;'>{_label}</div>",
                            unsafe_allow_html=True,
                        )
                    with col_re:
                        if st.button("Re-ask", key=f"rask_{hash(_q+_ts)}"):
                            _quick_ask(_q)

        st.markdown("---")

        # ── Active alerts ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">Active Alerts</div>', unsafe_allow_html=True)

        try:
            if _reg is not None and _risk_df is not None:
                _high_lots = _risk_df[_risk_df["risk_tier"] == "HIGH"]["lot_no"].tolist()[:3]
                _drift_df  = _reg.kpi.get_drift_signals()
                if not _high_lots and _drift_df.empty:
                    st.caption("No active alerts.")
                for _lot in _high_lots:
                    _row    = _risk_df[_risk_df["lot_no"] == _lot].iloc[0]
                    _claims = int(_row.get("claims_linked", 0))
                    _msg    = f"HIGH risk — {_claims} claim{'s' if _claims != 1 else ''}"
                    if st.button(_lot, key=f"alert_{_lot}", use_container_width=True):
                        _quick_ask(f"Tell me about lot {_lot}")
                    st.markdown(
                        f'<div class="alert-high"><strong>{_lot}</strong> &mdash; {_msg}</div>',
                        unsafe_allow_html=True,
                    )
                for _, _drow in _drift_df.iterrows():
                    _entity = f"{_drow['line']} {_drow['shift']}"
                    _tfr    = float(_drow["torque_fail_rate"])
                    _msg    = f"{_tfr:.1%} torque drift"
                    if st.button(_entity, key=f"alert_{_entity}", use_container_width=True):
                        _quick_ask(f"Tell me about {_entity}")
                    st.markdown(
                        f'<div class="alert-medium"><strong>{_entity}</strong> &mdash; {_msg}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Alerts load after database connects.")
        except Exception:
            st.caption("Could not load alerts.")

        st.markdown("---")
        st.markdown(
            f"""
            <div style="font-size:0.7rem;color:{COLORS['text_muted']};text-align:center;line-height:1.7;">
              Quality Agent v1.0<br>
              <span style="font-size:0.65rem;opacity:0.6;">Press <kbd style="background:#30363D;
              border-radius:3px;padding:1px 4px;font-size:0.65rem;">R</kbd> to refresh data</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _quick_ask(question: str):
    st.session_state.active_screen = "AI Copilot"
    st.session_state.pending_question = question
    st.rerun()


# ── Screen renderers ──────────────────────────────────────────────────────────

def _render_ai_copilot():
    from app.frontend.components.screen_a_copilot import render_copilot_screen
    render_copilot_screen(
        registry=st.session_state.registry,
        engine=st.session_state.db_engine,
        api_key_present=st.session_state.api_key_present,
    )


def _render_quality_dashboard():
    from app.frontend.components.screen_b_dashboard import render_dashboard_screen
    render_dashboard_screen(
        registry=st.session_state.registry,
        engine=st.session_state.db_engine,
    )


def _render_lot_drill_down():
    from app.frontend.components.screen_c_drilldown import render_drilldown_screen
    render_drilldown_screen(
        registry=st.session_state.registry,
        engine=st.session_state.db_engine,
    )


def _render_analytics():
    from app.frontend.components.screen_d_analytics import render_analytics_screen
    render_analytics_screen(
        registry=st.session_state.registry,
        engine=st.session_state.db_engine,
    )


_SCREEN_MAP = {
    "AI Copilot":        _render_ai_copilot,
    "Quality Dashboard": _render_quality_dashboard,
    "Lot Drill-Down":    _render_lot_drill_down,
    "Analytics":         _render_analytics,
}

# ── CSS to hide the native sidebar when sidebar_open = False ─────────────────
_HIDE_SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}
.main .block-container {
    padding-left: 2rem !important;
}
</style>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _ensure_loaded()

    if st.session_state.sidebar_open:
        _render_sidebar()
        screen_fn = _SCREEN_MAP.get(st.session_state.active_screen, _render_ai_copilot)
        screen_fn()
    else:
        # Sidebar hidden: suppress it with CSS, then render expand arrow + screen.
        st.markdown(_HIDE_SIDEBAR_CSS, unsafe_allow_html=True)

        # Expand arrow pinned to far left in a narrow column.
        arrow_col, content_col = st.columns([1, 24])
        with arrow_col:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if st.button("▶", key="expand_sidebar", help="Show sidebar"):
                st.session_state.sidebar_open = True
                st.rerun()
        with content_col:
            screen_fn = _SCREEN_MAP.get(st.session_state.active_screen, _render_ai_copilot)
            screen_fn()


if __name__ == "__main__":
    main()
