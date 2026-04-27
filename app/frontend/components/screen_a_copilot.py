"""
Screen A: AI Copilot — two-column chat + context panel.
Phase 4 Step 2.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from app.agent.mock_responder import render_mock_response
from app.frontend.theme import COLORS, RISK_COLORS, TIER_COLORS

_ROOT = Path(__file__).resolve().parents[4]

_SUGGESTIONS: List[tuple[str, str]] = [
    ("🔴", "What is the risk level of lot L-778?"),
    ("⚠️", "Any process drift on production lines?"),
    ("🏆", "Compare SUP-A vs SUP-B for safety-critical builds"),
    ("🔍", "Why did serial SR20260008 fail in the field?"),
    ("📋", "Where should I focus inspection this week?"),
]

_SAP_PATTERN = re.compile(r"\b(QA32|QM01|QE51N|ME57|MB51)\b")
_SAP_HIGHLIGHT = (
    '<span style="background:rgba(210,153,34,0.2);color:#D29922;'
    'border-radius:3px;padding:1px 4px;font-weight:600;'
    'font-family:monospace;">{}</span>'
)


def _highlight_sap_codes(text: str) -> str:
    """Wrap SAP transaction codes in amber highlight spans."""
    return _SAP_PATTERN.sub(lambda m: _SAP_HIGHLIGHT.format(m.group()), text)


def _time_ago(ts: datetime) -> str:
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    diff = int((now - ts).total_seconds())
    if diff < 5:
        return "just now"
    if diff < 60:
        return f"{diff}s ago"
    if diff < 3600:
        return f"{diff // 60}m ago"
    return f"{diff // 3600}h ago"


# ── Internals ─────────────────────────────────────────────────────────────────

def _handle_question(question: str, registry, engine, api_key_present: bool) -> None:
    clean = question.strip()
    if not clean:
        return
    st.session_state.chat_history.append({
        "role":    "user",
        "content": clean,
        "ts":      datetime.now(timezone.utc),
    })

    agent = st.session_state.get("agent")
    cache_hit = False
    ttft_ms: Optional[int] = None
    excel_agent = getattr(agent, "excel_agent", None) if agent is not None else None

    if excel_agent is not None:
        # Streaming path — render user message inline first (history already rendered above)
        with st.chat_message("user"):
            st.markdown(clean)
        try:
            meta: Dict[str, Any] = {}
            with st.chat_message("assistant"):
                full_text = st.write_stream(
                    excel_agent.ask_stream(
                        clean,
                        session_id=st.session_state.session_id,
                        meta=meta,
                    )
                )
            text        = full_text
            tools       = meta.get("tools", ["run_python"])
            suggestions = meta.get("suggestions", [])
            confidence  = None
            tokens      = meta.get("tokens", 0)
            cache_hit   = meta.get("cache_hit", False)
            ttft_ms     = meta.get("ttft_ms")
            try:
                from app.core.chat_store import get_chat_store
                get_chat_store().save_turn(
                    session_id=st.session_state.session_id,
                    question=clean,
                    response=full_text,
                    tokens=tokens,
                    cache_hit=cache_hit,
                    branch="excel_agent",
                )
            except Exception:
                pass
        except Exception as exc:
            text, tools, suggestions, confidence, tokens, ttft_ms = (
                f"Agent error: {exc}", [], [], None, 0, None
            )
    elif agent is not None:
        ttft_ms = None
        try:
            with st.spinner("Thinking..."):
                result = agent.ask(clean, session_id=st.session_state.session_id)
            text        = result.response_text
            tools       = [t if isinstance(t, str) else t.get("name", "") for t in (result.tools_called or [])]
            suggestions = result.follow_up_suggestions or []
            confidence  = None
            tokens      = getattr(result, "total_tokens", 0) or 0
            cache_hit   = getattr(result, "cache_hit", False)
            entities    = result.intent.entities if result.intent else {}
            if entities.get("lot_no"):
                st.session_state.current_lot = entities["lot_no"][0]
            if entities.get("supplier"):
                st.session_state.current_supplier = entities["supplier"][0]
        except Exception as exc:
            text, tools, suggestions, confidence, tokens = (
                f"Agent error: {exc}", [], [], None, 0
            )
    else:
        mock        = render_mock_response(clean, registry, engine, session_id=st.session_state.session_id)
        text        = mock["text"]
        tools       = mock.get("tools", [])
        suggestions = []
        confidence  = mock.get("confidence")
        tokens      = 0

    st.session_state.tokens_used = st.session_state.get("tokens_used", 0) + tokens
    st.session_state.chat_history.append({
        "role":        "agent",
        "content":     text,
        "tools":       tools,
        "confidence":  confidence,
        "suggestions": suggestions,
        "tokens":      tokens,
        "cache_hit":   cache_hit,
        "ttft_ms":     ttft_ms,
        "ts":          datetime.now(timezone.utc),
    })


def _render_chat_history() -> None:
    last_agent_turn = next(
        (t for t in reversed(st.session_state.chat_history) if t["role"] == "agent"),
        None,
    )
    for turn in st.session_state.chat_history:
        ts_str = _time_ago(turn["ts"]) if "ts" in turn else ""

        if turn["role"] == "user":
            with st.chat_message("user"):
                st.markdown(turn["content"])
                if ts_str:
                    st.markdown(
                        f"<div style='font-size:0.68rem;color:#484F58;margin-top:2px;'>{ts_str}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            with st.chat_message("assistant"):
                raw = turn["content"]
                st.markdown(raw)

                meta_parts: List[str] = []
                if ts_str:
                    meta_parts.append(ts_str)
                tok = turn.get("tokens", 0)
                if tok:
                    meta_parts.append(f"{tok:,} tokens")
                if turn.get("cache_hit"):
                    meta_parts.append("⚡ cached")
                ttft = turn.get("ttft_ms")
                if ttft is not None:
                    meta_parts.append(f"responded in {ttft / 1000:.1f}s")

                if meta_parts:
                    st.markdown(
                        f"<div style='font-size:0.68rem;color:#484F58;margin-top:2px;'>"
                        f"{' · '.join(meta_parts)}</div>",
                        unsafe_allow_html=True,
                    )

                if turn is last_agent_turn:
                    for s in turn.get("suggestions", []):
                        if st.button(s, key=f"fu_{hash(s)}", use_container_width=False):
                            st.session_state.pending_followup = s

    if st.session_state.get("pending_followup"):
        q = st.session_state.pop("pending_followup")
        st.session_state.chat_history.append({
            "role": "user", "content": q, "ts": datetime.now(timezone.utc),
        })
        st.rerun()


def _render_context_panel(registry, engine) -> None:
    from sqlalchemy import text as _text

    st.markdown(
        f"<h4 style='color:{COLORS['text_secondary']};margin:0 0 0.75rem;'>Context Panel</h4>",
        unsafe_allow_html=True,
    )

    # Quick Stats
    risk      = registry.kpi.get_lot_risk_scores()
    drift_df  = registry.kpi.get_drift_signals()
    rankings  = registry.kpi.get_supplier_rankings()
    high_lots = int((risk["risk_tier"] == "HIGH").sum())
    watchlist = int((rankings["tier"].fillna("").str.lower() == "watchlist").sum())
    try:
        with engine.connect() as conn:
            claim_count = conn.execute(_text("SELECT COUNT(*) FROM fact_warranty_claims")).scalar() or 0
    except Exception:
        claim_count = 0

    # Token usage display
    tokens_used = st.session_state.get("tokens_used", 0)
    if tokens_used:
        st.markdown(
            f"<div style='font-size:0.72rem;color:#484F58;margin-bottom:0.5rem;'>"
            f"Tokens used this session: {tokens_used:,}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("**📊 Quick Stats**")
    ca, cb = st.columns(2)
    ca.metric("HIGH Risk Lots",      high_lots)
    cb.metric("Drift Signals",       len(drift_df))
    cc, cd = st.columns(2)
    cc.metric("Watchlist Suppliers", watchlist)
    cd.metric("Warranty Claims",     int(claim_count))
    st.markdown("---")

    # Current lot card
    current_lot = st.session_state.get("current_lot")
    if current_lot:
        lot_row = risk[risk["lot_no"] == current_lot]
        if not lot_row.empty:
            r     = lot_row.iloc[0]
            tier  = str(r["risk_tier"])
            color = RISK_COLORS.get(tier, "#8B949E")
            txt   = "#0E1117" if tier == "LOW" else "#fff"
            st.markdown(
                f"""
                <div style="background:#21262D;border:1px solid {color}55;
                            border-radius:8px;padding:0.85rem;margin-bottom:1rem;">
                  <div style="font-size:0.68rem;color:#484F58;text-transform:uppercase;
                              letter-spacing:0.06em;margin-bottom:0.35rem;">Current Lot</div>
                  <div style="font-size:1.05rem;font-weight:700;color:{COLORS['text_primary']};">{current_lot}</div>
                  <div style="margin-top:0.35rem;">
                    <span style="background:{color};color:{txt};border-radius:999px;
                                 padding:2px 10px;font-size:0.7rem;font-weight:700;">{tier}</span>
                    <span style="font-size:0.78rem;color:{COLORS['text_secondary']};
                                 margin-left:0.5rem;">score {r['lot_risk_score']:.3f}</span>
                  </div>
                  <div style="font-size:0.75rem;color:{COLORS['text_secondary']};margin-top:0.35rem;">
                    Fail rate: {float(r['fail_rate']):.1%}&nbsp;|&nbsp;Claims: {int(r['claims_linked'])}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Current supplier card
    current_sup = st.session_state.get("current_supplier")
    if current_sup:
        sup_row = rankings[rankings["supplier"].str.contains(current_sup, case=False, na=False)]
        if not sup_row.empty:
            r     = sup_row.iloc[0]
            tier  = str(r.get("tier") or "")
            color = TIER_COLORS.get(tier.upper(), "#8B949E")
            qs    = r.get("quality_score")
            qs_s  = f"{float(qs):.0f}" if pd.notna(qs) else "n/a"
            fr    = r.get("incoming_fail_rate", 0)
            fr_s  = f"{float(fr):.2%}" if pd.notna(fr) else "n/a"
            st.markdown(
                f"""
                <div style="background:#21262D;border:1px solid {color}55;
                            border-radius:8px;padding:0.85rem;margin-bottom:1rem;">
                  <div style="font-size:0.68rem;color:#484F58;text-transform:uppercase;
                              letter-spacing:0.06em;margin-bottom:0.35rem;">Current Supplier</div>
                  <div style="font-size:1.05rem;font-weight:700;color:{COLORS['text_primary']};">{current_sup}</div>
                  <div style="margin-top:0.35rem;">
                    <span style="background:{color};color:#fff;border-radius:999px;
                                 padding:2px 10px;font-size:0.7rem;font-weight:700;">{tier}</span>
                    <span style="font-size:0.78rem;color:{COLORS['text_secondary']};
                                 margin-left:0.5rem;">QS: {qs_s}/100</span>
                  </div>
                  <div style="font-size:0.75rem;color:{COLORS['text_secondary']};margin-top:0.35rem;">
                    COO: {r.get('coo', 'n/a')}&nbsp;|&nbsp;Fail rate: {fr_s}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Active alerts
    st.markdown("**🎯 Active Alerts**")
    for lot_no in risk[risk["risk_tier"] == "HIGH"]["lot_no"].tolist()[:5]:
        st.markdown(
            f'<div class="alert-high"><strong>{lot_no}</strong> — HIGH risk lot</div>',
            unsafe_allow_html=True,
        )
    for _, row in drift_df.iterrows():
        st.markdown(
            f'<div class="alert-medium"><strong>{row["line"]} {row["shift"]}</strong>'
            f' — {float(row["torque_fail_rate"]):.1%} torque fail rate</div>',
            unsafe_allow_html=True,
        )


# ── Public entry point ────────────────────────────────────────────────────────

def render_copilot_screen(registry, engine, api_key_present: bool) -> None:
    chat_col, ctx_col = st.columns([65, 35])

    with chat_col:
        # Header row with title + clear button
        title_col, btn_col = st.columns([5, 1])
        with title_col:
            st.markdown(
                f"""
                <div style="margin-bottom:0.5rem;">
                  <h2 style="color:{COLORS['text_primary']};margin:0 0 4px;">🤖 AI Quality Copilot</h2>
                  <p style="color:{COLORS['text_secondary']};font-size:0.85rem;margin:0;">
                    Ask anything about lots, suppliers, process quality, or field failures
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with btn_col:
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            if st.button("🔄 Clear", key="clear_chat_top", help="Clear conversation"):
                st.session_state.chat_history    = []
                st.session_state.current_lot     = None
                st.session_state.current_supplier = None
                st.session_state.tokens_used     = 0
                st.rerun()

        if not api_key_present:
            st.info(
                "Running in **demo mode** — responses use real database data without the Claude API. "
                "Set `ANTHROPIC_API_KEY` to enable full AI reasoning.",
                icon="ℹ️",
            )

        # Suggested questions (only when history is empty)
        if not st.session_state.chat_history:
            st.markdown(
                '<div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;'
                'letter-spacing:0.08em;color:#484F58;margin-bottom:0.5rem;">Suggested Questions</div>',
                unsafe_allow_html=True,
            )
            for emoji, q in _SUGGESTIONS:
                label = f"{emoji} {q}"
                if st.button(label, key=f"sugg_{hash(q)}", use_container_width=True):
                    _handle_question(label, registry, engine, api_key_present)
                    st.rerun()
            st.markdown("---")

        # Auto-submit from sidebar alert clicks
        pending = st.session_state.pop("pending_question", None)
        if pending:
            _handle_question(pending, registry, engine, api_key_present)

        _render_chat_history()

        if question := st.chat_input("Ask about lots, suppliers, process quality..."):
            _handle_question(question, registry, engine, api_key_present)
            st.rerun()

    with ctx_col:
        _render_context_panel(registry, engine)
