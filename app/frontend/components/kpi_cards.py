"""
Reusable KPI card components for the Quality Agent Streamlit UI.
Phase 4 Step 4.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from app.frontend.theme import COLORS, RISK_COLORS, TIER_COLORS


# ── Risk badge ────────────────────────────────────────────────────────────────

def render_risk_badge(risk_tier: str) -> str:
    """Return an HTML <span> badge colored by risk tier."""
    tier  = str(risk_tier).upper()
    bg    = RISK_COLORS.get(tier, "#8B949E")
    fg    = "#0E1117" if tier == "LOW" else "#fff"
    return (
        f'<span style="background:{bg};color:{fg};border-radius:999px;'
        f'padding:2px 10px;font-size:0.72rem;font-weight:700;'
        f'letter-spacing:0.03em;">{tier}</span>'
    )


# ── Metric card ───────────────────────────────────────────────────────────────

def render_metric_card(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    color: Optional[str] = None,
) -> None:
    """Render a styled st.metric tile."""
    if color:
        st.markdown(
            f'<style>[data-testid="stMetricValue"]{{color:{color}!important}}</style>',
            unsafe_allow_html=True,
        )
    if delta is not None:
        st.metric(label=label, value=value, delta=delta)
    else:
        st.metric(label=label, value=value)


# ── Alert banner ──────────────────────────────────────────────────────────────

def render_alert_banner(message: str, level: str = "high") -> str:
    """Return an HTML alert div using theme CSS classes."""
    level_clean = level.lower().strip()
    css_class   = f"alert-{level_clean}"
    return f'<div class="{css_class}">{message}</div>'


# ── Supplier mini card ────────────────────────────────────────────────────────

_COO_FLAGS: Dict[str, str] = {
    "China":   "🇨🇳",
    "Germany": "🇩🇪",
    "Japan":   "🇯🇵",
    "USA":     "🇺🇸",
    "Mexico":  "🇲🇽",
    "UK":      "🇬🇧",
    "France":  "🇫🇷",
    "India":   "🇮🇳",
}


def render_supplier_mini_card(supplier_data: Dict[str, Any]) -> None:
    """Compact supplier card: name | tier badge | quality score | COO flag."""
    name  = str(supplier_data.get("supplier") or supplier_data.get("supplier_name") or "—")
    tier  = str(supplier_data.get("tier") or "").upper()
    qs    = supplier_data.get("quality_score")
    coo   = str(supplier_data.get("coo") or "")
    flag  = _COO_FLAGS.get(coo, "🌐")
    bg    = TIER_COLORS.get(tier, "#8B949E")
    qs_s  = f"{float(qs):.0f}/100" if qs is not None and str(qs) != "nan" else "n/a"

    st.markdown(
        f"""
        <div style="background:#21262D;border:1px solid #30363D;border-radius:8px;
                    padding:0.75rem 1rem;margin-bottom:0.5rem;display:flex;
                    align-items:center;gap:0.75rem;">
          <div style="flex:1;">
            <div style="font-weight:700;color:{COLORS['text_primary']};font-size:0.9rem;">{name}</div>
            <div style="font-size:0.75rem;color:{COLORS['text_secondary']};margin-top:2px;">
              QS: {qs_s} &nbsp;|&nbsp; {flag} {coo}
            </div>
          </div>
          <span style="background:{bg};color:#fff;border-radius:999px;
                       padding:2px 10px;font-size:0.7rem;font-weight:700;">{tier or "—"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Lot risk card ─────────────────────────────────────────────────────────────

def render_lot_risk_card(lot_data: Dict[str, Any]) -> None:
    """Big colored risk score card with Investigate button."""
    lot_no    = str(lot_data.get("lot_no") or "—")
    score     = lot_data.get("lot_risk_score")
    tier      = str(lot_data.get("risk_tier") or "UNKNOWN").upper()
    component = str(lot_data.get("component") or lot_data.get("component_id") or "—")
    supplier  = str(lot_data.get("supplier")  or lot_data.get("supplier_id")  or "—")
    color     = RISK_COLORS.get(tier, "#8B949E")
    fg        = "#0E1117" if tier == "LOW" else "#fff"
    score_s   = f"{float(score):.3f}" if score is not None else "—"

    st.markdown(
        f"""
        <div style="background:#21262D;border:2px solid {color};border-radius:10px;
                    padding:1rem;margin-bottom:0.5rem;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div style="font-size:0.68rem;color:#484F58;text-transform:uppercase;
                          letter-spacing:0.08em;margin-bottom:2px;">Lot Risk</div>
              <div style="font-size:1.6rem;font-weight:900;color:{color};line-height:1;">
                {score_s}
              </div>
              <div style="margin-top:4px;">
                <span style="background:{color};color:{fg};border-radius:999px;
                             padding:2px 10px;font-size:0.7rem;font-weight:700;">{tier}</span>
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:1rem;font-weight:700;color:{COLORS['text_primary']};">{lot_no}</div>
              <div style="font-size:0.75rem;color:{COLORS['text_secondary']};margin-top:2px;">{component}</div>
              <div style="font-size:0.75rem;color:{COLORS['text_muted']};">{supplier}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(f"Investigate {lot_no} →", key=f"inv_{lot_no}", use_container_width=True):
        st.session_state.current_lot   = lot_no
        st.session_state.active_screen = "Lot Drill-Down"
        st.rerun()
