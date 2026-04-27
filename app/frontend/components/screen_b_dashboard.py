"""
Screen B: Quality Dashboard — KPI cards, Plotly charts, data tables.
Phase 4 Step 2.
"""
from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text as _text

from app.frontend.theme import COLORS, RISK_COLORS, TIER_COLORS

_CHART_LAYOUT = dict(
    paper_bgcolor="#1A1D27",
    plot_bgcolor="#1A1D27",
    font=dict(color="#8B949E", size=11),
    margin=dict(t=24, b=20, l=20, r=20),
    showlegend=False,
)

_GRID = dict(gridcolor="#30363D", zerolinecolor="#30363D")

# CSS for pulsing L-778 row highlight
_PULSE_CSS = """
<style>
@keyframes row-pulse {
  0%   { background-color: rgba(248,81,73,0.22); }
  50%  { background-color: rgba(248,81,73,0.06); }
  100% { background-color: rgba(248,81,73,0.22); }
}
.l778-pulse {
  animation: row-pulse 2s ease-in-out infinite;
  border-left: 3px solid #F85149;
  padding-left: 6px;
  border-radius: 3px;
  font-weight: 700;
  color: #F85149;
}
</style>
"""


# ── Small helpers ─────────────────────────────────────────────────────────────

def _pct(v) -> str:
    return f"{float(v):.1%}" if pd.notna(v) else "n/a"

def _pct2(v) -> str:
    return f"{float(v):.2%}" if pd.notna(v) else "n/a"

def _score(v) -> str:
    return f"{float(v):.0f}" if pd.notna(v) else "n/a"

def _risk_color(tier: str) -> str:
    return RISK_COLORS.get(str(tier).upper(), "#8B949E")

def _tier_color(tier: str) -> str:
    return TIER_COLORS.get(str(tier).upper(), "#8B949E")


def _bar(
    x: list, y: list, colors: list, texts: list,
    height: int = 200, hline: float | None = None,
    hline_label: str = "", ytitle: str = "",
) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=x, y=y,
        marker_color=colors,
        text=texts, textposition="outside",
        cliponaxis=False,
    ))
    if hline is not None:
        fig.add_hline(
            y=hline, line_dash="dash", line_color="#D29922",
            annotation_text=hline_label,
            annotation_font_color="#D29922",
        )
    fig.update_layout(
        **_CHART_LAYOUT,
        height=height,
        yaxis=dict(title=ytitle, **_GRID),
        xaxis=dict(**_GRID),
    )
    return fig


# ── Row helpers ───────────────────────────────────────────────────────────────

def _row1_kpis(risk: pd.DataFrame, drift_flagged: pd.DataFrame,
               rankings: pd.DataFrame, lot_count: int) -> None:
    high_lots = int((risk["risk_tier"] == "HIGH").sum())
    medium_lots = int((risk["risk_tier"] == "MEDIUM").sum())
    watchlist   = int((rankings["tier"].fillna("").str.lower() == "watchlist").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Lots Monitored", int(lot_count))
    c2.metric("🔴 HIGH Risk Lots",    high_lots,
              delta=f"+{high_lots} need action", delta_color="inverse")
    c3.metric("⚠️ Drift Signals",     len(drift_flagged),
              delta="LINE-2 Night" if len(drift_flagged) else None, delta_color="inverse")
    c4.metric("Watchlist Suppliers",  watchlist,
              delta="SUP-C" if watchlist else None, delta_color="inverse")


def _row2_risk_drift(risk: pd.DataFrame, drift_all: pd.DataFrame, engine) -> None:
    left, right = st.columns(2)

    # ── Left: lot risk table ──────────────────────────────────────────────
    with left:
        st.markdown(
            f"<h4 style='color:{COLORS['text_primary']};margin:0 0 0.5rem;'>🔴 Risk Lot Ranking</h4>",
            unsafe_allow_html=True,
        )
        # Inject pulse CSS once
        st.markdown(_PULSE_CSS, unsafe_allow_html=True)

        try:
            comp_map = pd.read_sql(
                "SELECT component_id, component_name FROM dim_component", engine
            ).set_index("component_id")["component_name"].to_dict()
            sup_map = pd.read_sql(
                "SELECT supplier_id, supplier_name FROM dim_supplier", engine
            ).set_index("supplier_id")["supplier_name"].to_dict()
        except Exception:
            comp_map, sup_map = {}, {}

        top10 = risk.head(10).copy()
        top10["Component"] = top10["component_id"].map(comp_map).fillna(top10["component_id"].astype(str))
        top10["Supplier"]  = top10["supplier_id"].map(sup_map).fillna(top10["supplier_id"].astype(str))

        # Render L-778 as pulsing HTML row, rest as dataframe
        l778_mask = top10["lot_no"] == "L-778"
        if l778_mask.any():
            r778 = top10[l778_mask].iloc[0]
            st.markdown(
                f'<div class="l778-pulse" style="margin-bottom:4px;padding:6px 8px;">'
                f'⚡ <strong>L-778</strong> &nbsp;|&nbsp; {r778["Component"]} &nbsp;|&nbsp; '
                f'{r778["Supplier"]} &nbsp;|&nbsp; Score: {float(r778["lot_risk_score"]):.3f} &nbsp;|&nbsp; '
                f'Fail: {float(r778["fail_rate"]):.1%}'
                f'</div>',
                unsafe_allow_html=True,
            )

        display_rest = top10[~l778_mask].copy()
        display = pd.DataFrame({
            "Lot No":     display_rest["lot_no"],
            "Component":  display_rest["Component"],
            "Supplier":   display_rest["Supplier"],
            "Risk Score": display_rest["lot_risk_score"].apply(lambda x: f"{x:.3f}"),
            "Tier":       display_rest["risk_tier"],
            "Fail Rate":  display_rest["fail_rate"].apply(_pct),
        })

        def _lot_row_style(row):
            if row["Tier"] == "HIGH":
                return ["background-color:rgba(248,81,73,0.07)"] * len(row)
            if row["Tier"] == "MEDIUM":
                return ["background-color:rgba(210,153,34,0.07)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display.style.apply(_lot_row_style, axis=1),
            use_container_width=True,
            height=280,
        )

        sel = st.selectbox(
            "Drill into lot:", ["—"] + top10["lot_no"].tolist(), key="lot_drill_select"
        )
        if sel != "—":
            st.session_state.current_lot    = sel
            st.session_state.active_screen  = "Lot Drill-Down"
            st.rerun()

    # ── Right: drift table + chart ────────────────────────────────────────
    with right:
        st.markdown(
            f"<h4 style='color:{COLORS['text_primary']};margin:0 0 0.5rem;'>⚙️ Process Drift by Line & Shift</h4>",
            unsafe_allow_html=True,
        )
        drift_disp = drift_all.copy()
        drift_disp["Status"] = drift_disp["torque_fail_rate"].apply(
            lambda x: "🔴 DRIFT" if x > 0.10 else ("⚠️ WATCH" if x > 0.05 else "✅ OK")
        )

        def _drift_style(row):
            if "DRIFT" in str(row.get("Status", "")):
                return ["background-color:rgba(248,81,73,0.18)"] * len(row)
            if "WATCH" in str(row.get("Status", "")):
                return ["background-color:rgba(210,153,34,0.07)"] * len(row)
            return [""] * len(row)

        show = drift_disp[["line", "shift", "total_builds",
                            "torque_fail_rate", "leak_fail_rate", "Status"]].copy()
        show.columns = ["Line", "Shift", "Builds", "Torque Fail", "Leak Fail", "Status"]
        show["Torque Fail"] = show["Torque Fail"].apply(_pct)
        show["Leak Fail"]   = show["Leak Fail"].apply(_pct)

        st.dataframe(
            show.style.apply(_drift_style, axis=1),
            use_container_width=True,
            height=180,
        )

        labels  = [f"{r['line']} {r['shift']}" for _, r in drift_all.iterrows()]
        rates   = [float(r["torque_fail_rate"]) * 100 for _, r in drift_all.iterrows()]
        c_list  = [
            "#F85149" if v > 10 else "#D29922" if v > 5 else "#3FB950"
            for v in rates
        ]
        texts   = [f"{v:.1f}%" for v in rates]
        fig = _bar(labels, rates, c_list, texts,
                   hline=10, hline_label="10% threshold",
                   ytitle="Torque Fail %", height=190)
        st.plotly_chart(fig, use_container_width=True)


def _row3_supplier_coo(rankings: pd.DataFrame, coo_df: pd.DataFrame) -> None:
    left, right = st.columns(2)

    # ── Left: supplier scorecard ──────────────────────────────────────────
    with left:
        st.markdown(
            f"<h4 style='color:{COLORS['text_primary']};margin:0 0 0.5rem;'>🏭 Supplier Scorecard</h4>",
            unsafe_allow_html=True,
        )
        sup_disp = pd.DataFrame({
            "Supplier":   rankings["supplier"],
            "COO":        rankings["coo"],
            "Quality":    rankings["quality_score"].apply(_score),
            "Tier":       rankings["tier"].fillna("n/a"),
            "Fail Rate":  rankings["incoming_fail_rate"].apply(_pct2),
            "Premium":    rankings["premium_service_fit"].fillna("No"),
            "Rank":       rankings["composite_rank"],
        }).sort_values("Rank")

        def _sup_style(row):
            tier = str(row.get("Tier", ""))
            if tier.lower() == "watchlist":
                return ["background-color:rgba(248,81,73,0.12)"] * len(row)
            if tier.lower() == "preferred":
                return ["background-color:rgba(63,185,80,0.08)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            sup_disp.style.apply(_sup_style, axis=1),
            use_container_width=True,
            height=200,
        )

        # Quality score bar chart
        fig = _bar(
            x=rankings["supplier"].tolist(),
            y=rankings["quality_score"].fillna(0).tolist(),
            colors=[_tier_color(str(t)) for t in rankings["tier"].fillna("")],
            texts=[_score(v) for v in rankings["quality_score"].fillna(0)],
            hline=80, hline_label="Preferred >= 80",
            ytitle="Quality Score", height=190,
        )
        fig.update_layout(yaxis=dict(range=[0, 110], **_GRID))
        st.plotly_chart(fig, use_container_width=True)

    # ── Right: COO performance ────────────────────────────────────────────
    with right:
        st.markdown(
            f"<h4 style='color:{COLORS['text_primary']};margin:0 0 0.5rem;'>🌍 COO Performance</h4>",
            unsafe_allow_html=True,
        )
        coo_disp = pd.DataFrame({
            "Rank":         coo_df["rank"],
            "COO":          coo_df["coo"],
            "Incoming Fail": coo_df["coo_incoming_fail_rate"].apply(_pct2),
            "Warranty Rate": coo_df["coo_warranty_claim_rate"].apply(_pct2),
            "Samples":       coo_df["samples"].fillna(0).astype(int),
        })

        def _coo_style(row):
            v_raw = coo_df.loc[coo_df["coo"] == row["COO"], "coo_incoming_fail_rate"]
            v = float(v_raw.iloc[0]) if not v_raw.empty and pd.notna(v_raw.iloc[0]) else 0.0
            if v > 0.15:
                return ["background-color:rgba(248,81,73,0.12)"] * len(row)
            if v < 0.05:
                return ["background-color:rgba(63,185,80,0.08)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            coo_disp.style.apply(_coo_style, axis=1),
            use_container_width=True,
            height=200,
        )

        # COO bar chart — red for high fail rate, green for low
        _coo_pal = {
            "China": "#F85149", "Germany": "#3FB950", "Japan": "#3FB950",
            "USA": "#58A6FF",   "Mexico": "#D29922",
        }
        rates_coo = (coo_df["coo_incoming_fail_rate"].fillna(0) * 100).tolist()
        c_coo = [_coo_pal.get(c, "#58A6FF") for c in coo_df["coo"]]
        fig_coo = _bar(
            x=coo_df["coo"].tolist(),
            y=rates_coo,
            colors=c_coo,
            texts=[f"{v:.1f}%" for v in rates_coo],
            ytitle="Incoming Fail %", height=190,
        )
        st.plotly_chart(fig_coo, use_container_width=True)


def _row4_inspection_priority(strategy: dict) -> None:
    st.markdown(
        f"<h4 style='color:{COLORS['text_primary']};margin:0.5rem 0 0.5rem;'>📋 Inspection Priority List</h4>",
        unsafe_allow_html=True,
    )
    items = strategy.get("increase_sampling", [])
    if not items:
        st.info("No lots currently require increased inspection.")
        return

    rows = [
        {
            "Priority":           i,
            "Lot No":             item.get("lot_no", ""),
            "Component":          item.get("component") or "—",
            "Supplier":           item.get("supplier") or "—",
            "Risk Score":         f"{item.get('risk_score', 0):.3f}",
            "Tier":               item.get("risk_tier", ""),
            "Fail Rate":          f"{item.get('fail_rate', 0):.1%}",
            "Recommended Action": item.get("action", ""),
        }
        for i, item in enumerate(items, 1)
    ]
    prio_df = pd.DataFrame(rows)

    def _prio_style(row):
        if row["Lot No"] == "L-778":
            return ["background-color:rgba(248,81,73,0.18)"] * len(row)
        if row["Tier"] == "HIGH":
            return ["background-color:rgba(248,81,73,0.07)"] * len(row)
        if row["Tier"] == "MEDIUM":
            return ["background-color:rgba(210,153,34,0.07)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        prio_df.style.apply(_prio_style, axis=1),
        use_container_width=True,
        height=320,
    )

    reduce = strategy.get("reduce_inspection", [])
    if reduce:
        st.markdown(
            f"<h5 style='color:{COLORS['text_secondary']};margin:1rem 0 0.4rem;'>✅ Eligible for Reduced Inspection</h5>",
            unsafe_allow_html=True,
        )
        rd = pd.DataFrame(reduce)[["supplier", "coo", "tier", "quality_score", "action"]]
        rd.columns = ["Supplier", "COO", "Tier", "Quality Score", "Action"]
        rd["Quality Score"] = rd["Quality Score"].apply(lambda x: f"{float(x):.0f}" if pd.notna(x) else "n/a")
        st.dataframe(rd, use_container_width=True, height=120)


# ── Public entry point ────────────────────────────────────────────────────────

def render_dashboard_screen(registry, engine) -> None:
    # ── Header with refresh controls ──────────────────────────────────────
    hdr_col, refresh_col = st.columns([5, 1])
    with hdr_col:
        st.markdown(
            f"""
            <div style="margin-bottom:0.75rem;">
              <h2 style="color:{COLORS['text_primary']};margin:0 0 4px;">Quality Dashboard</h2>
              <p style="color:{COLORS['text_secondary']};font-size:0.85rem;margin:0;">
                Live KPI metrics — all figures computed from the production database
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with refresh_col:
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        if st.button("↻ Refresh", key="dashboard_refresh", help="Reload all KPIs from database"):
            # Clear cached KPI data so next render reloads from DB
            for key in list(st.session_state.keys()):
                if key.startswith("_kpi_"):
                    del st.session_state[key]
            st.session_state.pop("_dashboard_loaded_at", None)
            st.rerun()

    # Timestamp
    now_ts = time.time()
    loaded_at = st.session_state.get("_dashboard_loaded_at")
    if loaded_at is None:
        st.session_state["_dashboard_loaded_at"] = now_ts
        loaded_at = now_ts
    elapsed = int(now_ts - loaded_at)
    if elapsed < 5:
        refresh_label = "just now"
    elif elapsed < 60:
        refresh_label = f"{elapsed}s ago"
    elif elapsed < 3600:
        refresh_label = f"{elapsed // 60}m ago"
    else:
        refresh_label = f"{elapsed // 3600}h ago"

    st.markdown(
        f"<div style='font-size:0.72rem;color:#484F58;margin-bottom:0.75rem;'>"
        f"Last refreshed: {refresh_label}</div>",
        unsafe_allow_html=True,
    )

    # Load everything once
    with st.spinner("Loading KPIs..."):
        risk         = registry.kpi.get_lot_risk_scores()
        drift_all    = registry.kpi.get_process_drift_by_line_shift()
        drift_flagged= registry.kpi.get_drift_signals()
        rankings     = registry.kpi.get_supplier_rankings()
        coo_df       = registry.kpi.get_coo_performance()
        strategy     = registry.recommendations.get_inspection_strategy()
        try:
            with engine.connect() as conn:
                lot_count = conn.execute(
                    _text("SELECT COUNT(DISTINCT lot_id) FROM dim_lot")
                ).scalar() or 0
        except Exception:
            lot_count = len(risk)

    _row1_kpis(risk, drift_flagged, rankings, lot_count)
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    _row2_risk_drift(risk, drift_all, engine)
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    _row3_supplier_coo(rankings, coo_df)
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    _row4_inspection_priority(strategy)
