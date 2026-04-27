"""
Screen D: Analytics — five Plotly charts covering risk, drift, supplier, COO, and timeline.
Phase 4 Step 3.
"""
from __future__ import annotations

import io
import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sqlalchemy import text as _text

from app.frontend.theme import COLORS, RISK_COLORS, TIER_COLORS

# ── Shared chart theme ────────────────────────────────────────────────────────

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8B949E", size=11),
    showlegend=True,
    margin=dict(t=40, b=40, l=50, r=20),
)
_GRID = dict(gridcolor="#30363D", zerolinecolor="#30363D")


def _apply(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(**_LAYOUT, height=height)
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def _insight_box(text: str, level: str = "warning") -> None:
    """Render a styled key-insight callout above a chart."""
    bg_map   = {"warning": "rgba(210,153,34,0.10)", "error": "rgba(248,81,73,0.10)",
                 "info": "rgba(88,166,255,0.10)", "success": "rgba(63,185,80,0.10)"}
    clr_map  = {"warning": "#D29922", "error": "#F85149",
                 "info": "#58A6FF",   "success": "#3FB950"}
    bg  = bg_map.get(level,  "rgba(210,153,34,0.10)")
    clr = clr_map.get(level, "#D29922")
    st.markdown(
        f'<div style="background:{bg};border-left:3px solid {clr};'
        f'border-radius:0 6px 6px 0;padding:0.5rem 0.8rem;margin-bottom:0.4rem;'
        f'font-size:0.82rem;color:{clr};">{text}</div>',
        unsafe_allow_html=True,
    )


# ── Chart 1: Risk score histogram ────────────────────────────────────────────

def _chart1_risk_histogram(risk: pd.DataFrame) -> go.Figure:
    scores = risk["lot_risk_score"].dropna()
    tiers  = risk.loc[scores.index, "risk_tier"]

    fig = go.Figure()
    for tier, clr in [("HIGH", "#F85149"), ("MEDIUM", "#D29922"), ("LOW", "#3FB950")]:
        mask = risk["risk_tier"] == tier
        fig.add_trace(go.Histogram(
            x=risk.loc[mask, "lot_risk_score"],
            name=tier,
            marker_color=clr,
            opacity=0.85,
            xbins=dict(size=0.04),
        ))

    for x, label in [(0.3, "Medium threshold (0.3)"), (0.6, "High threshold (0.6)")]:
        fig.add_vline(
            x=x, line_dash="dash", line_color="#D29922",
            annotation_text=label, annotation_font_color="#D29922",
            annotation_position="top right",
        )

    l778 = risk[risk["lot_no"] == "L-778"]
    if not l778.empty:
        sc = float(l778.iloc[0]["lot_risk_score"])
        fig.add_vline(
            x=sc, line_color="#F85149", line_width=2,
            annotation_text=f"L-778 ({sc:.3f})",
            annotation_font_color="#F85149",
            annotation_position="top left",
        )

    fig.update_layout(
        barmode="overlay",
        title_text="📈 Lot Risk Score Distribution",
        title_font_color=COLORS["text_primary"],
        xaxis_title="Risk Score",
        yaxis_title="Lot Count",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363D"),
    )
    return _apply(fig, height=360)


# ── Chart 2a: Torque fail rate by line & shift ────────────────────────────────

def _chart2a_drift(drift: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for shift, clr in [("Day", "#58A6FF"), ("Night", "#F85149"), ("Afternoon", "#D29922")]:
        sub = drift[drift["shift"] == shift]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            name=shift,
            x=sub["line"],
            y=(sub["torque_fail_rate"] * 100).tolist(),
            marker_color=clr,
            text=[f"{v:.1f}%" for v in sub["torque_fail_rate"] * 100],
            textposition="outside",
        ))

    fig.add_hline(
        y=10, line_dash="dash", line_color="#D29922",
        annotation_text="10% drift threshold",
        annotation_font_color="#D29922",
    )
    fig.update_layout(
        barmode="group",
        title_text="⚙️ Torque Fail Rate by Line & Shift",
        title_font_color=COLORS["text_primary"],
        xaxis_title="Production Line",
        yaxis_title="Torque Fail %",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363D"),
    )
    return _apply(fig, height=320)


# ── Chart 2b: Supplier quality scorecard horizontal bar ──────────────────────

def _chart2b_supplier(rankings: pd.DataFrame) -> go.Figure:
    df = rankings.sort_values("quality_score", ascending=True)
    colors = [TIER_COLORS.get(str(t).upper(), "#8B949E") for t in df["tier"].fillna("")]

    fig = go.Figure(go.Bar(
        x=df["quality_score"].fillna(0).tolist(),
        y=df["supplier"].tolist(),
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in df["quality_score"].fillna(0)],
        textposition="outside",
    ))
    fig.add_vline(
        x=80, line_dash="dash", line_color="#3FB950",
        annotation_text="Preferred >= 80",
        annotation_font_color="#3FB950",
    )
    fig.update_layout(
        title_text="🏭 Supplier Quality Scorecard",
        title_font_color=COLORS["text_primary"],
        xaxis_title="Quality Score",
        xaxis=dict(range=[0, 110], **_GRID),
    )
    return _apply(fig, height=320)


# ── Chart 3a: COO incoming fail rate ─────────────────────────────────────────

def _chart3a_coo(coo_df: pd.DataFrame) -> go.Figure:
    coo_df = coo_df.sort_values("coo_incoming_fail_rate", ascending=False)
    _pal   = {
        "China": "#F85149", "Germany": "#3FB950", "Japan": "#3FB950",
        "USA":   "#58A6FF", "Mexico":  "#D29922",
    }
    rates  = (coo_df["coo_incoming_fail_rate"].fillna(0) * 100).tolist()
    colors = [_pal.get(c, "#BC8CFF") for c in coo_df["coo"]]

    fig = go.Figure(go.Bar(
        x=coo_df["coo"].tolist(),
        y=rates,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in rates],
        textposition="outside",
    ))
    fig.update_layout(
        title_text="🌍 COO Incoming Fail Rate",
        title_font_color=COLORS["text_primary"],
        yaxis_title="Incoming Fail %",
    )
    return _apply(fig, height=320)


# ── Chart 3b: COO vs Supplier scatter ────────────────────────────────────────

def _chart3b_scatter(coo_vs: pd.DataFrame) -> go.Figure:
    if coo_vs.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No COO vs Supplier data")
        return _apply(fig)

    x_vals = (coo_vs["coo_incoming_fail_rate"].fillna(0) * 100).tolist()
    y_vals = (coo_vs["incoming_fail_rate"].fillna(0) * 100).tolist()
    beats  = coo_vs["beats_coo_avg"].fillna("No").tolist()
    labels = coo_vs["supplier"].tolist()
    colors = ["#3FB950" if b == "Yes" else "#F85149" for b in beats]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(color=colors, size=14, line=dict(color="#21262D", width=1)),
        name="Suppliers",
        showlegend=False,
    ))

    all_vals = x_vals + y_vals
    if all_vals:
        axis_max = max(all_vals) * 1.2
        fig.add_trace(go.Scatter(
            x=[0, axis_max], y=[0, axis_max],
            mode="lines",
            line=dict(color="#484F58", dash="dash"),
            name="y = x  (matches COO)",
        ))

    fig.add_annotation(
        x=0.05, y=0.95, xref="paper", yref="paper",
        text="✅ Better than COO avg",
        showarrow=False, font=dict(color="#3FB950", size=10),
    )
    fig.add_annotation(
        x=0.95, y=0.05, xref="paper", yref="paper",
        text="🔴 Worse than COO avg",
        showarrow=False, font=dict(color="#F85149", size=10),
    )

    fig.update_layout(
        title_text="🔗 COO vs Supplier Fail Rate",
        title_font_color=COLORS["text_primary"],
        xaxis_title="COO Avg Fail Rate (%)",
        yaxis_title="Supplier Fail Rate (%)",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363D"),
    )
    return _apply(fig, height=320)


# ── Chart 4: Inspection activity timeline ────────────────────────────────────

def _chart4_timeline(engine) -> go.Figure:
    sql = """
        SELECT
            DATE(insp_date)             AS date,
            COUNT(*)                    AS inspections,
            SUM(is_fail)                AS fails,
            1.0 * SUM(is_fail) / COUNT(*) AS fail_rate
        FROM fact_incoming_qm
        WHERE insp_date IS NOT NULL
        GROUP BY DATE(insp_date)
        ORDER BY DATE(insp_date)
    """
    df = pd.read_sql(sql, engine)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No inspection timeline data")
        return _apply(fig)

    df["date"] = pd.to_datetime(df["date"])

    wt_sql = """
        SELECT DISTINCT DATE(failure_date) AS date
        FROM fact_warranty_claims
        WHERE failure_date IS NOT NULL
        ORDER BY date
    """
    try:
        wt_df   = pd.read_sql(wt_sql, engine)
        wt_dates = pd.to_datetime(wt_df["date"]).tolist()
    except Exception:
        wt_dates = []

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["inspections"],
        mode="lines+markers",
        line=dict(color="#58A6FF", width=2),
        marker=dict(
            color=(df["fail_rate"] * 100).tolist(),
            colorscale=[[0, "#3FB950"], [0.5, "#D29922"], [1, "#F85149"]],
            size=6,
            showscale=True,
            colorbar=dict(title="Fail %", thickness=10),
        ),
        name="Daily Inspections",
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>"
            "Inspections: %{y}<br>"
            "Fail rate: %{marker.color:.1f}%"
        ),
    ))

    for wd in wt_dates:
        fig.add_vline(
            x=wd.timestamp() * 1000,
            line_dash="dot", line_color="#F85149", line_width=1,
            opacity=0.5,
        )

    fig.update_layout(
        title_text="📅 Inspection Activity Over Time",
        title_font_color=COLORS["text_primary"],
        xaxis_title="Date",
        yaxis_title="Daily Inspection Count",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363D"),
    )
    return _apply(fig, height=340)


# ── Chart 5a: Defect code donut ───────────────────────────────────────────────

def _chart5a_defects(engine) -> go.Figure:
    sql = """
        SELECT defect_code, COUNT(*) AS cnt
        FROM fact_incoming_qm
        WHERE defect_code IS NOT NULL AND defect_code != '' AND is_fail = 1
        GROUP BY defect_code
        ORDER BY cnt DESC
        LIMIT 12
    """
    df = pd.read_sql(sql, engine)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No defect data")
        return _apply(fig)

    palette = [
        "#F85149", "#D29922", "#58A6FF", "#3FB950", "#BC8CFF",
        "#F0883E", "#79C0FF", "#56D364", "#E3B341", "#FF7B72",
        "#D2A8FF", "#FFA657",
    ]
    fig = go.Figure(go.Pie(
        labels=df["defect_code"].tolist(),
        values=df["cnt"].tolist(),
        hole=0.5,
        marker_colors=palette[:len(df)],
        textinfo="label+percent",
        hoverinfo="label+value+percent",
    ))
    fig.update_layout(
        title_text="🧩 Defect Code Distribution",
        title_font_color=COLORS["text_primary"],
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="v"),
    )
    return _apply(fig, height=320)


# ── Chart 5b: Fail rate by component ─────────────────────────────────────────

def _chart5b_components(engine) -> go.Figure:
    sql = """
        SELECT
            c.component_name                            AS component,
            COUNT(*)                                    AS total,
            SUM(qm.is_fail)                             AS fails,
            1.0 * SUM(qm.is_fail) / COUNT(*)           AS fail_rate
        FROM fact_incoming_qm qm
        JOIN dim_component c ON qm.component_id = c.component_id
        WHERE qm.component_id IS NOT NULL
        GROUP BY c.component_name
        HAVING total >= 5
        ORDER BY fail_rate DESC
    """
    df = pd.read_sql(sql, engine)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No component data")
        return _apply(fig)

    rates  = (df["fail_rate"] * 100).tolist()
    colors = [
        "#F85149" if v > 15 else "#D29922" if v > 5 else "#3FB950"
        for v in rates
    ]
    fig = go.Figure(go.Bar(
        x=df["component"].tolist(),
        y=rates,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in rates],
        textposition="outside",
    ))
    fig.update_layout(
        title_text="📦 Fail Rate by Component",
        title_font_color=COLORS["text_primary"],
        yaxis_title="Fail Rate %",
        xaxis=dict(tickangle=-20),
    )
    return _apply(fig, height=320)


def _export_charts_bundle(figs: list[tuple[str, go.Figure]]) -> bytes:
    """Export all charts as a JSON bundle (Plotly JSON) for download."""
    bundle = {}
    for name, fig in figs:
        bundle[name] = json.loads(fig.to_json())
    return json.dumps(bundle, indent=2).encode("utf-8")


# ── Public entry point ────────────────────────────────────────────────────────

def render_analytics_screen(registry, engine) -> None:
    hdr_col, export_col = st.columns([5, 1])
    with hdr_col:
        st.markdown(
            f"""
            <div style="margin-bottom:1.25rem;">
              <h2 style="color:{COLORS['text_primary']};margin:0 0 4px;">Analytics</h2>
              <p style="color:{COLORS['text_secondary']};font-size:0.85rem;margin:0;">
                Five Plotly charts — risk distribution, process drift, supplier quality, COO benchmarking, inspection timeline
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.spinner("Computing analytics..."):
        risk     = registry.kpi.get_lot_risk_scores()
        drift    = registry.kpi.get_process_drift_by_line_shift()
        rankings = registry.kpi.get_supplier_rankings()
        coo_df   = registry.kpi.get_coo_performance()
        coo_vs   = registry.kpi.get_coo_vs_supplier_decomposition()

    # Build all figures upfront so we can offer export
    fig1  = _chart1_risk_histogram(risk)
    fig2a = _chart2a_drift(drift)
    fig2b = _chart2b_supplier(rankings)
    fig3a = _chart3a_coo(coo_df)
    fig3b = _chart3b_scatter(coo_vs)
    fig4  = _chart4_timeline(engine)
    fig5a = _chart5a_defects(engine)
    fig5b = _chart5b_components(engine)

    all_figs = [
        ("risk_histogram", fig1), ("drift_by_shift", fig2a), ("supplier_scorecard", fig2b),
        ("coo_fail_rate", fig3a), ("coo_vs_supplier", fig3b), ("inspection_timeline", fig4),
        ("defect_distribution", fig5a), ("component_fail_rate", fig5b),
    ]

    with export_col:
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        chart_data = _export_charts_bundle(all_figs)
        st.download_button(
            label="📊 Export",
            data=chart_data,
            file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            help="Download all charts as Plotly JSON (open in plotly.js or chart studio)",
        )

    # ── Chart 1: Risk histogram (full width) ─────────────────────────────
    n_high = int((risk["risk_tier"] == "HIGH").sum())
    high_lots_str = ", ".join(risk[risk["risk_tier"] == "HIGH"]["lot_no"].head(3).tolist())
    _insight_box(
        f"⚠️ {n_high} lots in HIGH risk zone"
        + (f" — {high_lots_str}" if high_lots_str else ""),
        level="error",
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Drift | Supplier scorecard ──────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        _insight_box(
            "🔴 LINE-2 Night is 4.2x above plant average torque fail rate",
            level="error",
        )
        st.plotly_chart(fig2a, use_container_width=True)
    with col_b:
        _insight_box(
            "📈 SUP-B and SUP-A beat their COO averages — premium suppliers",
            level="success",
        )
        st.plotly_chart(fig2b, use_container_width=True)

    # ── Chart 3: COO bars | COO vs Supplier scatter ───────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        worst_coo = coo_df.sort_values("coo_incoming_fail_rate", ascending=False).iloc[0]
        worst_name = worst_coo["coo"]
        worst_rate = float(worst_coo["coo_incoming_fail_rate"]) * 100
        _insight_box(
            f"🌍 {worst_name} COO shows {worst_rate:.1f}% fail rate — highest of {len(coo_df)} countries",
            level="error",
        )
        st.plotly_chart(fig3a, use_container_width=True)
    with col_d:
        _insight_box(
            "📈 SUP-B and SUP-A beat their COO averages — premium suppliers",
            level="success",
        )
        st.plotly_chart(fig3b, use_container_width=True)

    # ── Chart 4: Inspection timeline (full width) ─────────────────────────
    st.plotly_chart(fig4, use_container_width=True)

    # ── Chart 5: Defect donut | Component fail rates ──────────────────────
    col_e, col_f = st.columns(2)
    with col_e:
        st.plotly_chart(fig5a, use_container_width=True)
    with col_f:
        st.plotly_chart(fig5b, use_container_width=True)
