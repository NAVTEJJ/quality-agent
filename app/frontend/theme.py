"""
Visual theme constants and CSS injection for the Quality Agent Streamlit UI.
"""

COLORS = {
    "bg_primary":    "#0E1117",
    "bg_secondary":  "#1A1D27",
    "bg_card":       "#21262D",
    "border":        "#30363D",
    "text_primary":  "#E6EDF3",
    "text_secondary":"#8B949E",
    "text_muted":    "#484F58",
    "accent_blue":   "#58A6FF",
    "accent_green":  "#3FB950",
    "accent_yellow": "#D29922",
    "accent_red":    "#F85149",
    "accent_purple": "#BC8CFF",
    "accent_orange": "#F0883E",
}

RISK_COLORS = {
    "HIGH":    "#F85149",
    "MEDIUM":  "#D29922",
    "LOW":     "#3FB950",
    "UNKNOWN": "#8B949E",
}

TIER_COLORS = {
    "PREFERRED":     "#3FB950",
    "APPROVED":      "#58A6FF",
    "CONDITIONAL":   "#D29922",
    "WATCHLIST":     "#F0883E",
    "DISQUALIFIED":  "#F85149",
}

CSS = """
<style>
/* ── Global ─────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    background-color: #0E1117;
    color: #E6EDF3;
}

/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu                           { visibility: hidden; }
footer                              { visibility: hidden; }
[data-testid="stToolbar"]           { visibility: hidden; }
[data-testid="stDecoration"]        { display: none; }
[data-testid="stStatusWidget"]      { visibility: hidden; }

/* Sidebar buttons left visible so Streamlit can always expand/collapse natively */

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1A1D27;
    border-right: 1px solid #30363D;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ── Nav buttons ─────────────────────────────────────────────────────────── */
div[data-testid="stSidebar"] button {
    width: 100%;
    text-align: left;
    background-color: transparent;
    border: 1px solid transparent;
    color: #8B949E;
    border-radius: 6px;
    padding: 0.45rem 0.75rem;
    margin-bottom: 0.25rem;
    font-size: 0.9rem;
    transition: all 0.15s ease;
}
div[data-testid="stSidebar"] button:hover {
    background-color: #21262D;
    border-color: #30363D;
    color: #E6EDF3;
}
div[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #21262D;
    border-color: #58A6FF;
    color: #58A6FF;
    font-weight: 600;
}

/* ── Metric cards ────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background-color: #21262D;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 1rem;
}
[data-testid="stMetricValue"] {
    color: #58A6FF;
    font-size: 1.6rem;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: #8B949E;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Alert badges ────────────────────────────────────────────────────────── */
.alert-high {
    background-color: rgba(248, 81, 73, 0.12);
    border-left: 3px solid #F85149;
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #E6EDF3;
}
.alert-medium {
    background-color: rgba(210, 153, 34, 0.12);
    border-left: 3px solid #D29922;
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #E6EDF3;
}
.alert-info {
    background-color: rgba(88, 166, 255, 0.10);
    border-left: 3px solid #58A6FF;
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #E6EDF3;
}

/* ── Risk chips ──────────────────────────────────────────────────────────── */
.risk-chip-high     { display:inline-block; background:#F85149; color:#fff;
                      border-radius:999px; padding:2px 10px; font-size:0.72rem;
                      font-weight:700; letter-spacing:0.03em; }
.risk-chip-medium   { display:inline-block; background:#D29922; color:#fff;
                      border-radius:999px; padding:2px 10px; font-size:0.72rem;
                      font-weight:700; letter-spacing:0.03em; }
.risk-chip-low      { display:inline-block; background:#3FB950; color:#0E1117;
                      border-radius:999px; padding:2px 10px; font-size:0.72rem;
                      font-weight:700; letter-spacing:0.03em; }

/* ── Chat bubbles ────────────────────────────────────────────────────────── */
.chat-user {
    background-color: #21262D;
    border: 1px solid #30363D;
    border-radius: 12px 12px 4px 12px;
    padding: 0.65rem 0.9rem;
    margin: 0.4rem 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.88rem;
}
.chat-agent {
    background-color: #1A1D27;
    border: 1px solid #58A6FF33;
    border-radius: 4px 12px 12px 12px;
    padding: 0.65rem 0.9rem;
    margin: 0.4rem 0;
    max-width: 90%;
    font-size: 0.88rem;
    line-height: 1.55;
}

/* ── Section headers ─────────────────────────────────────────────────────── */
.section-header {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #484F58;
    margin: 1rem 0 0.4rem;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid #21262D;
}

/* ── Suggestion chips ────────────────────────────────────────────────────── */
.suggestion-chip {
    display: inline-block;
    background-color: #21262D;
    border: 1px solid #30363D;
    border-radius: 999px;
    padding: 4px 14px;
    margin: 3px 4px 3px 0;
    font-size: 0.78rem;
    color: #8B949E;
    cursor: pointer;
    transition: all 0.12s ease;
}
.suggestion-chip:hover {
    border-color: #58A6FF;
    color: #58A6FF;
    background-color: rgba(88,166,255,0.08);
}

/* ── Data tables ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] table {
    background-color: #21262D;
    border: 1px solid #30363D;
    border-radius: 6px;
}
[data-testid="stDataFrame"] th {
    background-color: #1A1D27;
    color: #8B949E;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid #30363D;
}
[data-testid="stDataFrame"] td {
    color: #E6EDF3;
    font-size: 0.82rem;
    border-bottom: 1px solid #21262D;
}

/* ── Progress / spinner ──────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #58A6FF !important; }

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid #21262D;
    margin: 1rem 0;
}

/* ── Status dot ──────────────────────────────────────────────────────────── */
.status-dot-green  { display:inline-block; width:8px; height:8px;
                     border-radius:50%; background:#3FB950; margin-right:6px; }
.status-dot-red    { display:inline-block; width:8px; height:8px;
                     border-radius:50%; background:#F85149; margin-right:6px; }
.status-dot-yellow { display:inline-block; width:8px; height:8px;
                     border-radius:50%; background:#D29922; margin-right:6px; }
</style>
"""
