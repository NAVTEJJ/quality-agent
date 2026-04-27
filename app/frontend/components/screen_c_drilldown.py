"""
Screen C: Quality Traceability Investigation — drill from lot -> serial -> process -> warranty.
Phase 4 Step 3.
"""
from __future__ import annotations

import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from app.frontend.theme import COLORS, RISK_COLORS, TIER_COLORS

# ── Small helpers ─────────────────────────────────────────────────────────────

def _badge(label: str, bg: str, fg: str = "#fff") -> str:
    return (
        f'<span style="background:{bg};color:{fg};border-radius:999px;'
        f'padding:2px 10px;font-size:0.7rem;font-weight:700;'
        f'letter-spacing:0.03em;">{label}</span>'
    )

def _connector(label: str = "") -> None:
    if label:
        st.markdown(
            f"<div style='font-size:0.72rem;color:{COLORS['text_muted']};margin:0.4rem 0 0.15rem;'>{label}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

def _card_start(border_color: str = "#30363D") -> None:
    st.markdown(
        f'<div style="background:#21262D;border:1px solid {border_color};'
        f'border-radius:8px;padding:1rem;margin-bottom:0.25rem;">',
        unsafe_allow_html=True,
    )

def _card_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)

def _pct(v) -> str:
    return f"{float(v):.1%}" if v is not None and str(v) != "None" else "n/a"

def _fmt(v) -> str:
    return str(v) if v is not None else "—"


# ── Breadcrumb ────────────────────────────────────────────────────────────────

def _render_breadcrumb(chain: Dict[str, Any], lot_no: str) -> None:
    lot    = chain.get("lot_info", {})
    comp   = lot.get("component") or "—"
    sup    = lot.get("supplier")  or "—"
    n_insp = len(chain.get("inspection_records", []))

    crumb = (
        f'<div style="font-size:0.78rem;color:{COLORS["text_secondary"]};'
        f'margin-bottom:0.6rem;padding:0.4rem 0.8rem;background:#21262D;'
        f'border-radius:6px;border:1px solid #30363D;">'
        f'🗂 <span style="color:{COLORS["text_primary"]};font-weight:600;">Lot {lot_no}</span>'
        f' &nbsp;→&nbsp; {comp}'
        f' &nbsp;→&nbsp; {sup}'
        f' &nbsp;→&nbsp; {n_insp} inspection{"s" if n_insp != 1 else ""}'
        f'</div>'
    )
    st.markdown(crumb, unsafe_allow_html=True)


# ── Level renderers ───────────────────────────────────────────────────────────

def _level1_lot_summary(chain: Dict[str, Any]) -> None:
    lot   = chain["lot_info"]
    summ  = chain["summary"]
    risk  = chain.get("_risk_row", {})

    score     = risk.get("lot_risk_score")
    tier      = risk.get("risk_tier", "UNKNOWN")
    fail_rate = summ.get("fail_rate", 0.0)
    color     = RISK_COLORS.get(str(tier), "#8B949E")

    st.markdown(
        f'<div style="background:#21262D;border:2px solid {color};'
        f'border-radius:10px;padding:1.25rem;margin-bottom:0.25rem;">',
        unsafe_allow_html=True,
    )
    header_col, score_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            f"<h3 style='color:{COLORS['text_primary']};margin:0;'>"
            f"Lot {lot.get('lot_no', '?')}</h3>",
            unsafe_allow_html=True,
        )
        tier_bg  = color
        tier_fg  = "#0E1117" if tier == "LOW" else "#fff"
        st.markdown(
            f"{_badge(tier, tier_bg, tier_fg)}"
            f"&nbsp;<span style='color:{COLORS['text_secondary']};font-size:0.82rem;'>"
            f"{lot.get('component', '—')} &nbsp;|&nbsp; {lot.get('supplier', '—')}"
            f" &nbsp;|&nbsp; COO: {lot.get('coo', '—')}</span>",
            unsafe_allow_html=True,
        )
        mfg = lot.get("mfg_date") or "—"
        st.markdown(
            f"<span style='color:{COLORS['text_muted']};font-size:0.75rem;'>Mfg date: {mfg}</span>",
            unsafe_allow_html=True,
        )
    with score_col:
        if score is not None:
            st.metric("Risk Score", f"{float(score):.3f}")
        st.metric("Fail Rate", _pct(fail_rate))

    st.markdown("<hr style='border-color:#30363D;margin:0.75rem 0;'>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Inspections",       summ.get("total_inspections", 0))
    m2.metric("Fails",             summ.get("total_fails", 0))
    m3.metric("Affected Serials",  summ.get("affected_serials", 0))
    m4.metric("Warranty Claims",   summ.get("serials_with_warranty", 0))
    st.markdown("</div>", unsafe_allow_html=True)


def _level2_inspections(records: List[Dict[str, Any]]) -> None:
    n = len(records)
    with st.expander(f"📋 Incoming Inspection Records ({n})", expanded=True):
        if not records:
            st.info("No inspection records found for this lot.")
            return
        df = pd.DataFrame(records)
        disp_cols = ["insp_date", "insp_lot", "characteristic",
                     "measured_value", "uom", "result", "defect_code"]
        disp_cols = [c for c in disp_cols if c in df.columns]
        disp = df[disp_cols].copy()
        disp.columns = [c.replace("_", " ").title() for c in disp_cols]

        def _insp_style(row):
            idx = disp.index[disp.index == row.name]
            is_fail = df.loc[idx[0], "is_fail"] == 1 if len(idx) else False
            if is_fail:
                return ["background-color:rgba(248,81,73,0.15)"] * len(row)
            return ["background-color:rgba(63,185,80,0.06)"] * len(row)

        st.dataframe(
            disp.style.apply(_insp_style, axis=1),
            use_container_width=True,
            height=min(40 * n + 40, 280),
        )


def _level3_serials(serials: List[Dict[str, Any]]) -> Optional[str]:
    """Render serial list; return the selected serial_no or None."""
    n = len(serials)
    with st.expander(f"🔩 Affected Serial Numbers ({n} serials)", expanded=True):
        if not serials:
            st.info("No serials linked to this lot via BOM.")
            return None
        df = pd.DataFrame(serials)
        disp_cols = ["serial_no", "finished_material", "build_dt", "line", "shift", "plant"]
        disp_cols = [c for c in disp_cols if c in df.columns]
        disp = df[disp_cols].copy()
        disp.columns = [c.replace("_", " ").title() for c in disp_cols]

        def _serial_style(row):
            idx = disp.index[disp.index == row.name]
            line  = df.loc[idx[0], "line"]  if len(idx) else ""
            shift = df.loc[idx[0], "shift"] if len(idx) else ""
            if str(line) == "LINE-2" and str(shift) == "Night":
                return ["background-color:rgba(248,81,73,0.12)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            disp.style.apply(_serial_style, axis=1),
            use_container_width=True,
            height=min(40 * n + 40, 260),
        )

        serial_list = [s["serial_no"] for s in serials if s.get("serial_no")]
        selected = st.selectbox(
            "Select serial to inspect:", ["— pick a serial —"] + serial_list,
            key="drill_serial_select",
        )
        return None if selected.startswith("—") else selected


def _level4_process(measurements: List[Dict[str, Any]], selected_serial: str) -> None:
    serial_meas = [m for m in measurements if m.get("serial_no") == selected_serial]
    with st.expander(f"⚙️ Process Measurements — {selected_serial}", expanded=True):
        if not serial_meas:
            st.info("No process measurements found for this serial.")
            return
        df = pd.DataFrame(serial_meas)

        is_line2_night = any(
            str(r.get("line")) == "LINE-2" and str(r.get("shift")) == "Night"
            for r in serial_meas
        )
        if is_line2_night:
            st.warning("⚠️ This serial was built on **LINE-2 Night** — the drifting line/shift.")

        disp_cols = ["build_date", "line", "shift",
                     "torque_nm", "torque_result", "is_torque_fail",
                     "leak_rate_ccm", "leak_result", "is_leak_fail"]
        disp_cols = [c for c in disp_cols if c in df.columns]
        disp = df[disp_cols].copy()
        disp.columns = [c.replace("_", " ").title() for c in disp_cols]

        def _proc_style(row):
            idx = disp.index[disp.index == row.name]
            tf = df.loc[idx[0], "is_torque_fail"] == 1 if len(idx) else False
            lf = df.loc[idx[0], "is_leak_fail"]   == 1 if len(idx) else False
            if tf or lf:
                return ["background-color:rgba(248,81,73,0.15)"] * len(row)
            return ["background-color:rgba(63,185,80,0.06)"] * len(row)

        st.dataframe(
            disp.style.apply(_proc_style, axis=1),
            use_container_width=True,
        )


def _level5_warranty(outcomes: List[Dict[str, Any]], selected_serial: str) -> None:
    claim = next(
        (w for w in outcomes if w.get("serial_no") == selected_serial), None
    )
    with st.expander("🔴 Warranty / Field Outcome", expanded=True):
        if claim is None:
            st.success("✅ No field failures recorded for this serial.")
            return
        st.error("🔴 Field failure confirmed for this serial.")
        cols = st.columns(3)
        cols[0].metric("Claim ID",       _fmt(claim.get("claim_id")))
        cols[1].metric("Severity",       _fmt(claim.get("severity")))
        cols[2].metric("Region",         _fmt(claim.get("region")))
        st.markdown(
            f"**Failure date:** {_fmt(claim.get('failure_date'))}  \n"
            f"**Symptom:** {_fmt(claim.get('symptom'))}  \n"
            f"**Mileage/Hours at failure:** {_fmt(claim.get('mileage_or_hours'))}",
        )


def _level6_supplier(scorecard: Optional[Dict], coo_ctx: Optional[Dict]) -> None:
    with st.expander("🏭 Supplier & COO Context", expanded=True):
        if not scorecard:
            st.info("No supplier scorecard data found.")
            return

        tier  = str(scorecard.get("tier") or "")
        color = TIER_COLORS.get(tier.upper(), "#8B949E")
        beats = str(coo_ctx.get("beats_coo_avg", "")) if coo_ctx else ""
        gap   = coo_ctx.get("gap") if coo_ctx else None

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(
                f"**{scorecard.get('supplier', '?')}** &nbsp;"
                f"{_badge(tier, color)}",
                unsafe_allow_html=True,
            )
            qs = scorecard.get("quality_score")
            st.markdown(
                f"Quality score: **{float(qs):.0f}/100**" if qs else "Quality score: n/a"
            )
            st.markdown(f"COO: **{scorecard.get('coo', '—')}**")
            st.markdown(f"Process Cpk: **{scorecard.get('process_cpk', '—')}**")
            st.markdown(f"Engineering maturity: **{scorecard.get('engineering_maturity', '—')}**")
        with sc2:
            if coo_ctx:
                sup_fr  = coo_ctx.get("incoming_fail_rate")
                coo_fr  = coo_ctx.get("coo_incoming_fail_rate")
                verdict = coo_ctx.get("gap_interpretation", "")
                st.metric(
                    "Supplier fail rate vs COO avg",
                    _pct(sup_fr),
                    delta=f"COO avg: {_pct(coo_fr)}",
                    delta_color="off",
                )
                if beats == "Yes":
                    st.success(f"✅ {verdict}")
                else:
                    st.warning(f"⚠️ {verdict}")
                if gap is not None:
                    gap_pct = f"{abs(gap):.1%}"
                    direction = "better than" if gap > 0 else "worse than"
                    st.caption(f"Gap: {gap_pct} {direction} COO average")


# ── Markdown export ───────────────────────────────────────────────────────────

def _build_report(chain: Dict[str, Any], lot_no: str) -> str:
    lot    = chain["lot_info"]
    summ   = chain["summary"]
    risk   = chain.get("_risk_row", {})
    sc     = chain.get("supplier_scorecard") or {}
    coo    = chain.get("coo_context") or {}
    now    = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Quality Investigation Report — Lot {lot_no}",
        f"_Generated: {now}_",
        "",
        "## 1. Lot Summary",
        f"- **Lot No:** {lot.get('lot_no')}",
        f"- **Component:** {lot.get('component')}",
        f"- **Supplier:** {lot.get('supplier')} (COO: {lot.get('coo')})",
        f"- **Mfg Date:** {lot.get('mfg_date')}",
        f"- **Risk Score:** {float(risk['lot_risk_score']):.3f} — **{risk.get('risk_tier')}**"
        if risk.get("lot_risk_score") else "- **Risk Score:** n/a",
        f"- **Fail Rate:** {_pct(summ.get('fail_rate'))}",
        "",
        "## 2. Inspection Summary",
        f"- Total inspections: {summ.get('total_inspections')}",
        f"- Total fails: {summ.get('total_fails')}",
        f"- Affected serials: {summ.get('affected_serials')}",
        f"- Serials with warranty claims: {summ.get('serials_with_warranty')}",
        "",
        "## 3. Inspection Records",
    ]
    for r in chain.get("inspection_records", [])[:10]:
        status = "FAIL" if r.get("is_fail") else "PASS"
        lines.append(
            f"- {r.get('insp_date')} | {r.get('characteristic')} "
            f"| {r.get('measured_value')} {r.get('uom')} | **{status}**"
            + (f" | defect: {r.get('defect_code')}" if r.get("defect_code") else "")
        )

    lines += [
        "",
        "## 4. Warranty Outcomes",
    ]
    for w in chain.get("warranty_outcomes", []):
        lines.append(
            f"- Serial {w.get('serial_no')} | Claim {w.get('claim_id')} "
            f"| {w.get('failure_date')} | {w.get('symptom')} | Severity: {w.get('severity')}"
        )
    if not chain.get("warranty_outcomes"):
        lines.append("- No warranty claims linked to this lot.")

    lines += [
        "",
        "## 5. Supplier Context",
        f"- **Supplier:** {sc.get('supplier')} — Tier: {sc.get('tier')}",
        f"- **Quality Score:** {sc.get('quality_score')}",
        f"- **Incoming fail rate:** {_pct(sc.get('incoming_fail_rate'))}",
        f"- **Warranty claim rate:** {_pct(sc.get('warranty_claim_rate'))}",
        f"- **COO benchmark:** {coo.get('gap_interpretation', 'n/a')}",
    ]
    return "\n".join(lines)


def _build_share_summary(chain: Dict[str, Any], lot_no: str) -> str:
    lot  = chain.get("lot_info", {})
    summ = chain.get("summary", {})
    risk = chain.get("_risk_row", {})
    sc   = chain.get("supplier_scorecard") or {}

    tier  = risk.get("risk_tier", "UNKNOWN")
    score = risk.get("lot_risk_score")
    score_str = f"{float(score):.3f}" if score is not None else "n/a"
    w_count = summ.get("serials_with_warranty", 0)

    lines = [
        f"Investigation Summary — Lot {lot_no}",
        f"Risk: {tier} (score {score_str})",
        f"Component: {lot.get('component', '—')} | Supplier: {sc.get('supplier', '—')} | COO: {lot.get('coo', '—')}",
        f"Inspections: {summ.get('total_inspections', 0)} | Fails: {summ.get('total_fails', 0)} | Fail rate: {_pct(summ.get('fail_rate', 0))}",
        f"Affected serials: {summ.get('affected_serials', 0)} | Warranty claims: {w_count}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    return "\n".join(lines)


# ── Public entry point ────────────────────────────────────────────────────────

def render_drilldown_screen(registry, engine) -> None:
    st.markdown(
        f"""
        <div style="margin-bottom:1.25rem;">
          <h2 style="color:{COLORS['text_primary']};margin:0 0 4px;">
            🔍 Quality Traceability Investigation
          </h2>
          <p style="color:{COLORS['text_secondary']};font-size:0.85rem;margin:0;">
            Drill from lot risk &rarr; component &rarr; serial &rarr; process &rarr; field outcome
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sync external lot selection (Dashboard / Copilot sidebar clicks) ────
    # Only update the field when current_lot changes externally; never
    # overwrite on every render (that's what caused the auto-populate bug).
    _ext_lot = st.session_state.get("current_lot", "")
    if _ext_lot and _ext_lot != st.session_state.get("_drill_ext_lot_seen", ""):
        st.session_state.drill_lot_value = _ext_lot
        st.session_state._drill_ext_lot_seen = _ext_lot
    if "drill_lot_value" not in st.session_state:
        st.session_state.drill_lot_value = ""
    if "drill_serial_value" not in st.session_state:
        st.session_state.drill_serial_value = ""

    # ── Search bar ────────────────────────────────────────────────────────
    lot_col, lot_clr, serial_col, serial_clr, search_col = st.columns(
        [2.3, 0.35, 2.3, 0.35, 1]
    )

    with lot_col:
        lot_input = st.text_input(
            "Lot Number",
            placeholder="e.g. L-778",
            key="drill_lot_value",
        )
    with lot_clr:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        if st.button("✕", key="btn_clear_lot", help="Clear lot"):
            st.session_state.drill_lot_value = ""
            st.rerun()

    with serial_col:
        serial_input = st.text_input(
            "Serial Number",
            placeholder="e.g. SR20260008",
            key="drill_serial_value",
        )
    with serial_clr:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        if st.button("✕", key="btn_clear_serial", help="Clear serial"):
            st.session_state.drill_serial_value = ""
            st.rerun()

    with search_col:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        search = st.button("🔍 Search", use_container_width=True)

    lot_no   = lot_input.strip().upper()
    serial_no = serial_input.strip()

    # Auto-resolve serial → lot only when Search is explicitly clicked
    if search and serial_no and not lot_no:
        try:
            from sqlalchemy import text as _text
            with engine.connect() as conn:
                row = conn.execute(
                    _text("""
                        SELECT l.lot_no FROM fact_constituent_bom b
                        JOIN dim_serial ds ON b.serial_id = ds.serial_id
                        JOIN dim_lot l ON b.lot_id = l.lot_id
                        WHERE ds.serial_no = :sn LIMIT 1
                    """),
                    {"sn": serial_no},
                ).fetchone()
            if row:
                lot_no = str(row[0])
                st.session_state.drill_lot_value = lot_no
                st.info(f"Serial {serial_no} resolved to lot **{lot_no}**.")
        except Exception:
            pass

    if not lot_no:
        st.markdown(
            f"<div style='color:{COLORS['text_muted']};margin-top:1rem;'>"
            f"Enter a lot number above or click a lot from the Dashboard to begin investigation.</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Load chain ────────────────────────────────────────────────────────
    with st.spinner(f"Loading investigation chain for lot {lot_no}..."):
        try:
            chain = registry.drill_down.get_full_drill_down_chain(lot_no)
        except LookupError as exc:
            st.error(f"Lot **{lot_no}** not found: {exc}")
            return
        except Exception as exc:
            st.error(f"Error loading chain: {exc}")
            return

    # Attach risk row for score/tier display
    try:
        risk_df = registry.kpi.get_lot_risk_scores()
        r778 = risk_df[risk_df["lot_no"] == lot_no]
        chain["_risk_row"] = r778.iloc[0].to_dict() if not r778.empty else {}
    except Exception:
        chain["_risk_row"] = {}

    # ── Breadcrumb ────────────────────────────────────────────────────────
    _render_breadcrumb(chain, lot_no)

    # ── Render investigation chain ────────────────────────────────────────
    _level1_lot_summary(chain)

    insp_count = len(chain.get("inspection_records", []))
    _connector(f"{insp_count} inspection record{'s' if insp_count != 1 else ''}")
    _level2_inspections(chain.get("inspection_records", []))

    serial_count = len(chain.get("affected_serials", []))
    _connector(f"{serial_count} affected serial{'s' if serial_count != 1 else ''}")
    selected_serial = _level3_serials(chain.get("affected_serials", []))

    if selected_serial:
        st.session_state.drill_selected_serial = selected_serial
    selected_serial = st.session_state.get("drill_selected_serial")

    if selected_serial:
        _connector(f"process data for {selected_serial}")
        _level4_process(chain.get("process_measurements", []), selected_serial)

        warranty_count = len(chain.get("warranty_outcomes", []))
        _connector(
            f"{warranty_count} warranty claim{'s' if warranty_count != 1 else ''} linked to lot"
        )
        _level5_warranty(chain.get("warranty_outcomes", []), selected_serial)
    else:
        warranty_count = len(chain.get("warranty_outcomes", []))
        _connector(
            f"{warranty_count} warranty claim{'s' if warranty_count != 1 else ''} across all serials"
        )
        with st.expander("🔴 Warranty Summary (all serials)", expanded=True):
            outcomes = chain.get("warranty_outcomes", [])
            if outcomes:
                st.error(f"🔴 {len(outcomes)} field failure(s) linked to this lot.")
                wdf = pd.DataFrame(outcomes)
                st.dataframe(wdf, use_container_width=True, height=min(40 * len(outcomes) + 40, 220))
            else:
                st.success("✅ No field failures recorded for any serial in this lot.")

    _connector()
    _level6_supplier(chain.get("supplier_scorecard"), chain.get("coo_context"))

    # ── Export button ─────────────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    from app.frontend.components.export import generate_investigation_report
    report_md = generate_investigation_report(lot_no, chain, registry, engine)
    st.download_button(
        label="📥 Export Investigation Report (.md)",
        data=report_md,
        file_name=f"investigation_{lot_no}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
    )
