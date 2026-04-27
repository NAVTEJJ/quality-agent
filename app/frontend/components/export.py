"""
Markdown investigation report generator.
Phase 4 Step 4.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


def _pct(v) -> str:
    try:
        return f"{float(v):.1%}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt(v) -> str:
    return str(v) if v is not None else "—"


def _score(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def generate_investigation_report(
    lot_no: str,
    chain: Dict[str, Any],
    registry=None,
    engine=None,
) -> str:
    """Return a full markdown investigation report for *lot_no*.

    Args:
        lot_no:   The lot number under investigation.
        chain:    Dict returned by DrillDownService.get_full_drill_down_chain().
        registry: Optional ServiceRegistry — used to pull recommended actions.
        engine:   Optional SQLAlchemy engine (unused directly, kept for API symmetry).
    """
    lot   = chain.get("lot_info") or {}
    summ  = chain.get("summary") or {}
    insp  = chain.get("inspection_records") or []
    sers  = chain.get("affected_serials") or []
    proc  = chain.get("process_measurements") or []
    warr  = chain.get("warranty_outcomes") or []
    sc    = chain.get("supplier_scorecard") or {}
    coo   = chain.get("coo_context") or {}
    risk  = chain.get("_risk_row") or {}

    now       = datetime.now().strftime("%Y-%m-%d %H:%M")
    tier      = str(risk.get("risk_tier") or "UNKNOWN")
    score_val = risk.get("lot_risk_score")
    fail_rate = summ.get("fail_rate", 0.0)
    n_claims  = summ.get("serials_with_warranty", 0)
    n_insp    = summ.get("total_inspections", 0)
    n_serials = summ.get("affected_serials", 0)

    # Executive summary text
    risk_phrase = {
        "HIGH":   "HIGH risk and requires immediate containment",
        "MEDIUM": "MEDIUM risk and should be monitored closely",
        "LOW":    "LOW risk with no immediate action required",
    }.get(tier, "of unknown risk tier")
    exec_summary = (
        f"Lot {lot_no} ({lot.get('component', '?')} supplied by {lot.get('supplier', '?')}, "
        f"COO: {lot.get('coo', '?')}) is {risk_phrase}. "
        f"The composite risk score is {_score(score_val)} with an incoming fail rate of {_pct(fail_rate)} "
        f"across {n_insp:,} inspections affecting {n_serials:,} finished-goods serials. "
        f"{n_claims} warranty claim(s) have been linked to this lot in the field."
    )

    # Recommended actions
    actions = []
    if registry is not None:
        try:
            actions_dict = registry.recommendations.get_actions_for_lot_risk(
                lot_no=lot_no,
                risk_score=float(score_val) if score_val is not None else 0.0,
                supplier_tier=sc.get("tier"),
            )
            actions     = actions_dict.get("actions", [])
            sap_points  = actions_dict.get("sap_touchpoints", [])
        except Exception:
            actions, sap_points = [], []
    else:
        sap_points = []

    if not actions:
        actions = [
            f"Block lot {lot_no} pending supplier review",
            "Create supplier 8D corrective action report",
            "100% incoming inspection on next lot",
            "Trigger containment review for affected serials",
        ]
    if not sap_points:
        sap_points = ["QA32", "QE51N", "QM01"]

    _default_sap = (
        "1. QA32: Review inspection results\n"
        "2. QE51N: Update inspection plan\n"
        "3. QM01: Create quality notification"
    )

    # Inspection table rows (max 20)
    insp_rows = "\n".join(
        f"| {r.get('insp_date','—')} | {r.get('characteristic','—')} "
        f"| {r.get('measured_value','—')} {r.get('uom','')} "
        f"| {'**FAIL**' if r.get('is_fail') else 'PASS'} "
        f"| {r.get('defect_code') or '—'} |"
        for r in insp[:20]
    )

    # Warranty rows
    warr_rows = "\n".join(
        f"| {w.get('serial_no','—')} | {w.get('claim_id','—')} "
        f"| {w.get('failure_date','—')} | {w.get('symptom','—')} "
        f"| {w.get('severity','—')} | {w.get('region','—')} |"
        for w in warr
    ) or "_No warranty claims linked._"

    # Process summary
    proc_fails = sum(
        1 for r in proc
        if r.get("is_torque_fail") == 1 or r.get("is_leak_fail") == 1
    )
    proc_summary = (
        f"{len(proc)} process measurement(s) across {n_serials} serial(s). "
        f"{proc_fails} measurement(s) had torque or leak failures."
    )
    line2_night = any(
        str(r.get("line")) == "LINE-2" and str(r.get("shift")) == "Night"
        for r in proc
    )
    if line2_night:
        proc_summary += " **Note:** Some units were built on LINE-2 Night — the drifting line/shift."

    # COO context
    coo_line = ""
    if coo:
        beats = coo.get("beats_coo_avg", "No")
        gap   = coo.get("gap")
        coo_line = (
            f"- COO average fail rate: {_pct(coo.get('coo_incoming_fail_rate'))}\n"
            f"- Supplier vs COO: {'✅ Outperforms' if beats == 'Yes' else '⚠️ Underperforms'} its COO average"
        )
        if gap is not None:
            coo_line += f" (gap: {abs(float(gap)):.1%})"

    # Numbered actions
    action_lines = "\n".join(f"{i}. {a}" for i, a in enumerate(actions, 1))
    sap_lines    = "\n".join(
        f"{i}. **{tp}**: {'Review inspection results' if 'QA32' in tp else 'Update inspection plan' if 'QE51N' in tp else 'Create quality notification' if 'QM01' in tp else 'Execute action'}"
        for i, tp in enumerate(sap_points, 1)
    )

    return f"""# Quality Investigation Report — Lot {lot_no}
_Generated: {now} by AI Quality Copilot_

---

## Executive Summary

{exec_summary}

---

## Risk Assessment

| Attribute | Value |
|---|---|
| Risk Score | **{_score(score_val)}** |
| Risk Tier | **{tier}** |
| Incoming Fail Rate | **{_pct(fail_rate)}** |
| Total Inspections | {n_insp:,} |
| Affected Serials | {n_serials:,} |
| Warranty Claims | **{n_claims}** |
| Component | {lot.get('component', '—')} |
| Supplier | {lot.get('supplier', '—')} |
| COO | {lot.get('coo', '—')} |
| Mfg Date | {lot.get('mfg_date', '—')} |

---

## Inspection Evidence

| Date | Characteristic | Value | Result | Defect Code |
|---|---|---|---|---|
{insp_rows or "_No inspection records._"}

_Showing up to 20 most recent records. Total: {n_insp:,}_

---

## Affected Units

{n_serials} finished-goods serial numbers contain components from lot {lot_no}:

{', '.join(str(s.get('serial_no','?')) for s in sers[:30]) or '_None found._'}
{'_(truncated to 30)_' if len(sers) > 30 else ''}

---

## Process Quality

{proc_summary}

---

## Field / Warranty Outcomes

| Serial No | Claim ID | Failure Date | Symptom | Severity | Region |
|---|---|---|---|---|---|
{warr_rows}

---

## Supplier Context

- **Supplier:** {sc.get('supplier', '—')}
- **Tier:** {sc.get('tier', '—')}
- **Quality Score:** {_score(sc.get('quality_score'))} / 100
- **Incoming fail rate:** {_pct(sc.get('incoming_fail_rate'))}
- **Warranty claim rate:** {_pct(sc.get('warranty_claim_rate'))}
- **Process Cpk:** {_fmt(sc.get('process_cpk'))}
- **Engineering maturity:** {_fmt(sc.get('engineering_maturity'))}
{coo_line}

---

## Recommended Actions

{action_lines}

---

## SAP Workflow

{sap_lines if sap_lines else _default_sap}

---

_Generated by AI Quality Copilot | Powered by Claude | {now}_
"""
