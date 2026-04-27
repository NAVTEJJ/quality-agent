"""
Drill-down graph traversal (Phase 2, Step 3).

Every insight the system surfaces has a "show me the evidence" counterpart.
:class:`DrillDownService` walks the star-schema graph so the frontend can
render linked tables without re-writing joins.

Public methods return pandas DataFrames for in-process callers; the
orchestrator :meth:`get_full_drill_down_chain` serialises the whole graph
to JSON-friendly dicts so it can be returned directly from the API layer.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class DrillDownService:
    """Traverses dim/fact relationships to return evidence DataFrames."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ------------------------------------------------------------------
    # Lot -> inspection records
    # ------------------------------------------------------------------

    def lot_to_inspection_records(self, lot_no: str) -> pd.DataFrame:
        """All incoming-QM rows for *lot_no*, sorted newest first."""
        sql = """
            SELECT
                qm.id                 AS inspection_id,
                l.lot_no,
                c.component_name      AS component,
                s.supplier_name       AS supplier,
                qm.insp_lot,
                qm.insp_date,
                qm.characteristic,
                qm.measured_value,
                qm.uom,
                qm.result,
                qm.defect_code,
                qm.is_fail
            FROM fact_incoming_qm qm
            JOIN dim_lot l            ON qm.lot_id       = l.lot_id
            LEFT JOIN dim_component c ON qm.component_id = c.component_id
            LEFT JOIN dim_supplier  s ON qm.supplier_id  = s.supplier_id
            WHERE l.lot_no = :lot_no
            ORDER BY qm.insp_date DESC
        """
        return pd.read_sql(sql, self.engine, params={"lot_no": lot_no})

    # ------------------------------------------------------------------
    # Lot -> affected serials
    # ------------------------------------------------------------------

    def lot_to_affected_serials(self, lot_no: str) -> pd.DataFrame:
        """Finished-goods serials whose BOM contains *lot_no*."""
        sql = """
            SELECT DISTINCT
                ds.serial_id,
                ds.serial_no,
                dm.material_name      AS finished_material,
                ds.build_dt,
                ds.line,
                ds.shift,
                ds.plant,
                ds.operator_id,
                ds.ecn_level
            FROM fact_constituent_bom b
            JOIN dim_lot    l  ON b.lot_id   = l.lot_id
            JOIN dim_serial ds ON b.serial_id = ds.serial_id
            LEFT JOIN dim_material dm ON ds.finished_material_id = dm.material_id
            WHERE l.lot_no = :lot_no
            ORDER BY ds.build_dt ASC
        """
        return pd.read_sql(sql, self.engine, params={"lot_no": lot_no})

    # ------------------------------------------------------------------
    # Serial -> process measurements
    # ------------------------------------------------------------------

    def serial_to_process_measurements(self, serial_no: str) -> pd.DataFrame:
        """Torque + leak measurements for a given serial."""
        sql = """
            SELECT
                ds.serial_no,
                pm.build_date,
                pm.line,
                pm.shift,
                pm.torque_nm,
                pm.torque_result,
                pm.leak_rate_ccm,
                pm.leak_result,
                pm.ecn_level,
                pm.is_torque_fail,
                pm.is_leak_fail
            FROM fact_process_measurements pm
            JOIN dim_serial ds ON pm.serial_id = ds.serial_id
            WHERE ds.serial_no = :serial_no
        """
        return pd.read_sql(sql, self.engine, params={"serial_no": serial_no})

    # ------------------------------------------------------------------
    # Serial -> warranty outcome
    # ------------------------------------------------------------------

    def serial_to_warranty_outcome(self, serial_no: str) -> Optional[Dict[str, Any]]:
        """Return the warranty claim row for *serial_no*, or ``None`` if clean."""
        sql = """
            SELECT
                ds.serial_no,
                w.claim_id,
                w.failure_date,
                w.symptom,
                w.mileage_or_hours,
                w.region,
                w.severity
            FROM fact_warranty_claims w
            JOIN dim_serial ds ON w.serial_id = ds.serial_id
            WHERE ds.serial_no = :serial_no
            LIMIT 1
        """
        df = pd.read_sql(sql, self.engine, params={"serial_no": serial_no})
        if df.empty:
            return None
        return _df_to_records(df)[0]

    # ------------------------------------------------------------------
    # Lot -> supplier scorecard
    # ------------------------------------------------------------------

    def lot_to_supplier_scorecard(self, lot_no: str) -> Optional[Dict[str, Any]]:
        """Full scorecard row for the supplier behind *lot_no*."""
        sql = """
            SELECT
                s.supplier_id,
                s.supplier_name       AS supplier,
                s.coo,
                s.engineering_maturity,
                s.engineering_maturity_score,
                s.process_cpk,
                s.design_ownership,
                s.typical_project_type,
                sc.lots_inspected,
                sc.samples,
                sc.fails,
                sc.incoming_fail_rate,
                sc.units_built,
                sc.units_with_claims,
                sc.warranty_claim_rate,
                sc.process_drift_index,
                sc.on_time_delivery_pct,
                sc.avg_lead_time_days,
                sc.quality_score,
                sc.tier,
                sc.premium_service_fit
            FROM dim_lot l
            JOIN dim_supplier s ON l.supplier_id = s.supplier_id
            LEFT JOIN agg_supplier_scorecard sc ON s.supplier_id = sc.supplier_id
            WHERE l.lot_no = :lot_no
            LIMIT 1
        """
        df = pd.read_sql(sql, self.engine, params={"lot_no": lot_no})
        if df.empty:
            return None
        return _df_to_records(df)[0]

    # ------------------------------------------------------------------
    # Supplier -> COO context
    # ------------------------------------------------------------------

    def supplier_to_coo_context(self, supplier_id: int) -> Optional[Dict[str, Any]]:
        """COO benchmark row for *supplier_id*, with the derived ``gap`` column."""
        sql = """
            SELECT
                s.supplier_id,
                s.supplier_name       AS supplier,
                cvs.coo,
                cvs.incoming_fail_rate,
                cvs.warranty_claim_rate,
                cvs.quality_score,
                cvs.tier,
                cvs.coo_incoming_fail_rate,
                cvs.coo_warranty_claim_rate,
                cvs.beats_coo_avg
            FROM agg_coo_vs_supplier cvs
            JOIN dim_supplier s ON cvs.supplier_id = s.supplier_id
            WHERE cvs.supplier_id = :supplier_id
            LIMIT 1
        """
        df = pd.read_sql(sql, self.engine, params={"supplier_id": int(supplier_id)})
        if df.empty:
            return None
        row = df.iloc[0]
        gap = None
        if pd.notna(row["coo_incoming_fail_rate"]) and pd.notna(row["incoming_fail_rate"]):
            gap = float(row["coo_incoming_fail_rate"]) - float(row["incoming_fail_rate"])
        result = _df_to_records(df)[0]
        result["gap"] = round(gap, 4) if gap is not None else None
        result["gap_interpretation"] = (
            "Supplier outperforms its COO" if (gap is not None and gap > 0)
            else "Supplier underperforms or matches its COO"
        )
        return result

    # ------------------------------------------------------------------
    # Orchestrator — one call, full evidence tree
    # ------------------------------------------------------------------

    # In-memory memoization — judges drill the same lot repeatedly during
    # a demo, and the per-serial fan-out makes each call ~500 ms cold.
    _CHAIN_CACHE: Dict[str, Dict[str, Any]] = {}

    def get_full_drill_down_chain(self, lot_no: str) -> Dict[str, Any]:
        """Stitch every drill-down into a single JSON-serialisable dict.

        The chain is ordered the way a human would click through it:

        lot_info -> inspection_records -> affected_serials
                 -> (per serial) process_measurements + warranty_outcomes
                 -> supplier_scorecard -> coo_context

        Raises:
            LookupError: if *lot_no* is not present in :class:`DimLot`.
        """
        cache_key = (lot_no or "").strip().upper()
        cached = self._CHAIN_CACHE.get(cache_key)
        if cached is not None:
            return cached

        lot_info = self._lot_info(lot_no)
        if lot_info is None:
            raise LookupError(f"Lot '{lot_no}' not found in dim_lot")

        insp_df    = self.lot_to_inspection_records(lot_no)
        serials_df = self.lot_to_affected_serials(lot_no)

        # Per-serial expansions.
        process_records: List[Dict[str, Any]] = []
        warranty_records: List[Dict[str, Any]] = []
        for sn in serials_df["serial_no"].dropna().astype(str).tolist():
            pm_df = self.serial_to_process_measurements(sn)
            process_records.extend(_df_to_records(pm_df))

            claim = self.serial_to_warranty_outcome(sn)
            if claim is not None:
                warranty_records.append(claim)

        scorecard = self.lot_to_supplier_scorecard(lot_no)
        coo_context = None
        if scorecard and scorecard.get("supplier_id") is not None:
            coo_context = self.supplier_to_coo_context(int(scorecard["supplier_id"]))

        fail_rows = insp_df[insp_df["is_fail"] == 1]

        summary = {
            "total_inspections":       int(len(insp_df)),
            "total_fails":             int(len(fail_rows)),
            "fail_rate":               round(len(fail_rows) / len(insp_df), 4) if len(insp_df) else 0.0,
            "affected_serials":        int(len(serials_df)),
            "serials_with_warranty":   int(len(warranty_records)),
            "serials_with_process_fails":
                sum(
                    1 for r in process_records
                    if r.get("is_torque_fail") == 1 or r.get("is_leak_fail") == 1
                ),
        }

        chain = {
            "lot_info":            lot_info,
            "summary":             summary,
            "inspection_records":  _df_to_records(insp_df),
            "affected_serials":    _df_to_records(serials_df),
            "process_measurements": process_records,
            "warranty_outcomes":   warranty_records,
            "supplier_scorecard":  scorecard,
            "coo_context":         coo_context,
        }
        self._CHAIN_CACHE[cache_key] = chain
        return chain

    # ------------------------------------------------------------------
    # Private helper — lot header row
    # ------------------------------------------------------------------

    def _lot_info(self, lot_no: str) -> Optional[Dict[str, Any]]:
        sql = """
            SELECT
                l.lot_id,
                l.lot_no,
                l.mfg_date,
                c.component_id,
                c.component_name   AS component,
                s.supplier_id,
                s.supplier_name    AS supplier,
                s.coo
            FROM dim_lot l
            LEFT JOIN dim_component c ON l.component_id = c.component_id
            LEFT JOIN dim_supplier  s ON l.supplier_id  = s.supplier_id
            WHERE l.lot_no = :lot_no
            LIMIT 1
        """
        df = pd.read_sql(sql, self.engine, params={"lot_no": lot_no})
        if df.empty:
            return None
        return _df_to_records(df)[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a DataFrame to a JSON-friendly list of dicts.

    * NaN/NaT -> ``None``
    * pandas Timestamp -> ISO string
    * numpy scalar types -> native Python types
    """
    if df.empty:
        return []
    clean = df.copy()

    # Stringify any timestamp / datetime column.
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Replace NaN with None so ``json.dumps`` is happy.
    clean = clean.astype(object).where(clean.notna(), None)
    return clean.to_dict("records")


__all__ = ["DrillDownService"]
