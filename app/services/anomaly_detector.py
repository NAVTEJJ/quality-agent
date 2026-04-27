"""
Anomaly detection over the Phase 2 KPIs (Phase 2, Step 2).

The :class:`AnomalyDetector` surfaces three families of anomalies:

* **Lot anomalies** -- composite lot-risk score above the configured HIGH cut.
* **Process anomalies** -- line/shift combinations whose torque fail rate is
  a statistical outlier versus the plant average.
* **Supplier anomalies** -- Watchlist tier, dominant fail rate, or elevated
  warranty claims.

Every anomaly is returned as a typed ``*Record`` dataclass: a small, JSON-
serialisable evidence bundle that downstream explainers consume without
re-querying the warehouse.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from sqlalchemy.engine import Engine

from configs import settings
from app.services.kpi_engine import KPIEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses -- evidence bundles passed to the explainer
# ---------------------------------------------------------------------------

@dataclass
class LotAnomalyRecord:
    lot_id: int
    lot_no: str
    supplier_id: Optional[int]
    supplier_name: Optional[str]
    supplier_tier: Optional[str]
    supplier_quality_score: Optional[float]
    component_id: Optional[int]
    component_name: Optional[str]
    fail_count: int
    total_inspections: int
    fail_rate: float
    linked_warranty_claims: int
    most_common_defect_code: Optional[str]
    distinct_inspection_dates_with_fails: int
    supplier_distinct_fail_dates: int
    lot_risk_score: float
    risk_tier: str


@dataclass
class ProcessAnomalyRecord:
    line: str
    shift: str
    total_builds: int
    torque_fails: int
    torque_fail_rate: float
    leak_fails: int
    leak_fail_rate: float
    plant_torque_fail_rate: float
    sigma_above_mean: float
    affected_materials: List[str]
    affected_serials: int
    distinct_build_dates_with_fails: int
    first_half_fail_rate: float
    second_half_fail_rate: float
    is_worsening: bool
    reference_line: Optional[str]
    reference_line_fails: Optional[int]


@dataclass
class SupplierAnomalyRecord:
    supplier_id: int
    supplier_name: str
    coo: Optional[str]
    tier: Optional[str]
    quality_score: Optional[float]
    incoming_fail_rate: Optional[float]
    warranty_claim_rate: Optional[float]
    process_drift_index: Optional[float]
    process_cpk: Optional[float]
    engineering_maturity: Optional[str]
    coo_incoming_fail_rate: Optional[float]
    coo_warranty_claim_rate: Optional[float]
    beats_coo_avg: Optional[str]
    reason_flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Detects lot, process, and supplier anomalies from Phase 2 KPIs."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ------------------------------------------------------------------
    # Lots
    # ------------------------------------------------------------------

    def detect_lot_anomalies(self, kpi_engine: KPIEngine) -> List[LotAnomalyRecord]:
        risk = kpi_engine.get_lot_risk_scores()
        threshold = settings.HIGH_RISK_LOT_SCORE_THRESHOLD
        flagged = risk[risk["lot_risk_score"] > threshold].copy()

        if flagged.empty:
            logger.info("No lot anomalies above score threshold %.2f", threshold)
            return []

        # Decorate with supplier/component names and supplier fail-date breadth.
        suppliers = pd.read_sql(
            "SELECT supplier_id, supplier_name FROM dim_supplier", self.engine
        )
        components = pd.read_sql(
            "SELECT component_id, component_name FROM dim_component", self.engine
        )
        sup_fail_dates = pd.read_sql(
            """
            SELECT supplier_id,
                   COUNT(DISTINCT DATE(insp_date)) AS n
            FROM fact_incoming_qm
            WHERE is_fail = 1 AND supplier_id IS NOT NULL
            GROUP BY supplier_id
            """,
            self.engine,
        )

        # Dominant defect code per lot and distinct failing dates per lot.
        per_lot = pd.read_sql(
            """
            SELECT lot_id, insp_date, defect_code, is_fail
            FROM fact_incoming_qm
            WHERE lot_id IS NOT NULL
            """,
            self.engine,
        )
        per_lot["insp_date"] = pd.to_datetime(per_lot["insp_date"], errors="coerce")

        records: List[LotAnomalyRecord] = []
        sup_name = dict(zip(suppliers["supplier_id"], suppliers["supplier_name"]))
        comp_name = dict(zip(components["component_id"], components["component_name"]))
        sup_fail_map = dict(zip(sup_fail_dates["supplier_id"], sup_fail_dates["n"]))

        for _, row in flagged.iterrows():
            lot_rows = per_lot[per_lot["lot_id"] == row["lot_id"]]
            fail_rows = lot_rows[lot_rows["is_fail"] == 1]

            # Dominant defect code (non-empty only).
            codes = fail_rows["defect_code"].replace("", pd.NA).dropna()
            dominant = (
                codes.value_counts().idxmax() if not codes.empty else None
            )

            # Distinct failing dates for this lot.
            distinct_fail_dates = fail_rows["insp_date"].dt.date.nunique()

            records.append(
                LotAnomalyRecord(
                    lot_id=int(row["lot_id"]),
                    lot_no=str(row["lot_no"]),
                    supplier_id=int(row["supplier_id"]) if pd.notna(row["supplier_id"]) else None,
                    supplier_name=sup_name.get(row["supplier_id"]),
                    supplier_tier=row.get("tier") if pd.notna(row.get("tier")) else None,
                    supplier_quality_score=None,  # populated below
                    component_id=int(row["component_id"]) if pd.notna(row["component_id"]) else None,
                    component_name=comp_name.get(row["component_id"]),
                    fail_count=int(row["total_fails"]),
                    total_inspections=int(row["total_inspections"]),
                    fail_rate=float(row["fail_rate"]),
                    linked_warranty_claims=int(row["claims_linked"]),
                    most_common_defect_code=dominant,
                    distinct_inspection_dates_with_fails=int(distinct_fail_dates),
                    supplier_distinct_fail_dates=int(sup_fail_map.get(row["supplier_id"], 0)),
                    lot_risk_score=float(row["lot_risk_score"]),
                    risk_tier=str(row["risk_tier"]),
                )
            )

        # Backfill supplier_quality_score in one query.
        sc = pd.read_sql(
            "SELECT supplier_id, quality_score FROM agg_supplier_scorecard", self.engine
        )
        qs_map = dict(zip(sc["supplier_id"], sc["quality_score"]))
        for rec in records:
            if rec.supplier_id is not None:
                qs = qs_map.get(rec.supplier_id)
                rec.supplier_quality_score = float(qs) if qs is not None else None

        logger.info("Detected %d lot anomalies", len(records))
        return records

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def detect_process_anomalies(self, kpi_engine: KPIEngine) -> List[ProcessAnomalyRecord]:
        drift = kpi_engine.get_drift_signals()
        if drift.empty:
            logger.info("No process drift signals above threshold.")
            return []

        # Plant-wide baseline and line/shift distribution.
        all_lines = kpi_engine.get_process_drift_by_line_shift()
        plant_mean = float(all_lines["torque_fail_rate"].mean())
        plant_std = float(all_lines["torque_fail_rate"].std(ddof=0)) or 1e-9

        # Reference line = line+shift with fewest torque fails (the "LINE-3" anchor).
        ref_row = all_lines.sort_values("torque_fails", ascending=True).iloc[0]
        ref_line = f"{ref_row['line']} {ref_row['shift']}"
        ref_fails = int(ref_row["torque_fails"])

        # Pull process detail once -- slice per anomalous line/shift.
        detail = pd.read_sql(
            """
            SELECT pm.line, pm.shift, pm.build_date, pm.serial_id,
                   pm.is_torque_fail,
                   dm.material_name
            FROM fact_process_measurements pm
            LEFT JOIN dim_material dm ON pm.finished_material_id = dm.material_id
            """,
            self.engine,
        )
        detail["build_date"] = pd.to_datetime(detail["build_date"], errors="coerce")

        records: List[ProcessAnomalyRecord] = []
        for _, row in drift.iterrows():
            subset = detail[(detail["line"] == row["line"]) & (detail["shift"] == row["shift"])]
            fails_subset = subset[subset["is_torque_fail"] == 1]

            # First-half vs second-half drift comparison, split by build_date median.
            first_half_rate = second_half_rate = 0.0
            if len(subset) >= 2 and subset["build_date"].notna().any():
                ordered = subset.sort_values("build_date").reset_index(drop=True)
                mid = len(ordered) // 2
                first = ordered.iloc[:mid]
                second = ordered.iloc[mid:]
                first_half_rate = float(first["is_torque_fail"].mean()) if len(first) else 0.0
                second_half_rate = float(second["is_torque_fail"].mean()) if len(second) else 0.0

            sigma = (float(row["torque_fail_rate"]) - plant_mean) / plant_std

            records.append(
                ProcessAnomalyRecord(
                    line=str(row["line"]),
                    shift=str(row["shift"]),
                    total_builds=int(row["total_builds"]),
                    torque_fails=int(row["torque_fails"]),
                    torque_fail_rate=float(row["torque_fail_rate"]),
                    leak_fails=int(row["leak_fails"]),
                    leak_fail_rate=float(row["leak_fail_rate"]),
                    plant_torque_fail_rate=plant_mean,
                    sigma_above_mean=sigma,
                    affected_materials=sorted({m for m in subset["material_name"].dropna().unique()}),
                    affected_serials=int(subset["serial_id"].nunique()),
                    distinct_build_dates_with_fails=int(fails_subset["build_date"].dt.date.nunique()),
                    first_half_fail_rate=first_half_rate,
                    second_half_fail_rate=second_half_rate,
                    is_worsening=second_half_rate > first_half_rate,
                    reference_line=ref_line,
                    reference_line_fails=ref_fails,
                )
            )

        logger.info("Detected %d process anomalies", len(records))
        return records

    # ------------------------------------------------------------------
    # Suppliers
    # ------------------------------------------------------------------

    def detect_supplier_anomalies(self, kpi_engine: KPIEngine) -> List[SupplierAnomalyRecord]:
        sql = """
            SELECT
                s.supplier_id,
                s.supplier_name,
                s.coo,
                s.engineering_maturity,
                s.process_cpk,
                sc.tier,
                sc.quality_score,
                sc.incoming_fail_rate,
                sc.warranty_claim_rate,
                sc.process_drift_index,
                cvs.coo_incoming_fail_rate,
                cvs.coo_warranty_claim_rate,
                cvs.beats_coo_avg
            FROM dim_supplier s
            LEFT JOIN agg_supplier_scorecard sc ON s.supplier_id = sc.supplier_id
            LEFT JOIN agg_coo_vs_supplier cvs   ON s.supplier_id = cvs.supplier_id
        """
        df = pd.read_sql(sql, self.engine)

        # Best supplier rate (excluding null/zero denominators).
        positive_rates = df["incoming_fail_rate"].dropna()
        best_rate = float(positive_rates.min()) if not positive_rates.empty else 0.0

        records: List[SupplierAnomalyRecord] = []
        for _, row in df.iterrows():
            flags: List[str] = []

            tier = row["tier"]
            if isinstance(tier, str) and tier.strip().lower() == "watchlist":
                flags.append("WATCHLIST_TIER")

            ir = row["incoming_fail_rate"]
            if pd.notna(ir) and best_rate > 0 and float(ir) > 2 * best_rate:
                flags.append("INCOMING_FAIL_RATE_2X_BEST")

            wcr = row["warranty_claim_rate"]
            if pd.notna(wcr) and float(wcr) > 0.04:
                flags.append("WARRANTY_CLAIM_RATE_ABOVE_4PCT")

            if not flags:
                continue

            records.append(
                SupplierAnomalyRecord(
                    supplier_id=int(row["supplier_id"]),
                    supplier_name=str(row["supplier_name"]),
                    coo=row["coo"] if pd.notna(row["coo"]) else None,
                    tier=tier if isinstance(tier, str) else None,
                    quality_score=float(row["quality_score"]) if pd.notna(row["quality_score"]) else None,
                    incoming_fail_rate=float(ir) if pd.notna(ir) else None,
                    warranty_claim_rate=float(wcr) if pd.notna(wcr) else None,
                    process_drift_index=float(row["process_drift_index"]) if pd.notna(row["process_drift_index"]) else None,
                    process_cpk=float(row["process_cpk"]) if pd.notna(row["process_cpk"]) else None,
                    engineering_maturity=row["engineering_maturity"] if pd.notna(row["engineering_maturity"]) else None,
                    coo_incoming_fail_rate=float(row["coo_incoming_fail_rate"]) if pd.notna(row["coo_incoming_fail_rate"]) else None,
                    coo_warranty_claim_rate=float(row["coo_warranty_claim_rate"]) if pd.notna(row["coo_warranty_claim_rate"]) else None,
                    beats_coo_avg=row["beats_coo_avg"] if pd.notna(row["beats_coo_avg"]) else None,
                    reason_flags=flags,
                )
            )

        logger.info("Detected %d supplier anomalies", len(records))
        return records


__all__ = [
    "AnomalyDetector",
    "LotAnomalyRecord",
    "ProcessAnomalyRecord",
    "SupplierAnomalyRecord",
]
