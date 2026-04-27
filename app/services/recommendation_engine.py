"""
Rule-based action recommendation engine (Phase 2, Step 3).

The :class:`RecommendationEngine` pairs every flagged anomaly with a
concrete, rank-ordered action list.  Rules combine:

* Phase 2 KPIs (lot risk score, drift rate, supplier tier)
* the authoritative ``ref_action_playbook`` from the source workbook
* operating thresholds from :mod:`configs.settings`

Every method returns a dict of the form::

    {"actions": [...], "sap_touchpoints": [...], "urgency": "IMMEDIATE"|...}

so that the UI can render the recommendations with severity badges and
link directly to the SAP / MES transaction codes the quality team uses.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.engine import Engine

from configs import settings
from app.services.kpi_engine import KPIEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier-keyed action templates
# ---------------------------------------------------------------------------

_TIER_SPECIFIC_ACTIONS: Dict[str, List[str]] = {
    "watchlist": [
        "Consider alternative supplier for critical builds",
        "Lock sourcing to current approved lots only pending capability review",
    ],
    "standard": [
        "Request corrective action plan (8D) from supplier",
        "Maintain current sampling frequency; re-baseline after 2 clean lots",
    ],
    "preferred": [
        "Eligible for reduced-inspection (audit-only) programme",
        "Preferred sourcing for new-program RFQs",
    ],
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """Produces ranked action lists for lot, process, and supplier anomalies."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._playbook = pd.read_sql("SELECT * FROM ref_action_playbook", engine)

    # ------------------------------------------------------------------
    # Playbook lookup
    # ------------------------------------------------------------------

    def _playbook_row(self, key: str) -> Optional[pd.Series]:
        mask = self._playbook["insight_type"].str.contains(key, case=False, na=False)
        hits = self._playbook[mask]
        return hits.iloc[0] if not hits.empty else None

    def _sap_touchpoints(self, key: str, fallback: List[str]) -> List[str]:
        row = self._playbook_row(key)
        if row is None:
            return list(fallback)
        raw = str(row["sap_mes_touchpoint"])
        parts = [p.strip() for p in raw.split("/") if p.strip()]
        return parts or list(fallback)

    # ------------------------------------------------------------------
    # Lot risk
    # ------------------------------------------------------------------

    def get_actions_for_lot_risk(
        self,
        lot_no: str,
        risk_score: float,
        supplier_tier: Optional[str],
    ) -> Dict[str, Any]:
        """Containment plan for a risky lot.

        The hierarchy:

        * **score > 0.6 AND tier = Watchlist** -> block, 8D, 100% sampling
        * **score > 0.3**                      -> heightened sampling + watch
        * **otherwise**                        -> routine monitoring
        """
        tier = (supplier_tier or "").strip().lower()

        if risk_score > 0.6 and tier == "watchlist":
            actions = [
                f"Block lot {lot_no} pending review",
                "Create supplier 8D",
                "Increase incoming sampling to 100%",
                "Trigger containment for affected serials",
            ]
            urgency = "IMMEDIATE"
        elif risk_score > 0.3:
            actions = [
                f"Increase incoming sampling for lot {lot_no}",
                "Monitor next lot closely",
                "Review supplier corrective actions",
            ]
            urgency = "ELEVATED"
        else:
            actions = [
                "Maintain routine sampling cadence",
                "Log observation for trend monitoring",
            ]
            urgency = "ROUTINE"

        return {
            "lot_no":          lot_no,
            "risk_score":      round(float(risk_score), 3),
            "supplier_tier":   supplier_tier,
            "urgency":         urgency,
            "actions":         actions,
            "sap_touchpoints": self._sap_touchpoints(
                "High lot defect rate", ["QA32", "QE51N", "QM01"]
            ),
        }

    # ------------------------------------------------------------------
    # Process drift
    # ------------------------------------------------------------------

    def get_actions_for_process_drift(
        self,
        line: str,
        shift: str,
        fail_rate: float,
    ) -> Dict[str, Any]:
        """Calibration + controls playbook for a drifting line/shift."""
        actions = [
            "Verify torque tool calibration immediately",
            "Increase in-process torque check frequency",
            f"Refresh work instructions for {shift} shift",
            "Add handover controls between shifts",
            "Internal quality notification",
        ]

        # Urgency ladder based on the fail rate.
        if fail_rate >= 0.20:
            urgency = "IMMEDIATE"
        elif fail_rate > settings.PROCESS_DRIFT_FAIL_RATE_THRESHOLD:
            urgency = "ELEVATED"
        else:
            urgency = "ROUTINE"

        return {
            "line":            line,
            "shift":           shift,
            "fail_rate":       round(float(fail_rate), 4),
            "urgency":         urgency,
            "actions":         actions,
            "sap_touchpoints": self._sap_touchpoints(
                "Process drift", ["MES checks", "Internal QN"]
            ),
        }

    # ------------------------------------------------------------------
    # Supplier risk
    # ------------------------------------------------------------------

    def get_actions_for_supplier_risk(self, supplier_id: int) -> Dict[str, Any]:
        """Supplier-level playbook, tailored by tier."""
        sql = """
            SELECT
                s.supplier_name, s.coo,
                sc.tier, sc.quality_score, sc.premium_service_fit
            FROM dim_supplier s
            LEFT JOIN agg_supplier_scorecard sc ON s.supplier_id = sc.supplier_id
            WHERE s.supplier_id = :sid
            LIMIT 1
        """
        df = pd.read_sql(sql, self.engine, params={"sid": int(supplier_id)})
        if df.empty:
            raise LookupError(f"Supplier id={supplier_id} not found")
        row = df.iloc[0]
        tier = (row["tier"] or "").strip().lower() if pd.notna(row["tier"]) else ""

        # Base actions from the playbook.
        playbook_row = self._playbook_row("Warranty claim spike")
        base_actions: List[str] = []
        if playbook_row is not None:
            base_actions = [
                part.strip() for part in str(playbook_row["typical_action"]).split(";")
                if part.strip()
            ]

        tier_actions = _TIER_SPECIFIC_ACTIONS.get(tier, [])

        # Preferred suppliers also earn a strategic-sourcing nudge.
        if row.get("premium_service_fit") == "Yes":
            tier_actions = tier_actions + [
                "Evaluate for premium / safety-critical programme allocation",
            ]

        urgency = {
            "watchlist": "IMMEDIATE",
            "standard":  "ELEVATED",
            "preferred": "ROUTINE",
        }.get(tier, "ELEVATED")

        return {
            "supplier_id":     int(supplier_id),
            "supplier":        str(row["supplier_name"]),
            "coo":             row["coo"] if pd.notna(row["coo"]) else None,
            "tier":            row["tier"] if pd.notna(row["tier"]) else None,
            "quality_score":   float(row["quality_score"]) if pd.notna(row["quality_score"]) else None,
            "urgency":         urgency,
            "actions":         base_actions + tier_actions,
            "sap_touchpoints": self._sap_touchpoints(
                "Warranty claim spike", ["QM01", "Analytics"]
            ),
        }

    # ------------------------------------------------------------------
    # Inspection strategy (plant-wide)
    # ------------------------------------------------------------------

    def get_inspection_strategy(self) -> Dict[str, List[Dict[str, Any]]]:
        """Plant-wide inspection rebalance plan.

        Keys:
            * ``increase_sampling`` -- HIGH / MEDIUM risk lots (descending).
            * ``reduce_inspection`` -- Preferred-tier suppliers with premium fit.
            * ``move_to_watchlist`` -- suppliers with 2+ HIGH-risk lots.
        """
        kpi = KPIEngine(self.engine)
        risk = kpi.get_lot_risk_scores()
        suppliers = kpi.get_supplier_rankings()

        # Lookup maps for human-readable decoration.
        sup_name_map = pd.read_sql(
            "SELECT supplier_id, supplier_name FROM dim_supplier", self.engine
        ).set_index("supplier_id")["supplier_name"].to_dict()
        comp_name_map = pd.read_sql(
            "SELECT component_id, component_name FROM dim_component", self.engine
        ).set_index("component_id")["component_name"].to_dict()

        # Increase sampling: HIGH first, MEDIUM second.
        elevated = risk[risk["risk_tier"].isin(["HIGH", "MEDIUM"])].head(20)
        increase_sampling = [
            {
                "lot_no":     str(row["lot_no"]),
                "component":  comp_name_map.get(row["component_id"]),
                "supplier":   sup_name_map.get(row["supplier_id"]),
                "risk_score": round(float(row["lot_risk_score"]), 3),
                "risk_tier":  str(row["risk_tier"]),
                "fail_rate":  round(float(row["fail_rate"]), 4),
                "action":     "Increase sampling" if row["risk_tier"] == "MEDIUM"
                              else "Increase sampling to 100% + block pending review",
            }
            for _, row in elevated.iterrows()
        ]

        # Reduce inspection: tier = Preferred.
        preferred = suppliers[
            suppliers["tier"].fillna("").str.lower() == "preferred"
        ]
        reduce_inspection = [
            {
                "supplier":            str(row["supplier"]),
                "coo":                 row["coo"] if pd.notna(row["coo"]) else None,
                "tier":                str(row["tier"]),
                "quality_score":       float(row["quality_score"]) if pd.notna(row["quality_score"]) else None,
                "premium_service_fit": row.get("premium_service_fit"),
                "action":              "Move to audit-only inspection programme",
            }
            for _, row in preferred.iterrows()
        ]

        # Move to watchlist: suppliers with 2+ HIGH-risk lots who aren't already Watchlist.
        high_lots = risk[risk["risk_tier"] == "HIGH"]
        per_supplier = (
            high_lots.groupby("supplier_id")
            .agg(n_high_lots=("lot_no", "count"), components=("component_id", "nunique"))
            .reset_index()
        )
        to_watchlist_rows = per_supplier[per_supplier["n_high_lots"] >= 2]
        move_to_watchlist: List[Dict[str, Any]] = []
        for _, row in to_watchlist_rows.iterrows():
            sup_id = int(row["supplier_id"])
            sup_info = suppliers[suppliers["supplier_id"] == sup_id]
            current_tier = (
                str(sup_info.iloc[0]["tier"]) if not sup_info.empty and pd.notna(sup_info.iloc[0]["tier"])
                else None
            )
            # Skip suppliers already flagged Watchlist.
            if current_tier and current_tier.lower() == "watchlist":
                continue
            components = [
                comp_name_map.get(cid)
                for cid in high_lots[high_lots["supplier_id"] == sup_id]["component_id"].unique()
                if comp_name_map.get(cid)
            ]
            move_to_watchlist.append({
                "supplier":     sup_name_map.get(sup_id),
                "supplier_id":  sup_id,
                "current_tier": current_tier,
                "n_high_lots":  int(row["n_high_lots"]),
                "components":   components,
                "reason":       f"{int(row['n_high_lots'])} HIGH-risk lots across {len(components)} component(s)",
                "action":       "Escalate supplier to Watchlist tier",
            })

        return {
            "increase_sampling": increase_sampling,
            "reduce_inspection": reduce_inspection,
            "move_to_watchlist": move_to_watchlist,
        }


__all__ = ["RecommendationEngine"]
