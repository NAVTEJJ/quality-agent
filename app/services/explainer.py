"""
Explainability layer (Phase 2, Step 2).

Every anomaly in the warehouse is surfaced as an :class:`InsightExplanation`
-- a structured, JSON-serialisable object containing:

* a human-readable headline,
* 3-5 business-language reasons,
* the raw evidence behind each reason,
* a likely cause,
* recommended actions drawn from ``ref_action_playbook``,
* a confidence score, risk level, drill-down hints, and SAP/MES touchpoints.

The :func:`generate_all_insights` entry point runs the full detection +
explanation pipeline and persists the output to ``insights.json``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy.engine import Engine

from configs import settings
from app.services.anomaly_detector import (
    AnomalyDetector,
    LotAnomalyRecord,
    ProcessAnomalyRecord,
    SupplierAnomalyRecord,
)
from app.services.kpi_engine import KPIEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class InsightExplanation:
    """One structured explanation.  This is the schema the UI + API surface."""

    insight_type: str
    entity_id: str
    entity_name: str
    headline: str
    why: List[str]
    evidence: Dict
    likely_cause: str
    recommended_actions: List[str]
    confidence: float
    risk_level: str
    drill_down_hints: List[str]
    sap_touchpoints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class InsightExplainer:
    """Converts :mod:`anomaly_detector` records into ``InsightExplanation``s."""

    # Default fallbacks -- used when the playbook lookup misses.
    _FALLBACK_SAP: Dict[str, List[str]] = {
        "LOT_RISK":        ["QA32", "QE51N", "QM01"],
        "PROCESS_DRIFT":   ["MES checks", "Internal QN"],
        "SUPPLIER_RISK":   ["QM01", "Analytics"],
        "COO_TREND":       ["Sourcing Dashboard", "Vendor segmentation"],
    }

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._playbook = pd.read_sql("SELECT * FROM ref_action_playbook", engine)
        self._supplier_by_id = self._load_suppliers()

    # ------------------------------------------------------------------
    # Playbook lookup
    # ------------------------------------------------------------------

    def _playbook_row(self, key: str) -> Optional[pd.Series]:
        """Return the first playbook row whose insight_type contains *key* (case-insensitive)."""
        mask = self._playbook["insight_type"].str.contains(key, case=False, na=False)
        hits = self._playbook[mask]
        return hits.iloc[0] if not hits.empty else None

    def _actions_from_playbook(self, key: str) -> List[str]:
        row = self._playbook_row(key)
        if row is None:
            return []
        raw = str(row["typical_action"])
        # Actions are "a; b; c" -- split on semicolons.
        return [part.strip() for part in raw.split(";") if part.strip()]

    def _touchpoints_from_playbook(self, key: str, fallback_type: str) -> List[str]:
        row = self._playbook_row(key)
        if row is None:
            return list(self._FALLBACK_SAP[fallback_type])
        raw = str(row["sap_mes_touchpoint"])
        return [part.strip() for part in raw.split("/") if part.strip()]

    # ------------------------------------------------------------------
    # Cached supplier name resolver
    # ------------------------------------------------------------------

    def _load_suppliers(self) -> Dict[int, Dict]:
        sql = """
            SELECT
                s.supplier_id, s.supplier_name, s.coo,
                sc.tier, sc.quality_score
            FROM dim_supplier s
            LEFT JOIN agg_supplier_scorecard sc ON s.supplier_id = sc.supplier_id
        """
        df = pd.read_sql(sql, self.engine)
        return {int(r["supplier_id"]): r.to_dict() for _, r in df.iterrows()}

    # ------------------------------------------------------------------
    # Lot risk
    # ------------------------------------------------------------------

    def explain_lot_risk(
        self,
        lot_anomaly: LotAnomalyRecord,
        kpi_engine: KPIEngine,
    ) -> InsightExplanation:
        sup_name = lot_anomaly.supplier_name or "Unknown supplier"
        sup_tier = lot_anomaly.supplier_tier or "n/a"
        threshold_pct = settings.LOT_RISK_FAIL_RATE_THRESHOLD * 100

        why: List[str] = []

        # Bullet 1 -- fail rate vs threshold (verbatim phrasing).
        why.append(
            f"Incoming fail rate {lot_anomaly.fail_rate:.1%} "
            f"-- above the {threshold_pct:.0f}% threshold"
        )

        # Bullet 2 -- field warranty linkage.
        if lot_anomaly.linked_warranty_claims > 0:
            why.append(
                f"Linked to {lot_anomaly.linked_warranty_claims} warranty "
                f"claim{'s' if lot_anomaly.linked_warranty_claims != 1 else ''} "
                f"in the field"
            )
        else:
            why.append("No warranty claims linked yet -- leading indicator only")

        # Bullet 3 -- supplier tier, always cited explicitly.
        why.append(f"Supplier {sup_name} has {sup_tier} tier status")

        # Bullet 4 -- recurrence: prefer lot-level if we have it, else supplier-level.
        if lot_anomaly.distinct_inspection_dates_with_fails >= 2:
            why.append(
                f"Multiple inspection dates "
                f"({lot_anomaly.distinct_inspection_dates_with_fails}) "
                f"show recurring failures on this lot"
            )
        elif lot_anomaly.supplier_distinct_fail_dates >= 2:
            why.append(
                f"Supplier has failures on {lot_anomaly.supplier_distinct_fail_dates} "
                f"distinct inspection dates -- recurring pattern, not a one-off"
            )

        # Bullet 5 (optional) -- dominant defect code.
        if lot_anomaly.most_common_defect_code:
            why.append(
                f"Dominant defect code: {lot_anomaly.most_common_defect_code}"
            )

        evidence: Dict = {
            "lot_no":                     lot_anomaly.lot_no,
            "supplier":                   sup_name,
            "supplier_tier":              sup_tier,
            "supplier_quality_score":     lot_anomaly.supplier_quality_score,
            "component":                  lot_anomaly.component_name,
            "total_inspections":          lot_anomaly.total_inspections,
            "fail_count":                 lot_anomaly.fail_count,
            "fail_rate":                  round(lot_anomaly.fail_rate, 4),
            "linked_warranty_claims":     lot_anomaly.linked_warranty_claims,
            "most_common_defect_code":    lot_anomaly.most_common_defect_code,
            "distinct_fail_dates_lot":    lot_anomaly.distinct_inspection_dates_with_fails,
            "distinct_fail_dates_supplier": lot_anomaly.supplier_distinct_fail_dates,
            "lot_risk_score":             round(lot_anomaly.lot_risk_score, 4),
            "threshold":                  settings.HIGH_RISK_LOT_SCORE_THRESHOLD,
        }

        likely_cause = self._infer_lot_cause(lot_anomaly)
        confidence = min(0.99, 0.5 + lot_anomaly.lot_risk_score * 0.5)

        return InsightExplanation(
            insight_type="LOT_RISK",
            entity_id=str(lot_anomaly.lot_id),
            entity_name=lot_anomaly.lot_no,
            headline=(
                f"Lot {lot_anomaly.lot_no} shows elevated incoming risk "
                f"-- immediate action required"
            ),
            why=why,
            evidence=evidence,
            likely_cause=likely_cause,
            recommended_actions=self._actions_from_playbook("High lot defect rate")
                or ["Increase sampling", "Block lot pending review", "Open supplier 8D"],
            confidence=round(confidence, 3),
            risk_level=lot_anomaly.risk_tier,
            drill_down_hints=[
                f"Pull all characteristics measured on lot {lot_anomaly.lot_no}",
                f"Trace serials containing {lot_anomaly.component_name or 'this component'} from this lot via BOM",
                f"Request supplier 8D from {sup_name}",
                "Check ECN history for this component / lot timeframe",
            ],
            sap_touchpoints=self._touchpoints_from_playbook(
                "High lot defect rate", "LOT_RISK"
            ),
        )

    @staticmethod
    def _infer_lot_cause(rec: LotAnomalyRecord) -> str:
        """Heuristic cause inference from the evidence."""
        if rec.linked_warranty_claims >= 5 and (rec.supplier_tier or "").lower() == "watchlist":
            return (
                "Systemic supplier quality issue -- the lot's defects have "
                "propagated to field warranty claims, indicating an inspection "
                "escape rather than a one-off."
            )
        if rec.fail_rate >= 0.25:
            return "Lot-level variation at supplier -- likely inbound quality escape."
        if rec.linked_warranty_claims >= 3:
            return "Escape from incoming QM -- field failures trace back to this lot."
        return "Incoming quality drift at supplier -- containment warranted."

    # ------------------------------------------------------------------
    # Process drift
    # ------------------------------------------------------------------

    def explain_process_drift(
        self,
        process_anomaly: ProcessAnomalyRecord,
        kpi_engine: KPIEngine,
    ) -> InsightExplanation:
        why: List[str] = []

        # Bullet 1 -- direct comparison against the best-performing reference.
        if process_anomaly.reference_line is not None:
            why.append(
                f"{process_anomaly.torque_fails} torque failures on "
                f"{process_anomaly.line} {process_anomaly.shift} vs "
                f"{process_anomaly.reference_line_fails} on "
                f"{process_anomaly.reference_line} over the same window"
            )
        else:
            why.append(
                f"{process_anomaly.torque_fails} torque failures observed on "
                f"{process_anomaly.line} {process_anomaly.shift}"
            )

        # Bullet 2 -- sigma above plant average.
        why.append(
            f"Failure rate {process_anomaly.sigma_above_mean:.1f}sigma above the "
            f"plant average ({process_anomaly.plant_torque_fail_rate:.1%})"
        )

        # Bullet 3 -- multi-date pattern.
        if process_anomaly.distinct_build_dates_with_fails >= 2:
            why.append(
                f"Pattern consistent across "
                f"{process_anomaly.distinct_build_dates_with_fails} distinct "
                f"build dates -- not a single bad shift"
            )
        else:
            why.append("Failures concentrated -- monitor for repeat occurrence")

        # Bullet 4 -- trend direction.
        trend = (
            "worsening" if process_anomaly.is_worsening
            else "stable / improving"
        )
        why.append(
            f"Trend {trend} -- first-half fail rate "
            f"{process_anomaly.first_half_fail_rate:.1%} vs second-half "
            f"{process_anomaly.second_half_fail_rate:.1%}"
        )

        # Bullet 5 -- scope of exposure.
        if process_anomaly.affected_materials:
            why.append(
                f"Affects {len(process_anomaly.affected_materials)} finished "
                f"material(s): {', '.join(process_anomaly.affected_materials)}"
            )

        evidence = {
            "line":                  process_anomaly.line,
            "shift":                 process_anomaly.shift,
            "total_builds":          process_anomaly.total_builds,
            "torque_fails":          process_anomaly.torque_fails,
            "torque_fail_rate":      round(process_anomaly.torque_fail_rate, 4),
            "leak_fails":            process_anomaly.leak_fails,
            "leak_fail_rate":        round(process_anomaly.leak_fail_rate, 4),
            "plant_torque_fail_rate": round(process_anomaly.plant_torque_fail_rate, 4),
            "sigma_above_mean":      round(process_anomaly.sigma_above_mean, 3),
            "affected_serials":      process_anomaly.affected_serials,
            "affected_materials":    process_anomaly.affected_materials,
            "first_half_fail_rate":  round(process_anomaly.first_half_fail_rate, 4),
            "second_half_fail_rate": round(process_anomaly.second_half_fail_rate, 4),
            "is_worsening":          process_anomaly.is_worsening,
            "distinct_fail_dates":   process_anomaly.distinct_build_dates_with_fails,
            "reference_line":        process_anomaly.reference_line,
            "reference_line_fails":  process_anomaly.reference_line_fails,
            "threshold":             settings.PROCESS_DRIFT_FAIL_RATE_THRESHOLD,
        }

        likely_cause = self._infer_process_cause(process_anomaly)

        # Confidence scales with sigma and multi-date consistency.
        base_conf = 0.55 + min(abs(process_anomaly.sigma_above_mean), 4.0) / 10.0
        if process_anomaly.distinct_build_dates_with_fails >= 3:
            base_conf += 0.1
        confidence = round(min(0.99, base_conf), 3)

        risk_level = (
            "HIGH" if process_anomaly.torque_fail_rate >= 0.15
            else "MEDIUM"
        )

        return InsightExplanation(
            insight_type="PROCESS_DRIFT",
            entity_id=f"{process_anomaly.line}::{process_anomaly.shift}",
            entity_name=f"{process_anomaly.line} {process_anomaly.shift}",
            headline=(
                f"Process drift detected on {process_anomaly.line} "
                f"{process_anomaly.shift} shift -- torque failures above threshold"
            ),
            why=why,
            evidence=evidence,
            likely_cause=likely_cause,
            recommended_actions=self._actions_from_playbook("Process drift")
                or [
                    "Calibrate torque tools",
                    "Refresh work instructions",
                    "Increase in-process torque check frequency",
                ],
            confidence=confidence,
            risk_level=risk_level,
            drill_down_hints=[
                f"Review torque tool calibration records for {process_anomaly.line}",
                f"Compare operator IDs across failing serials on {process_anomaly.shift} shift",
                "Check shift-handover logs for missed checks",
                "Pull MES torque traces for the affected build window",
            ],
            sap_touchpoints=self._touchpoints_from_playbook(
                "Process drift", "PROCESS_DRIFT"
            ),
        )

    @staticmethod
    def _infer_process_cause(rec: ProcessAnomalyRecord) -> str:
        if rec.is_worsening and rec.distinct_build_dates_with_fails >= 3:
            return (
                "Progressive tool drift -- failure rate rising across build dates, "
                "suggesting calibration degradation or wear."
            )
        if rec.shift.lower() == "night":
            return (
                "Shift-specific practice issue -- night shift concentration "
                "points to staffing, handover, or supervision effects rather "
                "than hardware."
            )
        return "Process-side variation on a specific line/shift -- diagnose at source."

    # ------------------------------------------------------------------
    # Supplier risk
    # ------------------------------------------------------------------

    def explain_supplier_risk(
        self,
        supplier_anomaly: SupplierAnomalyRecord,
        kpi_engine: KPIEngine,
    ) -> InsightExplanation:
        name = supplier_anomaly.supplier_name
        coo = supplier_anomaly.coo or "Unknown"

        why: List[str] = []

        # Bullet 1 -- tier & quality score.
        if supplier_anomaly.tier:
            qs = supplier_anomaly.quality_score
            qs_str = f" (quality score {qs:.0f}/100)" if qs is not None else ""
            why.append(f"{name} rated {supplier_anomaly.tier}{qs_str}")

        # Bullet 2 -- COO decomposition (the nuanced insight).
        if (
            supplier_anomaly.coo_incoming_fail_rate is not None
            and supplier_anomaly.incoming_fail_rate is not None
        ):
            if supplier_anomaly.beats_coo_avg == "Yes":
                why.append(
                    f"Outperforms {coo} country average "
                    f"({supplier_anomaly.incoming_fail_rate:.1%} vs "
                    f"{supplier_anomaly.coo_incoming_fail_rate:.1%}) -- "
                    f"issue is supplier-specific, not a country effect"
                )
            else:
                why.append(
                    f"Underperforms {coo} country average "
                    f"({supplier_anomaly.incoming_fail_rate:.1%} vs "
                    f"{supplier_anomaly.coo_incoming_fail_rate:.1%}) -- "
                    f"both country and supplier factors in play"
                )

        # Bullet 3 -- warranty.
        if (
            supplier_anomaly.warranty_claim_rate is not None
            and supplier_anomaly.warranty_claim_rate > 0
        ):
            why.append(
                f"Warranty claim rate {supplier_anomaly.warranty_claim_rate:.1%} "
                f"across delivered units"
            )

        # Bullet 4 -- process drift index.
        if (
            supplier_anomaly.process_drift_index is not None
            and supplier_anomaly.process_drift_index > 0
        ):
            why.append(
                f"Process drift index {supplier_anomaly.process_drift_index:.3f} "
                f"-- internal capability signal"
            )

        # Bullet 5 -- specific reason flags (deduplicated)
        flag_labels = {
            "WATCHLIST_TIER":               "On Watchlist tier",
            "INCOMING_FAIL_RATE_2X_BEST":   "Incoming fail rate exceeds 2x best-in-class supplier",
            "WARRANTY_CLAIM_RATE_ABOVE_4PCT": "Warranty claim rate above 4% ceiling",
        }
        flag_labels_to_emit = [
            flag_labels[f] for f in supplier_anomaly.reason_flags if f in flag_labels
        ]
        if flag_labels_to_emit and len(why) < 5:
            why.append("Triggered: " + "; ".join(flag_labels_to_emit))

        evidence = {
            "supplier":                  name,
            "coo":                       supplier_anomaly.coo,
            "tier":                      supplier_anomaly.tier,
            "quality_score":             supplier_anomaly.quality_score,
            "incoming_fail_rate":        supplier_anomaly.incoming_fail_rate,
            "warranty_claim_rate":       supplier_anomaly.warranty_claim_rate,
            "process_drift_index":       supplier_anomaly.process_drift_index,
            "process_cpk":               supplier_anomaly.process_cpk,
            "coo_incoming_fail_rate":    supplier_anomaly.coo_incoming_fail_rate,
            "coo_warranty_claim_rate":   supplier_anomaly.coo_warranty_claim_rate,
            "beats_coo_avg":             supplier_anomaly.beats_coo_avg,
            "flags":                     supplier_anomaly.reason_flags,
            "engineering_maturity":      supplier_anomaly.engineering_maturity,
        }

        # Cause hinges on the COO decomposition.
        if supplier_anomaly.beats_coo_avg == "Yes":
            likely_cause = (
                f"Supplier-specific quality issue -- {name} is the weak link "
                f"even versus {coo} peers."
            )
        elif supplier_anomaly.coo_incoming_fail_rate is not None:
            likely_cause = (
                f"Mixed country + supplier effect -- {coo} average also elevated; "
                f"supplier amplifies it."
            )
        else:
            likely_cause = "Supplier capability gap -- insufficient COO data to decompose."

        # Confidence: severity-weighted.
        qs_gap = 1.0 - (supplier_anomaly.quality_score or 100) / 100.0
        confidence = round(min(0.99, 0.55 + qs_gap * 0.4), 3)

        risk_level = (
            "HIGH" if (supplier_anomaly.quality_score or 100) < 50
            else "MEDIUM"
        )

        return InsightExplanation(
            insight_type="SUPPLIER_RISK",
            entity_id=str(supplier_anomaly.supplier_id),
            entity_name=name,
            headline=f"{name} quality profile requires attention",
            why=why,
            evidence=evidence,
            likely_cause=likely_cause,
            recommended_actions=self._actions_from_playbook("Warranty claim spike")
                or [
                    "Open root-cause investigation",
                    "Containment on active lots",
                    "Schedule design / capability review",
                ],
            confidence=confidence,
            risk_level=risk_level,
            drill_down_hints=[
                f"Pull all lots from {name} with fail rate > 5%",
                f"Compare {name} against peers in {coo}",
                "Review CoA / certification trend over last 90 days",
                "Audit supplier QA system and inspection coverage",
            ],
            sap_touchpoints=self._touchpoints_from_playbook(
                "Warranty claim spike", "SUPPLIER_RISK"
            ),
        )

    # ------------------------------------------------------------------
    # COO trend
    # ------------------------------------------------------------------

    def explain_coo_trend(
        self,
        country: str,
        kpi_engine: KPIEngine,
    ) -> InsightExplanation:
        coo_df = kpi_engine.get_coo_performance()
        country_row = coo_df[coo_df["coo"] == country]
        if country_row.empty:
            raise ValueError(f"No COO trend data for country: {country}")
        country_row = country_row.iloc[0]

        # Supplier-level decomposition within this COO.
        decomp_sql = """
            SELECT
                s.supplier_name,
                cvs.incoming_fail_rate,
                cvs.warranty_claim_rate,
                cvs.quality_score,
                cvs.tier,
                cvs.beats_coo_avg
            FROM agg_coo_vs_supplier cvs
            JOIN dim_supplier s ON cvs.supplier_id = s.supplier_id
            WHERE cvs.coo = :coo
        """
        suppliers_in_coo = pd.read_sql(
            decomp_sql, self.engine, params={"coo": country}
        )

        # Macro vs specific-driver decomposition.
        beats = suppliers_in_coo[suppliers_in_coo["beats_coo_avg"] == "Yes"]
        underperformers = suppliers_in_coo[suppliers_in_coo["beats_coo_avg"] == "No"]
        n_suppliers = len(suppliers_in_coo)

        macro_vs_specific = self._classify_coo_driver(
            suppliers_in_coo, float(country_row["coo_incoming_fail_rate"])
        )

        coo_fail_rate = float(country_row["coo_incoming_fail_rate"])
        claim_rate = float(country_row["coo_warranty_claim_rate"])
        rank = int(country_row["rank"])
        total_countries = len(coo_df)

        why: List[str] = []

        why.append(
            f"{country} incoming fail rate {coo_fail_rate:.2%} -- rank "
            f"{rank} of {total_countries} countries"
        )
        why.append(
            f"Warranty claim rate {claim_rate:.2%} across units from "
            f"{country}-sourced components"
        )

        # The NUANCE bullet -- this is the whole point of COO explanation.
        if n_suppliers == 1:
            sole = str(suppliers_in_coo.iloc[0]["supplier_name"])
            why.append(
                f"Only one supplier ({sole}) from {country} in this dataset -- "
                f"the country signal IS this supplier's signal; do not generalise "
                f"the trend to {country} as a whole"
            )
        elif macro_vs_specific == "MACRO":
            why.append(
                f"Trend is country-level -- all {n_suppliers} suppliers in "
                f"{country} perform within a narrow band around the COO average"
            )
        elif macro_vs_specific == "DRIVEN":
            top_driver = (
                underperformers.sort_values("incoming_fail_rate", ascending=False)
                .iloc[0]["supplier_name"]
                if not underperformers.empty else "n/a"
            )
            beaters_str = ", ".join(beats["supplier_name"].astype(str).tolist()) or "none"
            why.append(
                f"Trend is driven by specific suppliers -- {top_driver} sits above "
                f"the country average while others ({beaters_str}) beat it"
            )
        else:
            why.append(
                "Insufficient supplier-level data to decompose -- treat as "
                "directional signal only"
            )

        # Named outperformers (opportunity framing).
        if not beats.empty:
            why.append(
                f"Opportunity: {len(beats)} supplier(s) from {country} beat "
                f"their COO average -- prefer them for critical builds"
            )

        evidence = {
            "country":               country,
            "coo_incoming_fail_rate": round(coo_fail_rate, 4),
            "coo_warranty_claim_rate": round(claim_rate, 4),
            "rank":                  rank,
            "n_countries":           total_countries,
            "n_suppliers_in_coo":    n_suppliers,
            "n_beating_coo_avg":     int(len(beats)),
            "beaters":               beats["supplier_name"].astype(str).tolist(),
            "underperformers":       underperformers["supplier_name"].astype(str).tolist(),
            "driver_classification": macro_vs_specific,
        }

        if n_suppliers == 1:
            sole = str(suppliers_in_coo.iloc[0]["supplier_name"])
            likely_cause = (
                f"Single-supplier country -- the COO signal for {country} equals "
                f"{sole}'s signal. Treat this as a supplier issue, not a "
                f"{country} issue."
            )
        elif macro_vs_specific == "DRIVEN":
            likely_cause = (
                f"Not a country problem -- {country}'s trend is driven by "
                f"specific supplier performance, not a universal effect."
            )
        elif macro_vs_specific == "MACRO":
            likely_cause = (
                f"Country-level structural factor -- {country} suppliers cluster "
                f"tightly, pointing to regional inputs, logistics, or standards."
            )
        else:
            likely_cause = "Indeterminate -- more supplier data needed."

        # COO signals are backed by whole-COO aggregates -- high base confidence.
        confidence = round(min(0.99, 0.70 + (0.05 * n_suppliers)), 3)

        if rank <= total_countries * 0.4:
            risk_level = "LOW"
        elif rank <= total_countries * 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        if n_suppliers == 1:
            context_str = (
                f"single supplier ({suppliers_in_coo.iloc[0]['supplier_name']}), "
                f"trend equals that supplier's"
            )
        elif macro_vs_specific == "DRIVEN":
            context_str = f"{n_suppliers} suppliers, supplier-driven pattern"
        elif macro_vs_specific == "MACRO":
            context_str = f"{n_suppliers} suppliers, country-level pattern"
        else:
            context_str = f"{n_suppliers} suppliers, insufficient data to decompose"

        return InsightExplanation(
            insight_type="COO_TREND",
            entity_id=country,
            entity_name=country,
            headline=f"COO macro trend for {country} -- {context_str}",
            why=why,
            evidence=evidence,
            likely_cause=likely_cause,
            recommended_actions=self._actions_from_playbook("Premium")
                + [
                    "Do not stereotype by COO -- use supplier scorecard for sourcing decisions",
                    "Prefer proven suppliers within the country for critical builds",
                    "Tighten controls for specific underperformers, not the whole country",
                ],
            confidence=confidence,
            risk_level=risk_level,
            drill_down_hints=[
                f"Review all {n_suppliers} supplier(s) from {country} side-by-side",
                "Inspect trade / logistics incident logs for the window",
                "Check component-mix skew by COO (are risky components concentrated?)",
                f"Segment warranty claims by sub-region within {country}",
            ],
            sap_touchpoints=self._touchpoints_from_playbook(
                "Premium", "COO_TREND"
            ),
        )

    @staticmethod
    def _classify_coo_driver(
        suppliers_df: pd.DataFrame, coo_rate: float
    ) -> str:
        """Decide whether a COO trend is macro or driven by specific suppliers.

        * **MACRO** -- all suppliers sit within +/-25% of the COO average.
        * **DRIVEN** -- spread exceeds that band (at least one clear outlier).
        * **UNKNOWN** -- not enough suppliers to say.
        """
        if len(suppliers_df) < 2 or coo_rate <= 0:
            return "UNKNOWN"
        spread = suppliers_df["incoming_fail_rate"].dropna()
        if spread.empty:
            return "UNKNOWN"
        ratios = (spread / coo_rate).abs()
        max_dev = float((ratios - 1).abs().max())
        return "MACRO" if max_dev < 0.25 else "DRIVEN"


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def generate_all_insights(engine: Engine) -> List[InsightExplanation]:
    """Detect every anomaly, generate explanations, and persist as JSON.

    Writes ``data/processed/insights.json`` and returns the list of
    :class:`InsightExplanation` objects in the same order they were written.
    """
    kpi = KPIEngine(engine)
    detector = AnomalyDetector(engine)
    explainer = InsightExplainer(engine)

    lot_anoms = detector.detect_lot_anomalies(kpi)
    proc_anoms = detector.detect_process_anomalies(kpi)
    sup_anoms = detector.detect_supplier_anomalies(kpi)

    insights: List[InsightExplanation] = []

    for la in lot_anoms:
        insights.append(explainer.explain_lot_risk(la, kpi))

    for pa in proc_anoms:
        insights.append(explainer.explain_process_drift(pa, kpi))

    for sa in sup_anoms:
        insights.append(explainer.explain_supplier_risk(sa, kpi))

    # One COO explanation per country -- we always want the macro narrative.
    for country in kpi.get_coo_performance()["coo"].tolist():
        insights.append(explainer.explain_coo_trend(country, kpi))

    # Persist.
    out_path = settings.PROCESSED_DIR / "insights.json"
    by_type: Dict[str, int] = {}
    for ins in insights:
        by_type[ins.insight_type] = by_type.get(ins.insight_type, 0) + 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count":        len(insights),
        "by_type":      by_type,
        "insights":     [ins.to_dict() for ins in insights],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Wrote %d insights to %s  (by type: %s)",
        len(insights), out_path, by_type,
    )

    return insights


__all__ = ["InsightExplanation", "InsightExplainer", "generate_all_insights"]
