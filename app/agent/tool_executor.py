"""
Tool executor for the Quality Agent (Phase 3, Step 2).

Maps every Claude tool call onto a service-registry method, returns a
uniform :class:`ToolResult` envelope, and swallows all exceptions so the
agent loop never sees a raw traceback.

Key contracts
-------------
* :meth:`ToolExecutor.execute` never raises. On any failure it returns a
  :class:`ToolResult` whose ``error`` field is a human-readable string.
* Every call is logged with tool name, input, row count, and elapsed ms.
* Return payloads are JSON-serialisable (no NaN / Timestamp / numpy).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from sqlalchemy.engine import Engine

from app.agent.tools import TOOL_NAMES
from app.services.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    tool_name: str
    input_used: Dict[str, Any]
    result_data: Dict[str, Any] = field(default_factory=dict)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# DataFrame -> JSON helper
# ---------------------------------------------------------------------------

def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """NaN / Timestamp-safe ``df.to_dict('records')``."""
    if df is None or df.empty:
        return []
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    clean = clean.astype(object).where(clean.notna(), None)
    return clean.to_dict("records")


def _count_rows(data: Any) -> int:
    """Best-effort row count for a response payload."""
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("rows", "records", "items", "inspection_records"):
            if isinstance(data.get(key), list):
                return len(data[key])
        return 1
    return 0


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Dispatches Claude tool calls to service-registry methods."""

    def __init__(self, registry: ServiceRegistry, engine: Engine) -> None:
        self.registry = registry
        self.engine = engine

        self._dispatch: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_lot_risk":            self.execute_get_lot_risk,
            "get_supplier_profile":    self.execute_get_supplier_profile,
            "get_process_drift":       self.execute_get_process_drift,
            "get_coo_trend":           self.execute_get_coo_trend,
            "get_drill_down":          self.execute_get_drill_down,
            "get_inspection_strategy": self.execute_get_inspection_strategy,
            "get_action_playbook":     self.execute_get_action_playbook,
            "search_insights":         self.execute_search_insights,
            "compare_suppliers":       self.execute_compare_suppliers,
            "get_warranty_trace":      self.execute_get_warranty_trace,
            "get_material_vendors":    self.execute_get_material_vendors,
        }
        # Surface accidental drift between tools.py and the executor ASAP.
        missing = set(TOOL_NAMES) - set(self._dispatch)
        assert not missing, f"ToolExecutor missing handlers for: {missing}"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> ToolResult:
        """Run *tool_name* with *tool_input*; always return a ToolResult."""
        start = time.perf_counter()
        logger.info(
            "tool_call  start  name=%s  input=%s", tool_name, tool_input
        )

        handler = self._dispatch.get(tool_name)
        if handler is None:
            elapsed = (time.perf_counter() - start) * 1000
            msg = f"Unknown tool: {tool_name!r}"
            logger.warning("tool_call  error  %s  (%.1fms)", msg, elapsed)
            return ToolResult(
                tool_name=tool_name,
                input_used=dict(tool_input or {}),
                error=msg,
                execution_time_ms=round(elapsed, 2),
            )

        try:
            data = handler(**(tool_input or {}))
            elapsed = (time.perf_counter() - start) * 1000
            row_count = _count_rows(data)
            logger.info(
                "tool_call  ok     name=%s  rows=%d  (%.1fms)",
                tool_name, row_count, elapsed,
            )
            return ToolResult(
                tool_name=tool_name,
                input_used=dict(tool_input or {}),
                result_data=data if isinstance(data, dict) else {"result": data},
                row_count=row_count,
                execution_time_ms=round(elapsed, 2),
            )
        except TypeError as exc:
            # Most commonly raised on bad / missing kwargs from Claude.
            elapsed = (time.perf_counter() - start) * 1000
            msg = f"Bad input for {tool_name}: {exc}"
            logger.warning("tool_call  error  %s  (%.1fms)", msg, elapsed)
            return ToolResult(
                tool_name=tool_name,
                input_used=dict(tool_input or {}),
                error=msg,
                execution_time_ms=round(elapsed, 2),
            )
        except Exception as exc:  # noqa: BLE001 -- intentional blanket catch
            elapsed = (time.perf_counter() - start) * 1000
            msg = f"{exc.__class__.__name__}: {exc}"
            logger.exception(
                "tool_call  error  name=%s  (%.1fms)", tool_name, elapsed
            )
            return ToolResult(
                tool_name=tool_name,
                input_used=dict(tool_input or {}),
                error=msg,
                execution_time_ms=round(elapsed, 2),
            )

    # ==================================================================
    # TOOL 1 — get_lot_risk
    # ==================================================================

    def execute_get_lot_risk(
        self,
        lot_no: str,
        include_warranty: bool = True,
    ) -> Dict[str, Any]:
        risk_df = self.registry.kpi.get_lot_risk_scores()
        row = risk_df[risk_df["lot_no"] == lot_no]
        if row.empty:
            return {"found": False, "message": f"Lot {lot_no!r} not found in lot_risk_scores"}
        r = row.iloc[0]

        insp_df = self.registry.drill_down.lot_to_inspection_records(lot_no)

        result: Dict[str, Any] = {
            "found":              True,
            "lot_no":             lot_no,
            "risk_score":         round(float(r["lot_risk_score"]), 4),
            "risk_tier":          str(r["risk_tier"]),
            "fail_rate":          round(float(r["fail_rate"]), 4),
            "total_inspections":  int(r["total_inspections"]),
            "total_fails":        int(r["total_fails"]),
            "defect_codes":       str(r["defect_codes"]) if r["defect_codes"] else "",
            "inspection_records": _df_to_records(insp_df),
        }

        if include_warranty:
            serials_df = self.registry.drill_down.lot_to_affected_serials(lot_no)
            warranty_claims: List[Dict[str, Any]] = []
            for sn in serials_df["serial_no"].astype(str).tolist():
                claim = self.registry.drill_down.serial_to_warranty_outcome(sn)
                if claim is not None:
                    warranty_claims.append(claim)
            result["affected_serial_count"] = int(len(serials_df))
            result["warranty_count"]        = len(warranty_claims)
            result["warranty_claims"]       = warranty_claims

        return result

    # ==================================================================
    # TOOL 2 — get_supplier_profile
    # ==================================================================

    def execute_get_supplier_profile(
        self,
        supplier_id: str,
        include_coo_decomposition: bool = True,
    ) -> Dict[str, Any]:
        rankings = self.registry.kpi.get_supplier_rankings()

        resolved_id = self._resolve_supplier(supplier_id)
        if resolved_id is None:
            return {"found": False, "message": f"Supplier {supplier_id!r} not found"}

        row = rankings[rankings["supplier_id"] == resolved_id]
        if row.empty:
            return {"found": False, "message": f"Supplier {supplier_id!r} has no scorecard row"}
        r = row.iloc[0]

        profile: Dict[str, Any] = {
            "found":                         True,
            "supplier_id":                   int(resolved_id),
            "supplier":                      str(r["supplier"]),
            "coo":                           r["coo"] if pd.notna(r["coo"]) else None,
            "tier":                          r["tier"] if pd.notna(r["tier"]) else None,
            "quality_score":                 _as_float(r["quality_score"]),
            "incoming_fail_rate":            _as_float(r["incoming_fail_rate"]),
            "warranty_claim_rate":           _as_float(r["warranty_claim_rate"]),
            "process_drift_index":           _as_float(r["process_drift_index"]),
            "on_time_delivery_pct":          _as_float(r["on_time_delivery_pct"]),
            "avg_lead_time_days":            _as_float(r["avg_lead_time_days"]),
            "engineering_maturity":          r["engineering_maturity"] if pd.notna(r["engineering_maturity"]) else None,
            "engineering_maturity_score":    _as_float(r["engineering_maturity_score"]),
            "process_cpk":                   _as_float(r["process_cpk"]),
            "premium_service_fit":           r["premium_service_fit"] if pd.notna(r["premium_service_fit"]) else None,
            "precision_score":               _as_float(r["precision_score"]),
            "composite_score":               _as_float(r["composite_score"]),
            "composite_rank":                int(r["composite_rank"]) if pd.notna(r["composite_rank"]) else None,
        }

        if include_coo_decomposition:
            ctx = self.registry.drill_down.supplier_to_coo_context(resolved_id)
            profile["coo_decomposition"] = ctx

        return profile

    # ==================================================================
    # TOOL 3 — get_process_drift
    # ==================================================================

    def execute_get_process_drift(
        self,
        line: Optional[str] = None,
        shift: Optional[str] = None,
        only_flagged: bool = False,
    ) -> Dict[str, Any]:
        from configs import settings as _settings
        threshold = _settings.PROCESS_DRIFT_FAIL_RATE_THRESHOLD
        df = self.registry.kpi.get_process_drift_by_line_shift()

        df = df.copy()
        df["is_drift_signal"] = df["torque_fail_rate"] > threshold

        if line:
            df = df[df["line"].str.upper() == line.upper()]
        if shift:
            df = df[df["shift"].str.lower() == shift.lower()]
        if only_flagged:
            df = df[df["is_drift_signal"]]

        # Optional: enrich each row with affected serial count.
        if not df.empty:
            serial_counts = pd.read_sql(
                "SELECT line, shift, COUNT(DISTINCT serial_id) AS affected_serials "
                "FROM fact_process_measurements GROUP BY line, shift",
                self.engine,
            )
            df = df.merge(serial_counts, on=["line", "shift"], how="left")

        return {
            "threshold":       threshold,
            "filters":         {"line": line, "shift": shift, "only_flagged": only_flagged},
            "drift_signals":   _df_to_records(df),
            "records":         _df_to_records(df),  # alias for row_count detection
        }

    # ==================================================================
    # TOOL 4 — get_coo_trend
    # ==================================================================

    def execute_get_coo_trend(self, coo: Optional[str] = None) -> Dict[str, Any]:
        df = self.registry.kpi.get_coo_performance()
        if coo:
            df = df[df["coo"].str.lower() == coo.lower()]

        # Supplier exceptions for each country in scope.
        cvs = self.registry.kpi.get_coo_vs_supplier_decomposition()
        if coo:
            cvs = cvs[cvs["coo"].str.lower() == coo.lower()]

        return {
            "filter":                 {"coo": coo},
            "trends":                 _df_to_records(df),
            "supplier_decomposition": _df_to_records(cvs),
            "records":                _df_to_records(df),
        }

    # ==================================================================
    # TOOL 5 — get_drill_down
    # ==================================================================

    def execute_get_drill_down(
        self,
        lot_no: Optional[str] = None,
        serial_no: Optional[str] = None,
        depth: str = "full",
    ) -> Dict[str, Any]:
        if not lot_no and not serial_no:
            return {"error": "Provide either lot_no or serial_no."}

        if lot_no:
            if depth == "lot_only":
                insp   = self.registry.drill_down.lot_to_inspection_records(lot_no)
                seri   = self.registry.drill_down.lot_to_affected_serials(lot_no)
                scard  = self.registry.drill_down.lot_to_supplier_scorecard(lot_no)
                return {
                    "mode":               "lot_only",
                    "lot_no":             lot_no,
                    "inspection_records": _df_to_records(insp),
                    "affected_serials":   _df_to_records(seri),
                    "supplier_scorecard": scard,
                }
            # default: full chain
            return self.registry.drill_down.get_full_drill_down_chain(lot_no)

        # serial-only path
        pm = self.registry.drill_down.serial_to_process_measurements(serial_no)
        claim = self.registry.drill_down.serial_to_warranty_outcome(serial_no)
        bom_sql = """
            SELECT
                ds.serial_no,
                dc.component_name AS component,
                l.lot_no,
                dsup.supplier_name AS supplier,
                b.comp_serial,
                b.coo,
                b.mfg_date
            FROM fact_constituent_bom b
            JOIN dim_serial   ds   ON b.serial_id   = ds.serial_id
            LEFT JOIN dim_component dc   ON b.component_id = dc.component_id
            LEFT JOIN dim_lot       l    ON b.lot_id       = l.lot_id
            LEFT JOIN dim_supplier  dsup ON b.supplier_id  = dsup.supplier_id
            WHERE ds.serial_no = :sn
        """
        bom = pd.read_sql(bom_sql, self.engine, params={"sn": serial_no})
        return {
            "mode":                "serial_only",
            "serial_no":           serial_no,
            "process_measurements": _df_to_records(pm),
            "warranty_outcome":    claim,
            "bom_back_refs":       _df_to_records(bom),
        }

    # ==================================================================
    # TOOL 6 — get_inspection_strategy
    # ==================================================================

    def execute_get_inspection_strategy(
        self,
        risk_threshold: str = "HIGH",
    ) -> Dict[str, Any]:
        plan = self.registry.recommendations.get_inspection_strategy()
        keep = {
            "HIGH":   {"HIGH"},
            "MEDIUM": {"HIGH", "MEDIUM"},
            "ALL":    {"HIGH", "MEDIUM", "LOW"},
        }.get(risk_threshold.upper(), {"HIGH"})

        plan["increase_sampling"] = [
            item for item in plan["increase_sampling"]
            if item.get("risk_tier") in keep
        ]
        plan["filter"] = {"risk_threshold": risk_threshold.upper()}
        plan["records"] = plan["increase_sampling"]
        return plan

    # ==================================================================
    # TOOL 7 — get_action_playbook
    # ==================================================================

    _TIER_ACTIONS = {
        "watchlist": [
            "Consider alternative supplier for critical builds",
            "Lock sourcing to current approved lots pending capability review",
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

    def execute_get_action_playbook(
        self,
        insight_type: str,
        supplier_tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        df = pd.read_sql("SELECT * FROM ref_action_playbook", self.engine)
        hits = df[df["insight_type"].str.contains(insight_type, case=False, na=False)]
        if hits.empty:
            return {
                "found":         False,
                "insight_type":  insight_type,
                "message":       f"No playbook row matches {insight_type!r}",
                "actions":       [],
                "sap_touchpoints": [],
            }
        row = hits.iloc[0]
        actions = [p.strip() for p in str(row["typical_action"]).split(";") if p.strip()]
        sap     = [p.strip() for p in str(row["sap_mes_touchpoint"]).split("/") if p.strip()]

        if supplier_tier:
            actions = actions + self._TIER_ACTIONS.get(supplier_tier.strip().lower(), [])

        return {
            "found":           True,
            "insight_type":    str(row["insight_type"]),
            "actions":         actions,
            "where_it_fits":   str(row["where_it_fits"]),
            "sap_touchpoints": sap,
            "supplier_tier":   supplier_tier,
        }

    # ==================================================================
    # TOOL 8 — search_insights
    # ==================================================================

    def execute_search_insights(
        self,
        query: str,
        risk_type: str = "Both",
    ) -> Dict[str, Any]:
        df = pd.read_sql(
            "SELECT * FROM ref_ai_insights WHERE pattern_detected IS NOT NULL",
            self.engine,
        )
        if df.empty:
            return {"query": query, "matches": [], "records": []}

        q = (query or "").lower().strip()
        if q:
            mask = (
                df["pattern_detected"].fillna("").str.lower().str.contains(q, regex=False)
                | df["evidence"].fillna("").str.lower().str.contains(q, regex=False)
                | df["ai_guidance"].fillna("").str.lower().str.contains(q, regex=False)
            )
            df = df[mask]

        if risk_type and risk_type != "Both":
            df = df[df["risk_or_opportunity"].fillna("").str.lower() == risk_type.lower()]

        records = _df_to_records(df)
        return {"query": query, "risk_type": risk_type, "matches": records, "records": records}

    # ==================================================================
    # TOOL 9 — compare_suppliers
    # ==================================================================

    def execute_compare_suppliers(
        self,
        supplier_ids: List[str],
        use_case: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(supplier_ids, list) or len(supplier_ids) < 2:
            return {"error": "supplier_ids must be a list of at least two suppliers."}

        profiles: List[Dict[str, Any]] = []
        for sid in supplier_ids:
            prof = self.execute_get_supplier_profile(sid, include_coo_decomposition=True)
            if prof.get("found"):
                profiles.append(prof)

        if len(profiles) < 2:
            return {
                "error": (
                    "Could not resolve at least two valid suppliers. "
                    f"Given: {supplier_ids}"
                ),
                "resolved": profiles,
            }

        # Dimensions laid out side-by-side for easy UI rendering.
        dims = [
            "quality_score", "precision_score", "composite_score",
            "incoming_fail_rate", "warranty_claim_rate", "process_drift_index",
            "process_cpk", "engineering_maturity_score",
            "on_time_delivery_pct", "avg_lead_time_days",
            "tier", "coo", "premium_service_fit",
        ]
        side_by_side = {
            "supplier":       [p["supplier"] for p in profiles],
            **{dim: [p.get(dim) for p in profiles] for dim in dims},
        }

        # Winner selection.
        safety_critical = bool(use_case) and any(
            token in use_case.lower() for token in ("safety", "critical")
        )
        if safety_critical:
            ranking_key = lambda p: (
                _as_float(p.get("engineering_maturity_score")) or 0.0,
                _as_float(p.get("process_cpk")) or 0.0,
                _as_float(p.get("quality_score")) or 0.0,
            )
            highlight = ["engineering_maturity_score", "process_cpk"]
            rationale = (
                "Safety-critical use case: ranked by engineering maturity "
                "and Cpk first; quality score used as tie-breaker."
            )
        else:
            ranking_key = lambda p: (
                _as_float(p.get("composite_score")) or 0.0,
                _as_float(p.get("quality_score")) or 0.0,
            )
            highlight = ["composite_score", "quality_score"]
            rationale = (
                "General use case: ranked by composite score (quality + "
                "precision) with quality_score as tie-breaker."
            )

        ordered = sorted(profiles, key=ranking_key, reverse=True)
        winner = ordered[0]

        return {
            "use_case":         use_case,
            "safety_critical":  safety_critical,
            "side_by_side":     side_by_side,
            "profiles":         profiles,
            "highlight_fields": highlight,
            "recommendation": {
                "winner":     winner["supplier"],
                "tier":       winner.get("tier"),
                "rationale":  rationale,
                "ranked":     [p["supplier"] for p in ordered],
            },
        }

    # ==================================================================
    # TOOL 10 — get_warranty_trace
    # ==================================================================

    def execute_get_warranty_trace(
        self,
        serial_no: Optional[str] = None,
        claim_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not any([serial_no, claim_id, supplier_id]):
            return {"error": "Provide at least one of serial_no, claim_id, or supplier_id."}

        # Resolve claim -> serial if the caller only has the claim.
        if claim_id and not serial_no:
            claim_df = pd.read_sql(
                """
                SELECT ds.serial_no, w.claim_id, w.failure_date, w.symptom,
                       w.severity, w.region, w.mileage_or_hours
                FROM fact_warranty_claims w
                JOIN dim_serial ds ON w.serial_id = ds.serial_id
                WHERE w.claim_id = :cid
                LIMIT 1
                """,
                self.engine,
                params={"cid": claim_id},
            )
            if claim_df.empty:
                return {"error": f"Claim {claim_id!r} not found"}
            serial_no = str(claim_df.iloc[0]["serial_no"])

        # Supplier-scoped: every warranty claim that traces back to a lot
        # supplied by this vendor.
        if supplier_id and not serial_no:
            resolved = self._resolve_supplier(supplier_id)
            if resolved is None:
                return {"error": f"Supplier {supplier_id!r} not found"}
            claims = pd.read_sql(
                """
                SELECT DISTINCT
                    w.claim_id, w.failure_date, w.symptom, w.severity,
                    ds.serial_no,
                    l.lot_no,
                    dc.component_name AS component,
                    s.supplier_name   AS supplier
                FROM fact_warranty_claims w
                JOIN dim_serial ds            ON w.serial_id   = ds.serial_id
                JOIN fact_constituent_bom b   ON b.serial_id   = ds.serial_id
                LEFT JOIN dim_lot       l     ON b.lot_id      = l.lot_id
                LEFT JOIN dim_component dc    ON b.component_id = dc.component_id
                LEFT JOIN dim_supplier  s     ON b.supplier_id  = s.supplier_id
                WHERE b.supplier_id = :sid
                ORDER BY w.failure_date
                """,
                self.engine,
                params={"sid": int(resolved)},
            )
            return {
                "mode":           "supplier_scoped",
                "supplier_id":    int(resolved),
                "claims_traced":  _df_to_records(claims),
                "records":        _df_to_records(claims),
            }

        # Full backward trace from a single serial.
        assert serial_no is not None
        claim = self.registry.drill_down.serial_to_warranty_outcome(serial_no)

        bom = pd.read_sql(
            """
            SELECT
                ds.serial_no,
                dc.component_name AS component,
                l.lot_no,
                l.lot_id,
                dsup.supplier_name AS supplier,
                dsup.supplier_id,
                dsup.coo AS supplier_coo,
                b.comp_serial,
                b.coo,
                b.mfg_date
            FROM fact_constituent_bom b
            JOIN dim_serial   ds   ON b.serial_id   = ds.serial_id
            LEFT JOIN dim_component dc   ON b.component_id = dc.component_id
            LEFT JOIN dim_lot       l    ON b.lot_id       = l.lot_id
            LEFT JOIN dim_supplier  dsup ON b.supplier_id  = dsup.supplier_id
            WHERE ds.serial_no = :sn
            """,
            self.engine,
            params={"sn": serial_no},
        )

        # Incoming inspections across every lot in the BOM.
        lot_ids = [int(lid) for lid in bom["lot_id"].dropna().unique().tolist()]
        if lot_ids:
            insp = pd.read_sql(
                f"""
                SELECT
                    l.lot_no, qm.insp_date, qm.characteristic,
                    qm.result, qm.defect_code, qm.measured_value
                FROM fact_incoming_qm qm
                JOIN dim_lot l ON qm.lot_id = l.lot_id
                WHERE qm.lot_id IN ({','.join(['?'] * len(lot_ids))})
                ORDER BY qm.insp_date
                """,
                self.engine,
                params=tuple(lot_ids),
            )
        else:
            insp = pd.DataFrame()

        return {
            "mode":              "serial_scoped",
            "serial_no":         serial_no,
            "warranty_claim":    claim,
            "bom":               _df_to_records(bom),
            "lot_inspections":   _df_to_records(insp),
            "records":           _df_to_records(bom),
        }

    # ==================================================================
    # TOOL 11 — get_material_vendors
    # ==================================================================

    def execute_get_material_vendors(
        self,
        material_name: str,
        metric: str = "precision",
    ) -> Dict[str, Any]:
        """Return vendors that actually supply *material_name* with their metrics."""
        from sqlalchemy import text as _text

        # Find supplier IDs that have supplied this component.
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    _text(
                        "SELECT DISTINCT l.supplier_id, c.component_name "
                        "FROM dim_lot l "
                        "JOIN dim_component c ON l.component_id = c.component_id "
                        "WHERE LOWER(c.component_name) LIKE :pat"
                    ),
                    {"pat": f"%{material_name.lower()}%"},
                ).fetchall()
        except Exception as exc:
            return {"found": False, "message": str(exc), "vendors": []}

        if not rows:
            # Fall back: list all known component names to help Claude advise the user.
            try:
                with self.engine.connect() as conn:
                    comps = conn.execute(
                        _text("SELECT DISTINCT component_name FROM dim_component ORDER BY component_name")
                    ).fetchall()
                known = [r[0] for r in comps]
            except Exception:
                known = []
            return {
                "found":            False,
                "material_queried": material_name,
                "message":          f"No lots found for component matching '{material_name}'.",
                "known_components": known,
                "vendors":          [],
            }

        supplier_ids = {r[0] for r in rows}
        component_name = rows[0][1]  # take the first matched component name

        rankings = self.registry.kpi.get_supplier_rankings()
        df = rankings[rankings["supplier_id"].isin(supplier_ids)].copy()

        # Sort by chosen metric.
        _col_map = {
            "precision": ("precision_score",  False),
            "quality":   ("quality_score",     False),
            "fail_rate": ("incoming_fail_rate", True),
            "cpk":       ("process_cpk",        False),
            "warranty":  ("warranty_claim_rate",True),
        }
        sort_col, ascending = _col_map.get(metric, ("precision_score", False))
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=ascending)

        vendors = []
        for _, r in df.iterrows():
            vendors.append({
                "supplier":           str(r["supplier"]),
                "tier":               r["tier"] if pd.notna(r.get("tier")) else None,
                "coo":                r["coo"]  if pd.notna(r.get("coo"))  else None,
                "precision_score":    _as_float(r.get("precision_score")),
                "quality_score":      _as_float(r.get("quality_score")),
                "incoming_fail_rate": _as_float(r.get("incoming_fail_rate")),
                "warranty_claim_rate":_as_float(r.get("warranty_claim_rate")),
                "process_cpk":        _as_float(r.get("process_cpk")),
                "process_drift_index":_as_float(r.get("process_drift_index")),
            })

        winner = vendors[0]["supplier"] if vendors else None
        return {
            "found":            True,
            "material_queried": material_name,
            "component_name":   component_name,
            "metric_ranked_by": metric,
            "vendor_count":     len(vendors),
            "winner":           winner,
            "vendors":          vendors,
            "records":          vendors,
        }

    # ==================================================================
    # Helpers
    # ==================================================================

    def _resolve_supplier(self, supplier_id: str) -> Optional[int]:
        """Accept 'SUP-C' OR an integer id (as int or string) and return the
        integer supplier_id that matches, or ``None`` if no match."""
        raw = str(supplier_id).strip()
        if not raw:
            return None
        # Integer id?
        if raw.isdigit():
            df = pd.read_sql(
                "SELECT supplier_id FROM dim_supplier WHERE supplier_id = :i",
                self.engine, params={"i": int(raw)},
            )
            return int(df.iloc[0]["supplier_id"]) if not df.empty else None
        # Name match (case-insensitive).
        df = pd.read_sql(
            "SELECT supplier_id, supplier_name FROM dim_supplier "
            "WHERE UPPER(supplier_name) = :n",
            self.engine, params={"n": raw.upper()},
        )
        return int(df.iloc[0]["supplier_id"]) if not df.empty else None


# ---------------------------------------------------------------------------
# Small scalar helpers
# ---------------------------------------------------------------------------

def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


__all__ = ["ToolExecutor", "ToolResult"]
