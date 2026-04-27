"""
Normalisation utilities and the full ETL pipeline for the Quality Agent.

Two public surfaces:
  normalise_columns()       – standalone column-rename helper (used by profiler)
  NormalizationPipeline     – orchestrates all dim / fact / agg / ref ETL
"""
import logging
import re
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def _snake(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\s\-/]+", "_", name)
    name = re.sub(r"[^\w]", "", name)
    return name.lower()


def normalise_columns(df: pd.DataFrame, sheet_name: str = "") -> pd.DataFrame:
    """Rename all columns to snake_case and strip string-cell whitespace."""
    renamed = {col: _snake(col) for col in df.columns}
    df = df.rename(columns=renamed)
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].apply(
        lambda s: s.str.strip() if hasattr(s, "str") else s
    )
    logger.debug("[%s] Columns normalised: %s", sheet_name, list(df.columns))
    return df


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class DataIntegrityError(Exception):
    """Raised when a mandatory post-load integrity check fails."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class NormalizationPipeline:
    """Transform raw Excel sheets into star-schema DataFrames and load them
    into the configured SQLite database.

    Methods should be called via :meth:`run_full_pipeline`.  Individual
    ``normalize_*`` methods are intentionally public so they can be unit-tested
    in isolation without a live database.
    """

    def __init__(self) -> None:
        # Surrogate-key lookup maps — populated by dim normalizers, consumed by
        # fact / agg normalizers.  All maps are {natural_key: integer_pk}.
        self._supplier_map: dict[str, int] = {}
        self._material_map: dict[str, int] = {}
        self._component_map: dict[str, int] = {}
        self._lot_map: dict[str, int] = {}
        self._serial_map: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raw(sheets: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
        """Return a defensive copy of a raw sheet."""
        return sheets[name].copy()

    @staticmethod
    def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from all object/string columns."""
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].str.strip()
        return df

    def _map_col(
        self,
        series: pd.Series,
        lookup: dict[str, int],
        col_label: str,
        table_label: str,
    ) -> pd.Series:
        """Apply a surrogate-key lookup, logging any unmatched values."""
        result = series.map(lookup)
        unmatched = series[result.isna() & series.notna()].unique()
        if len(unmatched):
            logger.warning(
                "[%s] %d unmatched '%s' value(s) -> NULL FK: %s",
                table_label,
                len(unmatched),
                col_label,
                list(unmatched[:5]),
            )
        return result

    @staticmethod
    def _write_table(
        df: pd.DataFrame,
        table_name: str,
        pk_col: str,
        engine: Engine,
    ) -> int:
        """Assign a 1-based integer PK index, write to DB, return row count."""
        out = df.copy()
        out.index = range(1, len(out) + 1)
        out.index.name = pk_col
        out.to_sql(table_name, engine, if_exists="replace", index=True)
        logger.info("  %-35s  %d rows written", table_name, len(out))
        return len(out)

    # ------------------------------------------------------------------
    # Dimension normalizers
    # ------------------------------------------------------------------

    def normalize_suppliers(
        self, sheets: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Build dim_supplier from Supplier_Scorecard ⋈ Vendor_Engineering_Profile.

        The scorecard has financial/quality KPIs; the engineering profile adds
        process and design attributes.  Both tables share the 'Supplier' key.
        The join is left on scorecard so every scored supplier is present even
        if the engineering profile row is missing.
        """
        sc = self._raw(sheets, "Supplier_Scorecard")[
            ["Supplier", "COO", "Engineering_Maturity", "Engineering_Maturity_Score"]
        ]
        vep = self._raw(sheets, "Vendor_Engineering_Profile")[
            ["Supplier", "Process_Cpk", "Design_Ownership", "Typical_Project_Type"]
        ]

        sc["Supplier"] = sc["Supplier"].str.strip().str.upper()
        vep["Supplier"] = vep["Supplier"].str.strip().str.upper()

        merged = sc.merge(vep, on="Supplier", how="left")

        df = pd.DataFrame({
            "supplier_name":            merged["Supplier"],
            "coo":                       merged["COO"].str.strip(),
            "engineering_maturity":      merged["Engineering_Maturity"].str.strip(),
            "engineering_maturity_score": merged["Engineering_Maturity_Score"].astype(float),
            "process_cpk":               merged["Process_Cpk"].astype(float),
            "design_ownership":          merged["Design_Ownership"].str.strip(),
            "typical_project_type":      merged["Typical_Project_Type"].str.strip(),
        }).drop_duplicates(subset=["supplier_name"]).reset_index(drop=True)

        # Populate surrogate-key map (1-based to match the index assigned in _write_table)
        self._supplier_map = {
            name: idx + 1 for idx, name in enumerate(df["supplier_name"])
        }
        logger.info(
            "[normalize_suppliers] %d suppliers -> map: %s",
            len(df),
            self._supplier_map,
        )
        return df

    def normalize_materials(
        self, sheets: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Unique finished-material names from AsBuilt_Serial."""
        abs_ = self._raw(sheets, "AsBuilt_Serial")
        names = (
            abs_["FinishedMaterial"]
            .str.strip()
            .dropna()
            .unique()
        )
        df = pd.DataFrame({"material_name": sorted(names)})
        self._material_map = {
            name: idx + 1 for idx, name in enumerate(df["material_name"])
        }
        logger.info(
            "[normalize_materials] %d materials -> %s",
            len(df),
            list(df["material_name"]),
        )
        return df

    def normalize_components(
        self, sheets: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Unique component names from Constituent_BOM ∪ Incoming_QM."""
        bom_comps = (
            self._raw(sheets, "Constituent_BOM")["Component"].str.strip().dropna()
        )
        qm_comps = (
            self._raw(sheets, "Incoming_QM")["Component"].str.strip().dropna()
        )
        names = pd.concat([bom_comps, qm_comps]).unique()
        df = pd.DataFrame({"component_name": sorted(names)})
        self._component_map = {
            name: idx + 1 for idx, name in enumerate(df["component_name"])
        }
        logger.info(
            "[normalize_components] %d components -> %s",
            len(df),
            list(df["component_name"]),
        )
        return df

    def normalize_lots(
        self, sheets: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Unique lots from Constituent_BOM ∪ Incoming_QM, BOM data takes
        priority for component/supplier/date when a lot appears in both.

        Lots are deduplicated by lot_no.  L-778 appears in both sources with
        different suppliers; the BOM authoritative supplier (SUP-C) is kept
        because BOM represents what was actually built into the product.
        """
        bom = self._raw(sheets, "Constituent_BOM")[
            ["LotNo", "Component", "Supplier", "MfgDate"]
        ].copy()
        bom.columns = ["lot_no", "component_name", "supplier_name", "mfg_date"]
        bom["lot_no"] = bom["lot_no"].str.strip()

        qm = self._raw(sheets, "Incoming_QM")[
            ["LotNo", "Component", "Supplier"]
        ].copy()
        qm.columns = ["lot_no", "component_name", "supplier_name"]
        qm["mfg_date"] = pd.NaT
        qm["lot_no"] = qm["lot_no"].str.strip()

        # BOM rows first so they win the drop_duplicates keep='first'
        all_lots = (
            pd.concat([bom, qm], ignore_index=True)
            .drop_duplicates(subset=["lot_no"], keep="first")
            .reset_index(drop=True)
        )

        all_lots["component_name"] = all_lots["component_name"].str.strip()
        all_lots["supplier_name"] = all_lots["supplier_name"].str.strip().str.upper()

        df = pd.DataFrame({
            "lot_no":       all_lots["lot_no"],
            "component_id": self._map_col(
                all_lots["component_name"], self._component_map,
                "component_name", "normalize_lots",
            ),
            "supplier_id":  self._map_col(
                all_lots["supplier_name"], self._supplier_map,
                "supplier_name", "normalize_lots",
            ),
            "mfg_date": pd.to_datetime(all_lots["mfg_date"], errors="coerce"),
        })

        self._lot_map = {
            lot_no: idx + 1
            for idx, lot_no in enumerate(df["lot_no"])
        }
        logger.info(
            "[normalize_lots] %d unique lots (includes L-778=%s)",
            len(df),
            "L-778" in self._lot_map,
        )
        return df

    def normalize_serials(
        self, sheets: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """All 300 finished-goods serials from AsBuilt_Serial."""
        abs_ = self._raw(sheets, "AsBuilt_Serial")
        abs_["SerialNo"] = abs_["SerialNo"].str.strip()
        abs_["FinishedMaterial"] = abs_["FinishedMaterial"].str.strip()

        df = pd.DataFrame({
            "serial_no":             abs_["SerialNo"],
            "finished_material_id":  self._map_col(
                abs_["FinishedMaterial"], self._material_map,
                "FinishedMaterial", "normalize_serials",
            ),
            "plant":                 abs_["Plant"].str.strip(),
            "line":                  abs_["Line"].str.strip(),
            "build_dt":              pd.to_datetime(abs_["BuildDT"], errors="coerce"),
            "shift":                 abs_["Shift"].str.strip(),
            "operator_id":           abs_["OperatorID"].str.strip(),
            "vendor_of_critical_assy": abs_["VendorOfCriticalAssy"].str.strip(),
            "coo_critical":          abs_["CountryOfOrigin_Critical"].str.strip(),
            "ecn_level":             abs_["ECN_Level"].str.strip(),
        })

        self._serial_map = {
            sn: idx + 1 for idx, sn in enumerate(df["serial_no"])
        }
        logger.info("[normalize_serials] %d serials loaded", len(df))
        return df

    # ------------------------------------------------------------------
    # Fact normalizers
    # ------------------------------------------------------------------

    def normalize_incoming_qm(
        self, sheets: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """1 486 incoming inspection measurements.

        * is_fail derived from Result == 'FAIL'
        * MeasuredValue coerced to float ('OK' -> NULL — non-numeric result
          flag, not a measurement)
        * DefectCode NULL -> empty string (NULL means the inspection passed,
          not that a defect was unrecorded)
        """
        qm = self._raw(sheets, "Incoming_QM")
        qm["Supplier"] = qm["Supplier"].str.strip().str.upper()
        qm["LotNo"] = qm["LotNo"].str.strip()
        qm["Component"] = qm["Component"].str.strip()
        qm["Result"] = qm["Result"].str.strip().str.upper()

        df = pd.DataFrame({
            "insp_lot":     qm["InspLot"].str.strip(),
            "component_id": self._map_col(
                qm["Component"], self._component_map,
                "Component", "normalize_incoming_qm",
            ),
            "supplier_id":  self._map_col(
                qm["Supplier"], self._supplier_map,
                "Supplier", "normalize_incoming_qm",
            ),
            "lot_id":       self._map_col(
                qm["LotNo"], self._lot_map,
                "LotNo", "normalize_incoming_qm",
            ),
            "insp_date":    pd.to_datetime(qm["InspDate"], errors="coerce"),
            "characteristic": qm["Characteristic"].str.strip(),
            # Non-numeric "OK" values -> NaN (NULL in DB)
            "measured_value": pd.to_numeric(qm["MeasuredValue"], errors="coerce"),
            "uom":          qm["UoM"].str.strip(),
            "result":       qm["Result"],
            "defect_code":  qm["DefectCode"].fillna("").str.strip(),
            "is_fail":      (qm["Result"] == "FAIL"),
        })

        fail_count = int(df["is_fail"].sum())
        logger.info(
            "[normalize_incoming_qm] %d rows  FAILs=%d (%.1f%%)",
            len(df),
            fail_count,
            100 * fail_count / len(df),
        )
        return df

    def normalize_process_measurements(
        self, sheets: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """300 process measurements — torque and leak test results per serial."""
        pm = self._raw(sheets, "Process_Measurements")
        pm["SerialNo"] = pm["SerialNo"].str.strip()
        pm["FinishedMaterial"] = pm["FinishedMaterial"].str.strip()
        pm["Torque_Result"] = pm["Torque_Result"].str.strip().str.upper()
        pm["Leak_Result"] = pm["Leak_Result"].str.strip().str.upper()

        df = pd.DataFrame({
            "serial_id":          self._map_col(
                pm["SerialNo"], self._serial_map,
                "SerialNo", "normalize_process_measurements",
            ),
            "finished_material_id": self._map_col(
                pm["FinishedMaterial"], self._material_map,
                "FinishedMaterial", "normalize_process_measurements",
            ),
            "build_date":    pd.to_datetime(pm["BuildDate"], errors="coerce"),
            "line":          pm["Line"].str.strip(),
            "shift":         pm["Shift"].str.strip(),
            "torque_nm":     pd.to_numeric(pm["Torque_Nm"], errors="coerce"),
            "torque_result": pm["Torque_Result"],
            "leak_rate_ccm": pd.to_numeric(pm["LeakRate_ccm"], errors="coerce"),
            "leak_result":   pm["Leak_Result"],
            "ecn_level":     pm["ECN_Level"].str.strip(),
            "is_torque_fail": (pm["Torque_Result"] == "FAIL"),
            "is_leak_fail":   (pm["Leak_Result"] == "FAIL"),
        })

        logger.info(
            "[normalize_process_measurements] %d rows  torque_fails=%d  leak_fails=%d",
            len(df),
            int(df["is_torque_fail"].sum()),
            int(df["is_leak_fail"].sum()),
        )
        return df

    def normalize_warranty_claims(
        self, sheets: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """13 field warranty claims.  NULL Region -> 'UNKNOWN'."""
        wc = self._raw(sheets, "Warranty_Claims")
        wc["SerialNo"] = wc["SerialNo"].str.strip()
        wc["Region"] = wc["Region"].fillna("UNKNOWN").str.strip()

        df = pd.DataFrame({
            "claim_id":       wc["ClaimID"].str.strip(),
            "serial_id":      self._map_col(
                wc["SerialNo"], self._serial_map,
                "SerialNo", "normalize_warranty_claims",
            ),
            "failure_date":   pd.to_datetime(wc["FailureDate"], errors="coerce"),
            "symptom":        wc["Symptom"].str.strip(),
            "mileage_or_hours": pd.to_numeric(wc["MileageOrHours"], errors="coerce"),
            "region":         wc["Region"],
            "severity":       wc["Severity"].str.strip(),
        })

        unknown_regions = int((df["region"] == "UNKNOWN").sum())
        logger.info(
            "[normalize_warranty_claims] %d rows  unknown_region=%d",
            len(df),
            unknown_regions,
        )
        return df

    def normalize_constituent_bom(
        self, sheets: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """1 200 as-built BOM rows — one component per serial number."""
        bom = self._raw(sheets, "Constituent_BOM")
        bom["SerialNo"] = bom["SerialNo"].str.strip()
        bom["Component"] = bom["Component"].str.strip()
        bom["Supplier"] = bom["Supplier"].str.strip().str.upper()
        bom["LotNo"] = bom["LotNo"].str.strip()

        df = pd.DataFrame({
            "serial_id":    self._map_col(
                bom["SerialNo"], self._serial_map,
                "SerialNo", "normalize_constituent_bom",
            ),
            "component_id": self._map_col(
                bom["Component"], self._component_map,
                "Component", "normalize_constituent_bom",
            ),
            "supplier_id":  self._map_col(
                bom["Supplier"], self._supplier_map,
                "Supplier", "normalize_constituent_bom",
            ),
            "lot_id":       self._map_col(
                bom["LotNo"], self._lot_map,
                "LotNo", "normalize_constituent_bom",
            ),
            "comp_serial":  bom["CompSerial"].str.strip(),
            "coo":          bom["COO"].str.strip(),
            "mfg_date":     pd.to_datetime(bom["MfgDate"], errors="coerce"),
            "cert_doc_id":  bom["CertDocID"].str.strip(),
        })

        logger.info("[normalize_constituent_bom] %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Aggregate normalizers
    # ------------------------------------------------------------------

    def normalize_aggregates(
        self, sheets: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Return pre-aggregated DataFrames for all three agg tables.

        Returns:
            Dict with keys ``agg_supplier_scorecard``, ``agg_coo_trends``,
            ``agg_coo_vs_supplier``.
        """
        # --- agg_supplier_scorecard ---
        sc = self._raw(sheets, "Supplier_Scorecard")
        sc["Supplier"] = sc["Supplier"].str.strip().str.upper()
        scorecard = pd.DataFrame({
            "supplier_id":         self._map_col(
                sc["Supplier"], self._supplier_map,
                "Supplier", "agg_supplier_scorecard",
            ),
            "lots_inspected":      pd.to_numeric(sc["LotsInspected"], errors="coerce"),
            "samples":             pd.to_numeric(sc["Samples"], errors="coerce"),
            "fails":               pd.to_numeric(sc["Fails"], errors="coerce"),
            "incoming_fail_rate":  pd.to_numeric(sc["Incoming_FailRate"], errors="coerce"),
            "units_built":         pd.to_numeric(sc["UnitsBuilt"], errors="coerce"),
            "units_with_claims":   pd.to_numeric(sc["UnitsWithClaims"], errors="coerce"),
            "warranty_claim_rate": pd.to_numeric(sc["Warranty_ClaimRate"], errors="coerce"),
            "process_drift_index": pd.to_numeric(sc["Process_Drift_Index"], errors="coerce"),
            "on_time_delivery_pct": pd.to_numeric(sc["OnTimeDelivery_%"], errors="coerce"),
            "avg_lead_time_days":  pd.to_numeric(sc["AvgLeadTime_Days"], errors="coerce"),
            "quality_score":       pd.to_numeric(sc["Quality_Score"], errors="coerce"),
            "tier":                sc["Tier"].str.strip(),
            "premium_service_fit": sc["Premium_Service_Fit"].str.strip(),
        })

        # --- agg_coo_trends ---
        ct = self._raw(sheets, "COO_Trends")
        coo_trends = pd.DataFrame({
            "coo":                     ct["COO"].str.strip(),
            "samples":                 pd.to_numeric(ct["Samples"], errors="coerce"),
            "fails":                   pd.to_numeric(ct["Fails"], errors="coerce"),
            "coo_incoming_fail_rate":  pd.to_numeric(ct["COO_Incoming_FailRate"], errors="coerce"),
            "coo_warranty_claim_rate": pd.to_numeric(ct["COO_Warranty_ClaimRate"], errors="coerce"),
        })

        # --- agg_coo_vs_supplier ---
        cvs = self._raw(sheets, "COO_vs_Supplier")
        cvs["Supplier"] = cvs["Supplier"].str.strip().str.upper()
        coo_vs_sup = pd.DataFrame({
            "supplier_id":             self._map_col(
                cvs["Supplier"], self._supplier_map,
                "Supplier", "agg_coo_vs_supplier",
            ),
            "coo":                     cvs["COO"].str.strip(),
            "incoming_fail_rate":      pd.to_numeric(cvs["Incoming_FailRate"], errors="coerce"),
            "warranty_claim_rate":     pd.to_numeric(cvs["Warranty_ClaimRate"], errors="coerce"),
            "quality_score":           pd.to_numeric(cvs["Quality_Score"], errors="coerce"),
            "tier":                    cvs["Tier"].str.strip(),
            "coo_incoming_fail_rate":  pd.to_numeric(cvs["COO_Incoming_FailRate"], errors="coerce"),
            "coo_warranty_claim_rate": pd.to_numeric(cvs["COO_Warranty_ClaimRate"], errors="coerce"),
            "beats_coo_avg":           cvs["Beats_COO_Avg"].str.strip(),
        })

        logger.info(
            "[normalize_aggregates] scorecard=%d  coo_trends=%d  coo_vs_supplier=%d",
            len(scorecard),
            len(coo_trends),
            len(coo_vs_sup),
        )
        return {
            "agg_supplier_scorecard": scorecard,
            "agg_coo_trends":         coo_trends,
            "agg_coo_vs_supplier":    coo_vs_sup,
        }

    # ------------------------------------------------------------------
    # Reference data normalizers
    # ------------------------------------------------------------------

    def normalize_reference_data(
        self, sheets: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Return cleaned DataFrames for AI_Insights and Action_Playbook.

        Rows where PatternDetected is NULL are sparse placeholder rows in the
        source spreadsheet — they are dropped before loading.

        Returns:
            Dict with keys ``ref_ai_insights`` and ``ref_action_playbook``.
        """
        # --- ref_ai_insights ---
        ai = self._raw(sheets, "AI_Insights")
        # Drop the two trailing unnamed columns if present
        ai = ai[[c for c in ai.columns if not str(c).startswith("Unnamed")]]
        ai = ai[ai["PatternDetected"].notna()].reset_index(drop=True)

        for col in ai.select_dtypes(include=["object", "string"]).columns:
            ai[col] = ai[col].str.strip()

        insights = pd.DataFrame({
            "pattern_detected":   ai["PatternDetected"],
            "evidence":           ai["Evidence"],
            "risk_or_opportunity": ai["RiskOrOpportunity"],
            "ai_guidance":        ai["AI_Guidance"],
            "suggested_actionables": ai.get("SuggestedActionables"),
        })

        # --- ref_action_playbook ---
        ap = self._raw(sheets, "Action_Playbook")
        for col in ap.select_dtypes(include=["object", "string"]).columns:
            ap[col] = ap[col].str.strip()

        playbook = pd.DataFrame({
            "insight_type":    ap["InsightType"],
            "typical_action":  ap["TypicalAction"],
            "where_it_fits":   ap["WhereItFits"],
            "sap_mes_touchpoint": ap["SAP_or_MES_Touchpoint"],
        })

        logger.info(
            "[normalize_reference_data] ai_insights=%d (dropped %d null-pattern rows)  playbook=%d",
            len(insights),
            sheets["AI_Insights"].shape[0]
            - sheets["AI_Insights"]["PatternDetected"].notna().sum(),
            len(playbook),
        )
        return {
            "ref_ai_insights":    insights,
            "ref_action_playbook": playbook,
        }

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        sheets: dict[str, pd.DataFrame],
        engine: Engine,
    ) -> dict[str, int]:
        """Execute the full ETL in dependency order and load into *engine*.

        Execution order
        ---------------
        1. Dimensions  (supplier -> material -> component -> lot -> serial)
        2. Facts       (incoming_qm -> process_measurements -> warranty_claims
                        -> constituent_bom)
        3. Aggregates  (supplier_scorecard · coo_trends · coo_vs_supplier)
        4. Reference   (ai_insights · action_playbook)
        5. Integrity   (L-778 must be in dim_lot AND joinable to
                        fact_incoming_qm)

        Args:
            sheets: Raw sheet dict from :func:`loader.load_all_sheets`.
            engine: Connected SQLAlchemy engine.

        Returns:
            ``pipeline_report`` — mapping of ``{table_name: row_count}``.

        Raises:
            DataIntegrityError: If the L-778 lot integrity check fails.
        """
        report: dict[str, int] = {}

        logger.info("-" * 60)
        logger.info("NormalizationPipeline — starting full ETL")
        logger.info("-" * 60)

        # -- 1. Dimensions ----------------------------------------------
        logger.info("[Phase] Dimensions")

        report["dim_supplier"] = self._write_table(
            self.normalize_suppliers(sheets),
            "dim_supplier", "supplier_id", engine,
        )
        report["dim_material"] = self._write_table(
            self.normalize_materials(sheets),
            "dim_material", "material_id", engine,
        )
        report["dim_component"] = self._write_table(
            self.normalize_components(sheets),
            "dim_component", "component_id", engine,
        )
        report["dim_lot"] = self._write_table(
            self.normalize_lots(sheets),
            "dim_lot", "lot_id", engine,
        )
        report["dim_serial"] = self._write_table(
            self.normalize_serials(sheets),
            "dim_serial", "serial_id", engine,
        )

        # -- 2. Facts ---------------------------------------------------
        logger.info("[Phase] Facts")

        report["fact_incoming_qm"] = self._write_table(
            self.normalize_incoming_qm(sheets),
            "fact_incoming_qm", "id", engine,
        )
        report["fact_process_measurements"] = self._write_table(
            self.normalize_process_measurements(sheets),
            "fact_process_measurements", "id", engine,
        )
        report["fact_warranty_claims"] = self._write_table(
            self.normalize_warranty_claims(sheets),
            "fact_warranty_claims", "id", engine,
        )
        report["fact_constituent_bom"] = self._write_table(
            self.normalize_constituent_bom(sheets),
            "fact_constituent_bom", "id", engine,
        )

        # -- 3. Aggregates ----------------------------------------------
        logger.info("[Phase] Aggregates")

        for table_name, df in self.normalize_aggregates(sheets).items():
            pk = "id"
            report[table_name] = self._write_table(df, table_name, pk, engine)

        # -- 4. Reference -----------------------------------------------
        logger.info("[Phase] Reference")

        for table_name, df in self.normalize_reference_data(sheets).items():
            report[table_name] = self._write_table(df, table_name, "id", engine)

        # -- 5. Integrity check -----------------------------------------
        logger.info("[Phase] Integrity checks")

        self._verify_l778(engine)

        # -- Summary ----------------------------------------------------
        total_rows = sum(report.values())
        logger.info("-" * 60)
        logger.info("ETL complete — %d tables  %d total rows", len(report), total_rows)
        for tbl, cnt in report.items():
            logger.info("  %-35s  %d", tbl, cnt)
        logger.info("-" * 60)

        return report

    # ------------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------------

    def _verify_l778(self, engine: Engine) -> None:
        """Verify lot L-778 exists in dim_lot and is joinable to fact_incoming_qm.

        Raises:
            DataIntegrityError: If L-778 is absent from dim_lot or has no
                corresponding rows in fact_incoming_qm.
        """
        lot_df = pd.read_sql(
            "SELECT lot_id FROM dim_lot WHERE lot_no = 'L-778'",
            engine,
        )
        if lot_df.empty:
            raise DataIntegrityError(
                "Integrity check FAILED: lot L-778 is missing from dim_lot. "
                "Check that Constituent_BOM was loaded before normalize_lots()."
            )

        lot_id = int(lot_df.iloc[0]["lot_id"])

        qm_df = pd.read_sql(
            f"SELECT COUNT(*) AS cnt FROM fact_incoming_qm WHERE lot_id = {lot_id}",
            engine,
        )
        cnt = int(qm_df.iloc[0]["cnt"])

        if cnt == 0:
            raise DataIntegrityError(
                f"Integrity check FAILED: L-778 (lot_id={lot_id}) is in dim_lot "
                "but has no matching rows in fact_incoming_qm. "
                "Check LotNo mapping in normalize_incoming_qm()."
            )

        logger.info(
            "[integrity] L-778 -> lot_id=%d  fact_incoming_qm rows=%d  OK",
            lot_id,
            cnt,
        )
