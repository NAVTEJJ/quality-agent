"""
Tests for NormalizationPipeline (Phase 1, Step 3).

Isolated from the real workbook — all tests build minimal synthetic sheets.
Integration tests that require the real Excel file are gated on
settings.EXCEL_PATH.exists().

Run with:  pytest tests/test_normalizer.py -v
"""
import pandas as pd
import pytest
from sqlalchemy import text

from app.ingestion.normalizer import DataIntegrityError, NormalizationPipeline
from app.models.schema import get_engine, init_database
from configs import settings


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic sheets that mirror real column names
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_sheets() -> dict[str, pd.DataFrame]:
    """Smallest possible set of sheets that exercises every normalize method."""
    return {
        "Supplier_Scorecard": pd.DataFrame({
            "Supplier":             ["SUP-A", "SUP-C"],
            "COO":                  ["Germany", "China"],
            "LotsInspected":        [10, 20],
            "Samples":              [100, 200],
            "Fails":                [2, 14],
            "Incoming_FailRate":    [0.02, 0.07],
            "Engineering_Maturity": ["High", "Medium"],
            "Engineering_Maturity_Score": [95, 72],
            "UnitsBuilt":           [50, 80],
            "UnitsWithClaims":      [1, 6],
            "Warranty_ClaimRate":   [0.02, 0.075],
            "Process_Drift_Index":  [0.06, 0.055],
            "OnTimeDelivery_%":     [97.2, 99.0],
            "AvgLeadTime_Days":     [23.2, 15.1],
            "Quality_Score":        [80, 33],
            "Tier":                 ["Standard", "Watchlist"],
            "Premium_Service_Fit":  ["No", "No"],
        }),
        "Vendor_Engineering_Profile": pd.DataFrame({
            "Supplier":             ["SUP-A", "SUP-C"],
            "COO":                  ["Germany", "China"],
            "Engineering_Maturity": ["High", "Medium"],
            "Process_Cpk":          [1.80, 1.35],
            "Design_Ownership":     ["Co-design", "Build-to-print"],
            "Typical_Project_Type": ["Safety-critical", "Cost-optimized"],
            "Engineering_Maturity_Score": [95, 72],
        }),
        "AsBuilt_Serial": pd.DataFrame({
            "SerialNo":              ["SR001", "SR002"],
            "FinishedMaterial":      ["BRAKE-MOD-220", "AXLE-ASSY-100"],
            "Plant":                 ["PL01", "PL01"],
            "Line":                  ["LINE-1", "LINE-2"],
            "BuildDT":               ["2025-10-01", "2025-10-02"],
            "Shift":                 ["Day", "Night"],
            "OperatorID":            ["OP001", "OP002"],
            "VendorOfCriticalAssy":  ["SUP-A", "SUP-C"],
            "CountryOfOrigin_Critical": ["Germany", "China"],
            "ECN_Level":             ["E1", "E1"],
        }),
        "Constituent_BOM": pd.DataFrame({
            "SerialNo":   ["SR001", "SR001", "SR002"],
            "Component":  ["SENSOR-HALL", "BEARING-SET", "SENSOR-HALL"],
            "CompSerial": ["SEN-001", "BEA-001", "SEN-002"],
            "Supplier":   ["SUP-C", "SUP-A", "SUP-C"],
            "COO":        ["China", "Germany", "China"],
            "LotNo":      ["L-778", "LOT-BEAR-001", "L-778"],
            "MfgDate":    ["2025-09-01", "2025-09-02", "2025-09-01"],
            "CertDocID":  ["CERT-001", "CERT-002", "CERT-003"],
        }),
        "Incoming_QM": pd.DataFrame({
            "InspLot":       ["IL001", "IL001", "IL002"],
            "Component":     ["SENSOR-HALL", "SENSOR-HALL", "BEARING-SET"],
            "Supplier":      ["SUP-C", "SUP-C", "SUP-A"],
            "COO":           ["China", "China", "Germany"],
            "LotNo":         ["L-778", "L-778", "LOT-BEAR-001"],
            "InspDate":      ["2025-09-25", "2025-09-26", "2025-09-27"],
            "Characteristic": ["Signal Stability", "Ingress Protection", "Radial Play"],
            "MeasuredValue": [1100.036, "OK", 0.039],
            "UoM":           ["mV", "OK", "mm"],
            "Result":        ["FAIL", "PASS", "PASS"],
            "DefectCode":    ["E-SIG", None, None],
        }),
        "Process_Measurements": pd.DataFrame({
            "SerialNo":        ["SR001", "SR002"],
            "FinishedMaterial": ["BRAKE-MOD-220", "AXLE-ASSY-100"],
            "BuildDate":       ["2025-10-01", "2025-10-02"],
            "Line":            ["LINE-1", "LINE-2"],
            "Shift":           ["Day", "Night"],
            "Torque_Nm":       [20.27, 18.5],
            "Torque_Result":   ["PASS", "FAIL"],
            "LeakRate_ccm":    [1.17, 0.03],
            "Leak_Result":     ["PASS", "PASS"],
            "ECN_Level":       ["E1", "E1"],
        }),
        "Warranty_Claims": pd.DataFrame({
            "ClaimID":        ["WC001", "WC002"],
            "SerialNo":       ["SR001", "SR002"],
            "FailureDate":    ["2025-11-30", "2026-01-24"],
            "Symptom":        ["Sensor Failure", "Noise/Vibration"],
            "MileageOrHours": [1858, 1694],
            "Region":         ["EU", None],
            "Severity":       ["Low", "Low"],
        }),
        "COO_Trends": pd.DataFrame({
            "COO":                    ["China", "Germany"],
            "Samples":                [200, 100],
            "Fails":                  [14, 2],
            "COO_Incoming_FailRate":  [0.07, 0.02],
            "COO_Warranty_ClaimRate": [0.08, 0.02],
        }),
        "COO_vs_Supplier": pd.DataFrame({
            "Supplier":               ["SUP-A", "SUP-C"],
            "COO":                    ["Germany", "China"],
            "Incoming_FailRate":      [0.02, 0.07],
            "Warranty_ClaimRate":     [0.016, 0.082],
            "Quality_Score":          [80, 33],
            "Tier":                   ["Standard", "Watchlist"],
            "COO_Incoming_FailRate":  [0.02, 0.07],
            "COO_Warranty_ClaimRate": [0.02, 0.08],
            "Beats_COO_Avg":          ["Yes", "No"],
        }),
        "AI_Insights": pd.DataFrame({
            "PatternDetected":    ["Lot L-778 high fail rate", None],
            "Evidence":           ["High FAIL in QM", None],
            "RiskOrOpportunity":  ["Risk", None],
            "AI_Guidance":        ["Increase sampling", None],
            "SuggestedActionables": ["Block lot", None],
            "Unnamed: 5":         [None, None],
            "Unnamed: 6":         [None, None],
        }),
        "Action_Playbook": pd.DataFrame({
            "InsightType":         ["High lot defect rate"],
            "TypicalAction":       ["Increase sampling"],
            "WhereItFits":         ["QM Incoming"],
            "SAP_or_MES_Touchpoint": ["QA32"],
        }),
    }


@pytest.fixture()
def pipeline() -> NormalizationPipeline:
    return NormalizationPipeline()


@pytest.fixture()
def loaded_pipeline(pipeline, minimal_sheets):
    """Pipeline with all dim maps pre-populated using minimal_sheets."""
    pipeline.normalize_suppliers(minimal_sheets)
    pipeline.normalize_materials(minimal_sheets)
    pipeline.normalize_components(minimal_sheets)
    pipeline.normalize_lots(minimal_sheets)
    pipeline.normalize_serials(minimal_sheets)
    return pipeline


@pytest.fixture()
def mem_engine():
    eng = get_engine("sqlite:///:memory:")
    init_database(eng)
    return eng


# ---------------------------------------------------------------------------
# normalize_suppliers
# ---------------------------------------------------------------------------

def test_suppliers_row_count(pipeline, minimal_sheets):
    df = pipeline.normalize_suppliers(minimal_sheets)
    assert len(df) == 2


def test_suppliers_map_populated(pipeline, minimal_sheets):
    pipeline.normalize_suppliers(minimal_sheets)
    assert "SUP-A" in pipeline._supplier_map
    assert "SUP-C" in pipeline._supplier_map


def test_suppliers_uppercase_normalisation(pipeline, minimal_sheets):
    # Inject a lowercase supplier name to prove normalisation fires
    minimal_sheets["Supplier_Scorecard"].loc[0, "Supplier"] = " sup-a "
    minimal_sheets["Vendor_Engineering_Profile"].loc[0, "Supplier"] = " sup-a "
    pipeline.normalize_suppliers(minimal_sheets)
    assert "SUP-A" in pipeline._supplier_map


def test_suppliers_has_cpk_column(pipeline, minimal_sheets):
    df = pipeline.normalize_suppliers(minimal_sheets)
    assert "process_cpk" in df.columns
    assert df["process_cpk"].notna().all()


# ---------------------------------------------------------------------------
# normalize_materials
# ---------------------------------------------------------------------------

def test_materials_unique(pipeline, minimal_sheets):
    df = pipeline.normalize_materials(minimal_sheets)
    assert len(df) == len(df["material_name"].unique())


def test_materials_map_populated(pipeline, minimal_sheets):
    pipeline.normalize_materials(minimal_sheets)
    assert "BRAKE-MOD-220" in pipeline._material_map
    assert "AXLE-ASSY-100" in pipeline._material_map


# ---------------------------------------------------------------------------
# normalize_components
# ---------------------------------------------------------------------------

def test_components_union_of_bom_and_qm(pipeline, minimal_sheets):
    df = pipeline.normalize_components(minimal_sheets)
    names = set(df["component_name"])
    assert "SENSOR-HALL" in names
    assert "BEARING-SET" in names


# ---------------------------------------------------------------------------
# normalize_lots
# ---------------------------------------------------------------------------

def test_lots_deduplicated_by_lot_no(pipeline, minimal_sheets):
    pipeline.normalize_suppliers(minimal_sheets)
    pipeline.normalize_components(minimal_sheets)
    df = pipeline.normalize_lots(minimal_sheets)
    # L-778 appears 2× in BOM but must be deduplicated to 1 row
    assert df[df["lot_no"] == "L-778"].shape[0] == 1


def test_l778_in_lot_map(pipeline, minimal_sheets):
    pipeline.normalize_suppliers(minimal_sheets)
    pipeline.normalize_components(minimal_sheets)
    pipeline.normalize_lots(minimal_sheets)
    assert "L-778" in pipeline._lot_map


def test_lots_bom_supplier_wins_over_qm(pipeline, minimal_sheets):
    """BOM is the authoritative source for the lot→supplier assignment."""
    pipeline.normalize_suppliers(minimal_sheets)
    pipeline.normalize_components(minimal_sheets)
    df = pipeline.normalize_lots(minimal_sheets)
    l778 = df[df["lot_no"] == "L-778"].iloc[0]
    # BOM says SUP-C for L-778
    expected_sup_id = pipeline._supplier_map["SUP-C"]
    assert l778["supplier_id"] == expected_sup_id


# ---------------------------------------------------------------------------
# normalize_incoming_qm
# ---------------------------------------------------------------------------

def test_incoming_qm_is_fail_derived(loaded_pipeline, minimal_sheets):
    df = loaded_pipeline.normalize_incoming_qm(minimal_sheets)
    fail_rows = df[df["result"] == "FAIL"]
    assert fail_rows["is_fail"].all()
    pass_rows = df[df["result"] == "PASS"]
    assert not pass_rows["is_fail"].any()


def test_incoming_qm_defect_code_null_becomes_empty_string(loaded_pipeline, minimal_sheets):
    df = loaded_pipeline.normalize_incoming_qm(minimal_sheets)
    assert df["defect_code"].isna().sum() == 0   # no NaNs
    assert "" in df["defect_code"].values         # nulls became ''


def test_incoming_qm_measured_value_ok_becomes_null(loaded_pipeline, minimal_sheets):
    """'OK' (non-numeric) measured values must be coerced to NaN."""
    df = loaded_pipeline.normalize_incoming_qm(minimal_sheets)
    # The 'Ingress Protection' row has MeasuredValue='OK'
    ip_row = df[df["characteristic"] == "Ingress Protection"]
    assert ip_row["measured_value"].isna().all()


def test_incoming_qm_l778_lot_id_assigned(loaded_pipeline, minimal_sheets):
    df = loaded_pipeline.normalize_incoming_qm(minimal_sheets)
    l778_rows = df[df["lot_id"] == loaded_pipeline._lot_map["L-778"]]
    assert len(l778_rows) == 2   # 2 L-778 rows in minimal_sheets Incoming_QM


# ---------------------------------------------------------------------------
# normalize_process_measurements
# ---------------------------------------------------------------------------

def test_process_measurements_is_torque_fail(loaded_pipeline, minimal_sheets):
    df = loaded_pipeline.normalize_process_measurements(minimal_sheets)
    # SR002 has Torque_Result=FAIL in minimal_sheets
    fail_rows = df[df["torque_result"] == "FAIL"]
    assert fail_rows["is_torque_fail"].all()


def test_process_measurements_is_leak_fail_default_false(loaded_pipeline, minimal_sheets):
    df = loaded_pipeline.normalize_process_measurements(minimal_sheets)
    assert not df["is_leak_fail"].any()


# ---------------------------------------------------------------------------
# normalize_warranty_claims
# ---------------------------------------------------------------------------

def test_warranty_claims_null_region_becomes_unknown(loaded_pipeline, minimal_sheets):
    df = loaded_pipeline.normalize_warranty_claims(minimal_sheets)
    assert "UNKNOWN" in df["region"].values
    assert df["region"].isna().sum() == 0


# ---------------------------------------------------------------------------
# normalize_reference_data
# ---------------------------------------------------------------------------

def test_reference_data_drops_null_pattern_rows(pipeline, minimal_sheets):
    result = pipeline.normalize_reference_data(minimal_sheets)
    ai = result["ref_ai_insights"]
    # minimal_sheets has 1 non-null PatternDetected row
    assert len(ai) == 1
    assert ai["pattern_detected"].notna().all()


def test_reference_data_playbook_row_count(pipeline, minimal_sheets):
    result = pipeline.normalize_reference_data(minimal_sheets)
    assert len(result["ref_action_playbook"]) == 1


# ---------------------------------------------------------------------------
# run_full_pipeline — integration tests
# ---------------------------------------------------------------------------

def test_run_full_pipeline_returns_report(pipeline, minimal_sheets, mem_engine):
    report = pipeline.run_full_pipeline(minimal_sheets, mem_engine)
    expected_tables = {
        "dim_supplier", "dim_material", "dim_component", "dim_lot", "dim_serial",
        "fact_incoming_qm", "fact_process_measurements", "fact_warranty_claims",
        "fact_constituent_bom", "agg_supplier_scorecard", "agg_coo_trends",
        "agg_coo_vs_supplier", "ref_ai_insights", "ref_action_playbook",
    }
    assert expected_tables == set(report.keys())


def test_run_full_pipeline_row_counts(pipeline, minimal_sheets, mem_engine):
    report = pipeline.run_full_pipeline(minimal_sheets, mem_engine)
    assert report["dim_supplier"] == 2
    assert report["dim_serial"] == 2
    assert report["fact_incoming_qm"] == 3
    assert report["fact_warranty_claims"] == 2
    assert report["ref_ai_insights"] == 1   # 1 null row dropped


def test_run_full_pipeline_l778_integrity_passes(pipeline, minimal_sheets, mem_engine):
    """Pipeline completes without DataIntegrityError when L-778 is present."""
    pipeline.run_full_pipeline(minimal_sheets, mem_engine)   # must not raise


def test_run_full_pipeline_l778_integrity_fails(pipeline, minimal_sheets, mem_engine):
    """DataIntegrityError raised if L-778 is removed from the source data."""
    minimal_sheets["Constituent_BOM"]["LotNo"] = "LOT-OTHER-001"
    minimal_sheets["Incoming_QM"]["LotNo"] = "LOT-OTHER-001"
    with pytest.raises(DataIntegrityError, match="L-778"):
        pipeline.run_full_pipeline(minimal_sheets, mem_engine)


def test_run_full_pipeline_is_idempotent(pipeline, minimal_sheets, mem_engine):
    """Running the pipeline twice must succeed (if_exists='replace' semantics)."""
    pipeline.run_full_pipeline(minimal_sheets, mem_engine)
    pipeline2 = NormalizationPipeline()
    report2 = pipeline2.run_full_pipeline(minimal_sheets, mem_engine)
    assert report2["fact_incoming_qm"] == 3


# ---------------------------------------------------------------------------
# Real-workbook integration test
# ---------------------------------------------------------------------------

def test_full_pipeline_against_real_workbook():
    """End-to-end test against the actual Excel file.  Skipped in CI."""
    if not settings.EXCEL_PATH.exists():
        pytest.skip("Source workbook not present")

    from app.ingestion.loader import load_all_sheets

    sheets = load_all_sheets(settings.EXCEL_PATH)
    engine = get_engine("sqlite:///:memory:")
    init_database(engine)
    pipeline = NormalizationPipeline()
    report = pipeline.run_full_pipeline(sheets, engine)

    assert report["dim_supplier"] == 5
    assert report["dim_material"] == 3
    assert report["dim_component"] == 4
    assert report["dim_serial"] == 300
    assert report["fact_incoming_qm"] == 1486
    assert report["fact_process_measurements"] == 300
    assert report["fact_warranty_claims"] == 13
    assert report["fact_constituent_bom"] == 1200
