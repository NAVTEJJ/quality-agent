"""
Tests for the ingestion pipeline (Phase 1).

Sections
--------
1. loader          – sheet loading, column cleaning
2. normalizer      – snake_case, whitespace
3. profiler        – tab inventory JSON
4. integration     – database-level assertions against the full pipeline
   (module-scoped fixture builds one in-memory DB for all integration tests)

Run with:  pytest tests/test_ingestion.py -v
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from app.ingestion.loader import load_all_sheets
from app.ingestion.normalizer import normalise_columns
from app.ingestion.profiler import generate_tab_inventory
from configs import settings


# ---------------------------------------------------------------------------
# Fixtures — raw sheet samples
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_sheets() -> dict[str, pd.DataFrame]:
    return {
        "Sheet1": pd.DataFrame({
            "LotID":      ["L001", "L002", "L003"],
            "Result":     ["PASS", "FAIL", "PASS"],
            "DefectCode": [None, "D01", None],
        }),
        "Sheet2": pd.DataFrame({
            "SupplierID": ["S1", "S2"],
            "Score":      [90, 75],
            "Region":     [None, "West"],
        }),
    }


# ---------------------------------------------------------------------------
# Module-scoped DB fixture for integration tests
# ---------------------------------------------------------------------------

_EXPECTED_SHEETS = [
    "README", "AsBuilt_Serial", "Constituent_BOM", "Incoming_QM",
    "Process_Measurements", "Warranty_Claims", "Vendor_Engineering_Profile",
    "Supplier_Scorecard", "COO_Trends", "COO_vs_Supplier",
    "AI_Insights", "Action_Playbook", "Data_Dictionary",
]


@pytest.fixture(scope="module")
def loaded_engine():
    """Build one in-memory DB from the real workbook for all integration tests.

    Skipped when the workbook is absent (CI without assets).
    """
    if not settings.EXCEL_PATH.exists():
        pytest.skip("Source workbook not present — skipping integration tests")

    from app.ingestion.normalizer import NormalizationPipeline
    from app.models.schema import get_engine, init_database

    sheets = load_all_sheets(settings.EXCEL_PATH)
    engine = get_engine("sqlite:///:memory:")
    init_database(engine)
    NormalizationPipeline().run_full_pipeline(sheets, engine)
    return engine


# ---------------------------------------------------------------------------
# 1. loader
# ---------------------------------------------------------------------------

def test_load_all_sheets_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_all_sheets(Path("/nonexistent/path/file.xlsx"))


def test_load_all_sheets_returns_dict():
    if not settings.EXCEL_PATH.exists():
        pytest.skip("Source workbook not present")
    sheets = load_all_sheets(settings.EXCEL_PATH)
    assert isinstance(sheets, dict)
    assert len(sheets) > 0
    for name, df in sheets.items():
        assert isinstance(df, pd.DataFrame), f"Sheet '{name}' is not a DataFrame"
        assert len(df) > 0, f"Sheet '{name}' is unexpectedly empty"


def test_ai_insights_unnamed_columns_dropped():
    if not settings.EXCEL_PATH.exists():
        pytest.skip("Source workbook not present")
    sheets = load_all_sheets(settings.EXCEL_PATH)
    if "AI_Insights" in sheets:
        cols = sheets["AI_Insights"].columns
        assert "Unnamed: 5" not in cols
        assert "Unnamed: 6" not in cols


# ---------------------------------------------------------------------------
# 2. normalizer (column helpers)
# ---------------------------------------------------------------------------

def test_normalise_columns_snake_case():
    df = pd.DataFrame({"Lot ID": [1], "Part Number": [2], "DefectCode": [3]})
    result = normalise_columns(df)
    assert list(result.columns) == ["lot_id", "part_number", "defectcode"]


def test_normalise_columns_strips_whitespace():
    df = pd.DataFrame({"Name": ["  Alice  ", "Bob "]})
    result = normalise_columns(df)
    assert result["name"].tolist() == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# 3. profiler — tab inventory
# ---------------------------------------------------------------------------

def test_generate_tab_inventory_structure(sample_sheets, tmp_path):
    out = tmp_path / "tab_inventory.json"
    inventory = generate_tab_inventory(sample_sheets, output_path=out)
    assert set(inventory.keys()) == {"Sheet1", "Sheet2"}
    for entry in inventory.values():
        assert "row_count" in entry
        assert "columns" in entry
        assert "null_counts" in entry
        assert "sample_values" in entry


def test_generate_tab_inventory_writes_json(sample_sheets, tmp_path):
    out = tmp_path / "tab_inventory.json"
    generate_tab_inventory(sample_sheets, output_path=out)
    assert out.exists()
    with out.open() as fh:
        data = json.load(fh)
    assert "Sheet1" in data


def test_generate_tab_inventory_null_counts(sample_sheets, tmp_path):
    out = tmp_path / "inv.json"
    inventory = generate_tab_inventory(sample_sheets, output_path=out)
    assert inventory["Sheet1"]["null_counts"].get("DefectCode", 0) == 2


def test_generate_tab_inventory_sample_values_for_id_cols(sample_sheets, tmp_path):
    out = tmp_path / "inv.json"
    inventory = generate_tab_inventory(sample_sheets, output_path=out)
    assert "LotID" in inventory["Sheet1"]["sample_values"]


# ---------------------------------------------------------------------------
# 4. Integration — database assertions
# ---------------------------------------------------------------------------

def test_all_sheets_loaded():
    """All 13 expected sheet names must be present in the loaded dict."""
    if not settings.EXCEL_PATH.exists():
        pytest.skip("Source workbook not present")
    sheets = load_all_sheets(settings.EXCEL_PATH)
    loaded_names = set(sheets.keys())
    for expected in _EXPECTED_SHEETS:
        assert expected in loaded_names, f"Missing sheet: {expected}"
    assert len(loaded_names) == 13


def test_row_counts(loaded_engine):
    """Exact row counts for every sheet that feeds a fact or dimension table."""
    expected = {
        "dim_serial":             300,
        "fact_constituent_bom":  1200,
        "fact_incoming_qm":      1486,
        "fact_process_measurements": 300,
        "fact_warranty_claims":    13,
        "agg_supplier_scorecard":   5,
        "agg_coo_trends":           5,
    }
    for table, expected_count in expected.items():
        result = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", loaded_engine)
        actual = int(result.iloc[0]["n"])
        assert actual == expected_count, (
            f"{table}: expected {expected_count} rows, got {actual}"
        )


def test_join_integrity(loaded_engine):
    """Every serial_id in fact tables must resolve to a dim_serial row."""
    checks = [
        ("fact_process_measurements", "serial_id"),
        ("fact_warranty_claims",      "serial_id"),
        ("fact_constituent_bom",      "serial_id"),
    ]
    for table, col in checks:
        orphans = pd.read_sql(
            f"SELECT COUNT(*) as n FROM {table} "
            f"WHERE {col} NOT IN (SELECT serial_id FROM dim_serial)",
            loaded_engine,
        )
        assert int(orphans.iloc[0]["n"]) == 0, (
            f"Orphan {col} rows found in {table}"
        )


def test_l778_exists(loaded_engine):
    """Lot L-778 must be present in dim_lot and have inspection records."""
    lot = pd.read_sql(
        "SELECT lot_id FROM dim_lot WHERE lot_no = 'L-778'",
        loaded_engine,
    )
    assert not lot.empty, "L-778 missing from dim_lot"

    lot_id = int(lot.iloc[0]["lot_id"])
    qm = pd.read_sql(
        f"SELECT COUNT(*) as n FROM fact_incoming_qm WHERE lot_id = {lot_id}",
        loaded_engine,
    )
    assert int(qm.iloc[0]["n"]) >= 1, (
        f"L-778 (lot_id={lot_id}) has no rows in fact_incoming_qm"
    )


def test_line2_night_drift(loaded_engine):
    """LINE-2 Night shift must have at least 9 confirmed torque failures."""
    result = pd.read_sql(
        "SELECT SUM(is_torque_fail) as fails "
        "FROM fact_process_measurements "
        "WHERE line = 'LINE-2' AND shift = 'Night'",
        loaded_engine,
    )
    fails = int(result.iloc[0]["fails"])
    assert fails >= 9, (
        f"Expected >= 9 LINE-2 Night torque failures, found {fails}"
    )


def test_no_orphan_lots(loaded_engine):
    """Every lot_id referenced in fact_incoming_qm must exist in dim_lot."""
    orphans = pd.read_sql(
        "SELECT COUNT(*) as n FROM fact_incoming_qm "
        "WHERE lot_id NOT IN (SELECT lot_id FROM dim_lot)",
        loaded_engine,
    )
    assert int(orphans.iloc[0]["n"]) == 0, "Orphan lot_id values in fact_incoming_qm"


def test_supplier_scorecard_complete(loaded_engine):
    """All 5 suppliers must appear in agg_supplier_scorecard."""
    scorecard_count = pd.read_sql(
        "SELECT COUNT(DISTINCT supplier_id) as n FROM agg_supplier_scorecard",
        loaded_engine,
    )
    assert int(scorecard_count.iloc[0]["n"]) == 5, (
        "agg_supplier_scorecard does not contain all 5 suppliers"
    )


def test_derived_flags(loaded_engine):
    """is_fail in fact_incoming_qm must contain both True and False values."""
    result = pd.read_sql(
        "SELECT "
        "  SUM(CASE WHEN is_fail = 1 THEN 1 ELSE 0 END) as true_count, "
        "  SUM(CASE WHEN is_fail = 0 THEN 1 ELSE 0 END) as false_count "
        "FROM fact_incoming_qm",
        loaded_engine,
    )
    assert int(result.iloc[0]["true_count"])  > 0, "No FAIL rows found in fact_incoming_qm"
    assert int(result.iloc[0]["false_count"]) > 0, "No PASS rows found in fact_incoming_qm"
