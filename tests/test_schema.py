"""
Tests for the SQLAlchemy ORM schema and database utilities.

Run with:  pytest tests/test_schema.py -v
"""
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.models.schema import (
    AggCooTrends,
    AggCooVsSupplier,
    AggSupplierScorecard,
    Base,
    DimComponent,
    DimLot,
    DimMaterial,
    DimSerial,
    DimSupplier,
    FactConstituentBOM,
    FactIncomingQM,
    FactProcessMeasurements,
    FactWarrantyClaims,
    RefActionPlaybook,
    RefAIInsights,
    get_engine,
    init_database,
)

EXPECTED_TABLES = {
    "dim_supplier",
    "dim_material",
    "dim_component",
    "dim_lot",
    "dim_serial",
    "fact_incoming_qm",
    "fact_process_measurements",
    "fact_warranty_claims",
    "fact_constituent_bom",
    "agg_supplier_scorecard",
    "agg_coo_trends",
    "agg_coo_vs_supplier",
    "ref_ai_insights",
    "ref_action_playbook",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """In-memory SQLite engine — isolated per test module."""
    eng = get_engine("sqlite:///:memory:")
    init_database(eng)
    return eng


@pytest.fixture()
def session(engine):
    with Session(engine) as s:
        yield s
        s.rollback()


# ---------------------------------------------------------------------------
# Schema presence
# ---------------------------------------------------------------------------

def test_all_tables_created(engine):
    inspector = inspect(engine)
    created = set(inspector.get_table_names())
    assert EXPECTED_TABLES == created, (
        f"Missing: {EXPECTED_TABLES - created}  Extra: {created - EXPECTED_TABLES}"
    )


@pytest.mark.parametrize("table_name", sorted(EXPECTED_TABLES))
def test_table_has_primary_key(engine, table_name):
    inspector = inspect(engine)
    pk_cols = inspector.get_pk_constraint(table_name)["constrained_columns"]
    assert len(pk_cols) >= 1, f"{table_name} has no primary key"


# ---------------------------------------------------------------------------
# Column types
# ---------------------------------------------------------------------------

def test_fact_incoming_qm_boolean_columns(engine):
    inspector = inspect(engine)
    cols = {c["name"]: c for c in inspector.get_columns("fact_incoming_qm")}
    assert "is_fail" in cols
    # SQLite stores booleans as INTEGER; SQLAlchemy reflects type name
    assert cols["is_fail"]["type"].__class__.__name__ in ("BOOLEAN", "Boolean", "INTEGER")


def test_fact_process_measurements_boolean_columns(engine):
    inspector = inspect(engine)
    cols = {c["name"]: c for c in inspector.get_columns("fact_process_measurements")}
    assert "is_torque_fail" in cols
    assert "is_leak_fail" in cols


def test_agg_supplier_scorecard_float_columns(engine):
    inspector = inspect(engine)
    cols = {c["name"]: c for c in inspector.get_columns("agg_supplier_scorecard")}
    for col_name in ("incoming_fail_rate", "warranty_claim_rate", "quality_score"):
        assert col_name in cols, f"Missing float column: {col_name}"


# ---------------------------------------------------------------------------
# Foreign key integrity
# ---------------------------------------------------------------------------

def test_foreign_keys_declared(engine):
    inspector = inspect(engine)
    fk_map = {
        "dim_lot": ["dim_component", "dim_supplier"],
        "dim_serial": ["dim_material"],
        "fact_incoming_qm": ["dim_component", "dim_supplier", "dim_lot"],
        "fact_process_measurements": ["dim_serial", "dim_material"],
        "fact_warranty_claims": ["dim_serial"],
        "fact_constituent_bom": ["dim_serial", "dim_component", "dim_supplier", "dim_lot"],
        "agg_supplier_scorecard": ["dim_supplier"],
        "agg_coo_vs_supplier": ["dim_supplier"],
    }
    for table, expected_refs in fk_map.items():
        fks = inspector.get_foreign_keys(table)
        referred = {fk["referred_table"] for fk in fks}
        for ref in expected_refs:
            assert ref in referred, (
                f"{table} is missing FK → {ref}  (found: {referred})"
            )


# ---------------------------------------------------------------------------
# Round-trip insert/query
# ---------------------------------------------------------------------------

def test_insert_and_query_dim_supplier(session):
    sup = DimSupplier(
        supplier_name="Acme Corp",
        coo="DE",
        engineering_maturity="High",
        engineering_maturity_score=4.5,
        process_cpk=1.67,
    )
    session.add(sup)
    session.flush()

    fetched = session.get(DimSupplier, sup.supplier_id)
    assert fetched is not None
    assert fetched.supplier_name == "Acme Corp"
    assert fetched.process_cpk == pytest.approx(1.67)


def test_insert_fact_with_foreign_keys(session):
    supplier = DimSupplier(supplier_name="Beta Inc", coo="US")
    component = DimComponent(component_name="Valve Seat")
    lot = DimLot(lot_no="LOT-X1", supplier=supplier, component=component)
    session.add_all([supplier, component, lot])
    session.flush()

    row = FactIncomingQM(
        insp_lot="IL-001",
        supplier=supplier,
        component=component,
        lot=lot,
        result="FAIL",
        defect_code="D05",
        is_fail=True,
        characteristic="Diameter",
        measured_value=12.7,
        uom="mm",
    )
    session.add(row)
    session.flush()

    fetched = session.get(FactIncomingQM, row.id)
    assert fetched.is_fail is True
    assert fetched.supplier.supplier_name == "Beta Inc"
    assert fetched.lot.lot_no == "LOT-X1"


def test_is_fail_defaults_false(session):
    supplier = DimSupplier(supplier_name="Gamma Ltd", coo="JP")
    component = DimComponent(component_name="O-Ring")
    session.add_all([supplier, component])
    session.flush()

    row = FactIncomingQM(
        insp_lot="IL-002",
        supplier=supplier,
        component=component,
        result="PASS",
        characteristic="Weight",
        measured_value=5.1,
        uom="g",
    )
    session.add(row)
    session.flush()

    fetched = session.get(FactIncomingQM, row.id)
    assert fetched.is_fail is False


def test_init_database_is_idempotent(engine):
    """Calling init_database twice must not raise."""
    init_database(engine)   # second call — tables already exist


def test_ref_ai_insights_nullable_columns(session):
    insight = RefAIInsights(
        pattern_detected="High fail rate on Lot-42",
        risk_or_opportunity="Risk",
        # evidence and other fields intentionally omitted (nullable)
    )
    session.add(insight)
    session.flush()

    fetched = session.get(RefAIInsights, insight.id)
    assert fetched.evidence is None
    assert fetched.pattern_detected == "High fail rate on Lot-42"
