"""
SQLAlchemy ORM models and database utilities for the Quality Agent.

Star-schema layout
──────────────────
  dim_*  – dimension tables (suppliers, materials, components, lots, serials)
  fact_* – grain-level transactional tables (inspections, process, warranty, BOM)
  agg_*  – pre-aggregated analytical tables (scorecard, COO trends, COO vs supplier)
  ref_*  – reference / metadata tables (AI insights, action playbook)

Usage
─────
  from app.models.schema import get_engine, init_database
  engine = get_engine()
  init_database(engine)
"""

import logging
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from configs import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Dimension tables
# ---------------------------------------------------------------------------

class DimSupplier(Base):
    """One row per unique supplier.  Natural key is supplier_name + coo."""

    __tablename__ = "dim_supplier"

    supplier_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    supplier_name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True, index=True)
    coo: Mapped[str | None] = mapped_column(String(100))
    engineering_maturity: Mapped[str | None] = mapped_column(String(50))
    engineering_maturity_score: Mapped[float | None] = mapped_column(Float)
    process_cpk: Mapped[float | None] = mapped_column(Float)
    design_ownership: Mapped[str | None] = mapped_column(String(100))
    typical_project_type: Mapped[str | None] = mapped_column(String(100))

    # Relationships (populated during ETL)
    lots: Mapped[list["DimLot"]] = relationship("DimLot", back_populates="supplier")
    incoming_inspections: Mapped[list["FactIncomingQM"]] = relationship(
        "FactIncomingQM", back_populates="supplier"
    )
    bom_entries: Mapped[list["FactConstituentBOM"]] = relationship(
        "FactConstituentBOM", back_populates="supplier"
    )
    scorecard: Mapped["AggSupplierScorecard | None"] = relationship(
        "AggSupplierScorecard", back_populates="supplier", uselist=False
    )
    coo_vs_supplier: Mapped[list["AggCooVsSupplier"]] = relationship(
        "AggCooVsSupplier", back_populates="supplier"
    )


class DimMaterial(Base):
    """Finished-goods material master."""

    __tablename__ = "dim_material"

    material_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    material_name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True, index=True)

    serials: Mapped[list["DimSerial"]] = relationship("DimSerial", back_populates="finished_material")
    process_measurements: Mapped[list["FactProcessMeasurements"]] = relationship(
        "FactProcessMeasurements", back_populates="finished_material"
    )


class DimComponent(Base):
    """Component / sub-assembly master."""

    __tablename__ = "dim_component"

    component_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    component_name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True, index=True)

    lots: Mapped[list["DimLot"]] = relationship("DimLot", back_populates="component")
    incoming_inspections: Mapped[list["FactIncomingQM"]] = relationship(
        "FactIncomingQM", back_populates="component"
    )
    bom_entries: Mapped[list["FactConstituentBOM"]] = relationship(
        "FactConstituentBOM", back_populates="component"
    )


class DimLot(Base):
    """Supplier production lot — links a component, supplier, and manufacture date."""

    __tablename__ = "dim_lot"

    lot_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lot_no: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    component_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dim_component.component_id"))
    supplier_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dim_supplier.supplier_id"))
    mfg_date: Mapped[datetime | None] = mapped_column(DateTime)

    component: Mapped["DimComponent | None"] = relationship("DimComponent", back_populates="lots")
    supplier: Mapped["DimSupplier | None"] = relationship("DimSupplier", back_populates="lots")
    incoming_inspections: Mapped[list["FactIncomingQM"]] = relationship(
        "FactIncomingQM", back_populates="lot"
    )
    bom_entries: Mapped[list["FactConstituentBOM"]] = relationship(
        "FactConstituentBOM", back_populates="lot"
    )


class DimSerial(Base):
    """Finished-goods serial number — the primary traceability anchor."""

    __tablename__ = "dim_serial"

    serial_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    serial_no: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    finished_material_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("dim_material.material_id")
    )
    plant: Mapped[str | None] = mapped_column(String(100))
    line: Mapped[str | None] = mapped_column(String(100))
    build_dt: Mapped[datetime | None] = mapped_column(DateTime)
    shift: Mapped[str | None] = mapped_column(String(50))
    operator_id: Mapped[str | None] = mapped_column(String(100))
    vendor_of_critical_assy: Mapped[str | None] = mapped_column(String(200))
    coo_critical: Mapped[str | None] = mapped_column(String(100))
    ecn_level: Mapped[str | None] = mapped_column(String(100))

    finished_material: Mapped["DimMaterial | None"] = relationship(
        "DimMaterial", back_populates="serials"
    )
    process_measurements: Mapped[list["FactProcessMeasurements"]] = relationship(
        "FactProcessMeasurements", back_populates="serial"
    )
    warranty_claims: Mapped[list["FactWarrantyClaims"]] = relationship(
        "FactWarrantyClaims", back_populates="serial"
    )
    bom_entries: Mapped[list["FactConstituentBOM"]] = relationship(
        "FactConstituentBOM", back_populates="serial"
    )


# ---------------------------------------------------------------------------
# Fact tables
# ---------------------------------------------------------------------------

class FactIncomingQM(Base):
    """Grain: one row per inspection characteristic measurement."""

    __tablename__ = "fact_incoming_qm"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    insp_lot: Mapped[str | None] = mapped_column(String(100), index=True)
    component_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dim_component.component_id"))
    supplier_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dim_supplier.supplier_id"))
    lot_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dim_lot.lot_id"))
    insp_date: Mapped[datetime | None] = mapped_column(DateTime)
    characteristic: Mapped[str | None] = mapped_column(String(200))
    measured_value: Mapped[float | None] = mapped_column(Float)
    uom: Mapped[str | None] = mapped_column(String(50))
    result: Mapped[str | None] = mapped_column(String(10))          # "PASS" | "FAIL"
    defect_code: Mapped[str | None] = mapped_column(String(50))     # NULL when result == PASS
    is_fail: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    component: Mapped["DimComponent | None"] = relationship("DimComponent", back_populates="incoming_inspections")
    supplier: Mapped["DimSupplier | None"] = relationship("DimSupplier", back_populates="incoming_inspections")
    lot: Mapped["DimLot | None"] = relationship("DimLot", back_populates="incoming_inspections")


class FactProcessMeasurements(Base):
    """Grain: one row per serial number — torque and leak test results."""

    __tablename__ = "fact_process_measurements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    serial_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dim_serial.serial_id"))
    finished_material_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("dim_material.material_id")
    )
    build_date: Mapped[datetime | None] = mapped_column(DateTime)
    line: Mapped[str | None] = mapped_column(String(100))
    shift: Mapped[str | None] = mapped_column(String(50))
    torque_nm: Mapped[float | None] = mapped_column(Float)
    torque_result: Mapped[str | None] = mapped_column(String(10))   # "PASS" | "FAIL"
    leak_rate_ccm: Mapped[float | None] = mapped_column(Float)
    leak_result: Mapped[str | None] = mapped_column(String(10))     # "PASS" | "FAIL"
    ecn_level: Mapped[str | None] = mapped_column(String(100))
    is_torque_fail: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_leak_fail: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    serial: Mapped["DimSerial | None"] = relationship("DimSerial", back_populates="process_measurements")
    finished_material: Mapped["DimMaterial | None"] = relationship(
        "DimMaterial", back_populates="process_measurements"
    )


class FactWarrantyClaims(Base):
    """Grain: one row per field warranty claim."""

    __tablename__ = "fact_warranty_claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[str | None] = mapped_column(String(100), unique=True, index=True)
    serial_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dim_serial.serial_id"))
    failure_date: Mapped[datetime | None] = mapped_column(DateTime)
    symptom: Mapped[str | None] = mapped_column(String(500))
    mileage_or_hours: Mapped[float | None] = mapped_column(Float)
    region: Mapped[str | None] = mapped_column(String(100))         # nullable — 2 known gaps
    severity: Mapped[str | None] = mapped_column(String(50))

    serial: Mapped["DimSerial | None"] = relationship("DimSerial", back_populates="warranty_claims")


class FactConstituentBOM(Base):
    """Grain: one component row per finished serial — as-built bill of materials."""

    __tablename__ = "fact_constituent_bom"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    serial_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dim_serial.serial_id"))
    component_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dim_component.component_id"))
    supplier_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dim_supplier.supplier_id"))
    lot_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dim_lot.lot_id"))
    comp_serial: Mapped[str | None] = mapped_column(String(100))
    coo: Mapped[str | None] = mapped_column(String(100))
    mfg_date: Mapped[datetime | None] = mapped_column(DateTime)
    cert_doc_id: Mapped[str | None] = mapped_column(String(200))

    serial: Mapped["DimSerial | None"] = relationship("DimSerial", back_populates="bom_entries")
    component: Mapped["DimComponent | None"] = relationship("DimComponent", back_populates="bom_entries")
    supplier: Mapped["DimSupplier | None"] = relationship("DimSupplier", back_populates="bom_entries")
    lot: Mapped["DimLot | None"] = relationship("DimLot", back_populates="bom_entries")


# ---------------------------------------------------------------------------
# Aggregate tables
# ---------------------------------------------------------------------------

class AggSupplierScorecard(Base):
    """Pre-computed quality + delivery scorecard per supplier."""

    __tablename__ = "agg_supplier_scorecard"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    supplier_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_supplier.supplier_id"), nullable=False, unique=True
    )
    lots_inspected: Mapped[int | None] = mapped_column(Integer)
    samples: Mapped[int | None] = mapped_column(Integer)
    fails: Mapped[int | None] = mapped_column(Integer)
    incoming_fail_rate: Mapped[float | None] = mapped_column(Float)
    units_built: Mapped[int | None] = mapped_column(Integer)
    units_with_claims: Mapped[int | None] = mapped_column(Integer)
    warranty_claim_rate: Mapped[float | None] = mapped_column(Float)
    process_drift_index: Mapped[float | None] = mapped_column(Float)
    on_time_delivery_pct: Mapped[float | None] = mapped_column(Float)
    avg_lead_time_days: Mapped[float | None] = mapped_column(Float)
    quality_score: Mapped[float | None] = mapped_column(Float)
    tier: Mapped[str | None] = mapped_column(String(50))
    premium_service_fit: Mapped[str | None] = mapped_column(String(10))   # "Yes" | "No"

    supplier: Mapped["DimSupplier"] = relationship("DimSupplier", back_populates="scorecard")


class AggCooTrends(Base):
    """Aggregated incoming-QM and warranty rates per country of origin."""

    __tablename__ = "agg_coo_trends"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    coo: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    samples: Mapped[int | None] = mapped_column(Integer)
    fails: Mapped[int | None] = mapped_column(Integer)
    coo_incoming_fail_rate: Mapped[float | None] = mapped_column(Float)
    coo_warranty_claim_rate: Mapped[float | None] = mapped_column(Float)


class AggCooVsSupplier(Base):
    """Supplier performance benchmarked against its COO average."""

    __tablename__ = "agg_coo_vs_supplier"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    supplier_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dim_supplier.supplier_id"))
    coo: Mapped[str | None] = mapped_column(String(100))
    incoming_fail_rate: Mapped[float | None] = mapped_column(Float)
    warranty_claim_rate: Mapped[float | None] = mapped_column(Float)
    quality_score: Mapped[float | None] = mapped_column(Float)
    tier: Mapped[str | None] = mapped_column(String(50))
    coo_incoming_fail_rate: Mapped[float | None] = mapped_column(Float)
    coo_warranty_claim_rate: Mapped[float | None] = mapped_column(Float)
    beats_coo_avg: Mapped[str | None] = mapped_column(String(10))    # "Yes" | "No"

    supplier: Mapped["DimSupplier | None"] = relationship("DimSupplier", back_populates="coo_vs_supplier")


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

class RefAIInsights(Base):
    """AI-generated pattern detections and guidance stored verbatim from source."""

    __tablename__ = "ref_ai_insights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pattern_detected: Mapped[str | None] = mapped_column(String(500))
    evidence: Mapped[str | None] = mapped_column(String(1000))
    risk_or_opportunity: Mapped[str | None] = mapped_column(String(50))
    ai_guidance: Mapped[str | None] = mapped_column(String(1000))
    suggested_actionables: Mapped[str | None] = mapped_column(String(2000))


class RefActionPlaybook(Base):
    """Prescribed remediation actions mapped to insight types and ERP touchpoints."""

    __tablename__ = "ref_action_playbook"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    insight_type: Mapped[str | None] = mapped_column(String(200))
    typical_action: Mapped[str | None] = mapped_column(String(500))
    where_it_fits: Mapped[str | None] = mapped_column(String(500))
    sap_mes_touchpoint: Mapped[str | None] = mapped_column(String(500))


# ---------------------------------------------------------------------------
# Database utilities
# ---------------------------------------------------------------------------

def get_engine(database_url: str = settings.DATABASE_URL) -> Engine:
    """Return a SQLAlchemy engine bound to *database_url*.

    Args:
        database_url: SQLAlchemy connection string.  Defaults to
            ``settings.DATABASE_URL`` (SQLite, path inside ``data/processed/``).

    Returns:
        Configured :class:`~sqlalchemy.engine.Engine` instance.
    """
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},   # required for SQLite + threading
        echo=False,
    )
    logger.debug("Engine created for: %s", database_url)
    return engine


def init_database(engine: Engine) -> None:
    """Create all tables defined in this module if they do not already exist.

    Safe to call repeatedly — uses ``CREATE TABLE IF NOT EXISTS`` semantics
    via ``checkfirst=True`` (the SQLAlchemy default for ``create_all``).

    Args:
        engine: A connected :class:`~sqlalchemy.engine.Engine`.
    """
    logger.info("Initialising database schema …")
    Base.metadata.create_all(engine)
    table_names = list(Base.metadata.tables.keys())
    logger.info(
        "Schema ready — %d table(s): %s",
        len(table_names),
        sorted(table_names),
    )
