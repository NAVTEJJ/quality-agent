"""
Pydantic validators for API request/response bodies and ingestion boundary checks.

These are deliberately separate from the SQLAlchemy ORM in schema.py so the
two concerns (persistence mapping vs. data validation) stay decoupled.
"""
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class IncomingInspection(BaseModel):
    """Validated row from the Incoming_QM sheet."""

    inspection_id: str
    lot_id: str
    part_number: str
    supplier_id: str
    inspection_date: date
    result: str  # PASS | FAIL
    defect_code: Optional[str] = None


class WarrantyClaim(BaseModel):
    """Validated row from the Warranty_Claims sheet."""

    claim_id: str
    part_number: str
    failure_date: date
    region: Optional[str] = None
    defect_category: str


class TabInventoryEntry(BaseModel):
    """Single entry in the tab inventory produced by the profiler."""

    name: str
    row_count: int
    column_count: int
    columns: list[str]
    null_counts: dict[str, int] = Field(default_factory=dict)
    sample_values: dict[str, list] = Field(default_factory=dict)
