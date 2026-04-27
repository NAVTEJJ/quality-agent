"""
Tests for Phase 2 -- KPIs, anomaly detection, explanations, drill-down,
recommendations.

Module-scoped fixtures build **one** in-memory DB and **one** insights list
for the whole file, so the suite runs in under a second once the workbook
has been read.  Skipped automatically when the source workbook is absent.

Run with:  pytest tests/test_phase2.py -v
"""
import pandas as pd
import pytest

from app.ingestion.loader import load_all_sheets
from app.ingestion.normalizer import NormalizationPipeline
from app.models.schema import get_engine, init_database
from app.services.explainer import generate_all_insights
from app.services.service_registry import clear_registry_cache, get_registry
from configs import settings


# ---------------------------------------------------------------------------
# Fixtures — one DB + one registry + one insights run per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def registry():
    """In-memory registry built from the real workbook."""
    if not settings.EXCEL_PATH.exists():
        pytest.skip("Source workbook not present -- skipping Phase 2 tests")

    clear_registry_cache()
    sheets = load_all_sheets(settings.EXCEL_PATH)
    engine = get_engine("sqlite:///:memory:")
    init_database(engine)
    NormalizationPipeline().run_full_pipeline(sheets, engine)
    return get_registry(engine)


@pytest.fixture(scope="module")
def insights(registry):
    """Generate insights once per test module -- saves repeated pipeline runs."""
    return generate_all_insights(registry.engine)


# ---------------------------------------------------------------------------
# 1. Lot risk -- L-778 is the marquee HIGH-risk lot
# ---------------------------------------------------------------------------

def test_lot_risk_scores_computed(registry):
    df = registry.kpi.get_lot_risk_scores()
    assert not df.empty
    l778 = df[df["lot_no"] == "L-778"]
    assert not l778.empty, "L-778 missing from lot_risk_scores"
    assert l778["risk_tier"].iloc[0] == "HIGH", (
        f"L-778 must be HIGH risk, got {l778['risk_tier'].iloc[0]}"
    )


# ---------------------------------------------------------------------------
# 2. Process drift -- LINE-2 Night is the marquee drift signal
# ---------------------------------------------------------------------------

def test_line2_night_highest_drift(registry):
    df = registry.kpi.get_process_drift_by_line_shift()
    top = df.iloc[0]
    assert top["line"] == "LINE-2" and top["shift"] == "Night", (
        f"Expected LINE-2 Night as top drift, got {top['line']} {top['shift']}"
    )
    assert top["torque_fail_rate"] > 0.1


# ---------------------------------------------------------------------------
# 3. Insight explanations are structurally complete
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = [
    "insight_type", "entity_id", "entity_name", "headline",
    "why", "evidence", "likely_cause", "recommended_actions",
    "confidence", "risk_level", "drill_down_hints", "sap_touchpoints",
]


def test_insight_explanation_complete(insights):
    assert len(insights) > 0, "No insights generated"
    for ins in insights:
        d = ins.to_dict()
        for field in _REQUIRED_FIELDS:
            assert field in d, f"{ins.entity_name}: missing field '{field}'"
        assert ins.headline, f"{ins.entity_name}: empty headline"
        assert len(ins.why) >= 3, (
            f"{ins.entity_name}: 'why' must have 3+ bullets (got {len(ins.why)})"
        )
        assert isinstance(ins.evidence, dict) and ins.evidence, (
            f"{ins.entity_name}: empty evidence"
        )
        assert len(ins.recommended_actions) >= 1, (
            f"{ins.entity_name}: no recommended_actions"
        )
        assert 0.0 <= ins.confidence <= 1.0, (
            f"{ins.entity_name}: confidence out of range ({ins.confidence})"
        )


# ---------------------------------------------------------------------------
# 4. Drill-down chain for L-778 has all branches populated
# ---------------------------------------------------------------------------

def test_drill_down_l778_complete(registry):
    chain = registry.drill_down.get_full_drill_down_chain("L-778")

    assert chain["lot_info"] is not None
    assert chain["lot_info"]["lot_no"] == "L-778"

    assert len(chain["inspection_records"]) > 0, "no inspection records"
    assert len(chain["affected_serials"]) > 0, "no affected serials"

    assert chain["supplier_scorecard"] is not None, "supplier_scorecard missing"
    assert chain["coo_context"] is not None, "coo_context missing"

    # Summary matches record counts.
    assert chain["summary"]["total_inspections"] == len(chain["inspection_records"])
    assert chain["summary"]["affected_serials"] == len(chain["affected_serials"])


# ---------------------------------------------------------------------------
# 5. Premium suppliers -- SUP-A or SUP-B must qualify
# ---------------------------------------------------------------------------

def test_premium_suppliers_identified(registry):
    df = registry.kpi.get_premium_suppliers()
    assert len(df) >= 1, "no premium suppliers identified"
    names = set(df["supplier"].astype(str))
    assert names & {"SUP-A", "SUP-B"}, (
        f"Expected SUP-A or SUP-B in premium suppliers, got {names}"
    )


# ---------------------------------------------------------------------------
# 6. Recommendation engine returns 4+ actions for HIGH-risk Watchlist lot
# ---------------------------------------------------------------------------

def test_recommendation_actions_non_empty(registry):
    rec = registry.recommendations.get_actions_for_lot_risk(
        "L-778", risk_score=0.655, supplier_tier="Watchlist"
    )
    assert rec["urgency"] == "IMMEDIATE"
    assert len(rec["actions"]) >= 4, (
        f"Expected >=4 actions for HIGH-risk Watchlist lot, got {len(rec['actions'])}"
    )
    assert rec["sap_touchpoints"], "SAP touchpoints missing"


# ---------------------------------------------------------------------------
# 7. COO decomposition nuance -- beats_coo_avg populated
# ---------------------------------------------------------------------------

def test_coo_decomposition_nuance(registry):
    df = registry.kpi.get_coo_vs_supplier_decomposition()
    assert "beats_coo_avg" in df.columns
    assert df["beats_coo_avg"].notna().any(), "beats_coo_avg is entirely null"
    # Must resolve to Yes/No strings, not NaN or floats.
    values = set(df["beats_coo_avg"].dropna().unique())
    assert values <= {"Yes", "No"}, f"Unexpected beats_coo_avg values: {values}"


# ---------------------------------------------------------------------------
# 8. Every insight must carry at least one SAP / MES touchpoint
# ---------------------------------------------------------------------------

def test_all_insights_have_sap_touchpoints(insights):
    for ins in insights:
        assert ins.sap_touchpoints, (
            f"Insight '{ins.entity_name}' ({ins.insight_type}) has empty "
            f"sap_touchpoints"
        )
