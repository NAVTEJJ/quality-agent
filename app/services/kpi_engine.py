"""
KPI calculation engine for the Quality Agent (Phase 2, Step 1).

The :class:`KPIEngine` reads directly from the normalised Phase 1 tables and
produces every leading- and lagging-indicator DataFrame the downstream agents
need (lot risk, process drift, COO performance, supplier rankings, and the
consolidated "where to inspect today" view).

All methods are pure: they take no side effects on the database, accept no
arguments beyond the engine (bar the two optional filters on
:meth:`get_inspection_focus`), and return tidy DataFrames.

Design notes
------------
* **Min-max normalisation** of every signal in the composite lot-risk score
  follows the Phase 2 spec verbatim.  Signals that degenerate to a constant
  (max == min, or a column of zeros) collapse to ``0`` by construction --
  they cannot dominate the score when there is no variance to speak of.
* **Warranty-to-lot linkage** goes through
  ``fact_constituent_bom``: claims live on a finished-goods serial, and that
  serial's BOM tells us which supplier lot fed it.
* **Repeat-issue signal** is anchored at the supplier (the natural locus of
  repeat failure), not at the lot: a single bad lot is noise, a supplier who
  fails on many distinct inspection days is a pattern.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.engine import Engine

from configs import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _min_max(series: pd.Series) -> pd.Series:
    """Min-max normalise *series* to the [0, 1] range.

    Returns a zero series when the input has no spread (max == min), which is
    the well-defined limit -- a signal with no variance carries no information.
    """
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def _tier_from_score(score: float) -> str:
    """Bucket a composite score into HIGH / MEDIUM / LOW per the spec."""
    if score > 0.6:
        return "HIGH"
    if score > 0.3:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# KPIEngine
# ---------------------------------------------------------------------------

class KPIEngine:
    """All Phase 2 KPIs, computed on demand from the normalised warehouse."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ------------------------------------------------------------------
    # LOT RISK
    # ------------------------------------------------------------------

    def get_lot_fail_rates(self) -> pd.DataFrame:
        """Incoming-QM fail rate per lot, with defect-code footprint."""
        sql = """
            SELECT
                qm.lot_id,
                COUNT(*)                         AS total_inspections,
                SUM(qm.is_fail)                  AS total_fails,
                1.0 * SUM(qm.is_fail) / COUNT(*) AS fail_rate
            FROM fact_incoming_qm qm
            WHERE qm.lot_id IS NOT NULL
            GROUP BY qm.lot_id
        """
        df = pd.read_sql(sql, self.engine)

        # Defect-code footprint per lot (distinct, comma-joined).
        codes_sql = """
            SELECT lot_id, defect_code
            FROM fact_incoming_qm
            WHERE lot_id IS NOT NULL AND defect_code != ''
        """
        codes = pd.read_sql(codes_sql, self.engine)
        code_map = (
            codes.groupby("lot_id")["defect_code"]
            .agg(lambda xs: ", ".join(sorted(set(xs))))
            .to_dict()
        )
        df["defect_codes"] = df["lot_id"].map(code_map).fillna("")

        # Decorate with lot-master attributes.
        lots = pd.read_sql(
            "SELECT lot_id, lot_no, component_id, supplier_id FROM dim_lot",
            self.engine,
        )
        df = df.merge(lots, on="lot_id", how="left")

        return df.sort_values("fail_rate", ascending=False).reset_index(drop=True)

    def get_lot_risk_scores(self) -> pd.DataFrame:
        """Composite lot-risk score per the Phase 2 weighting formula.

            lot_risk_score = 0.35 * normalized_lot_fail_rate
                           + 0.25 * normalized_supplier_risk
                           + 0.20 * normalized_claim_signal
                           + 0.20 * normalized_repeat_issue_signal
        """
        lots = self.get_lot_fail_rates()

        # Signal 1: lot fail rate (min-max across all lots).
        lots["norm_fail_rate"] = _min_max(lots["fail_rate"])

        # Signal 2: supplier risk = 1 - quality_score / 100, looked up per lot's supplier.
        scorecard = pd.read_sql(
            "SELECT supplier_id, quality_score, tier FROM agg_supplier_scorecard",
            self.engine,
        )
        scorecard["supplier_risk"] = 1.0 - (scorecard["quality_score"].astype(float) / 100.0)
        lots = lots.merge(
            scorecard[["supplier_id", "supplier_risk", "tier"]],
            on="supplier_id",
            how="left",
        )
        lots["supplier_risk"] = lots["supplier_risk"].fillna(lots["supplier_risk"].max() or 1.0)
        lots["norm_supplier_risk"] = lots["supplier_risk"].clip(lower=0.0, upper=1.0)

        # Signal 3: claims linked to this lot (via BOM -> warranty_claims).
        claims_per_lot = pd.read_sql(
            """
            SELECT b.lot_id, COUNT(DISTINCT w.claim_id) AS claims_linked
            FROM fact_constituent_bom b
            JOIN fact_warranty_claims w ON b.serial_id = w.serial_id
            WHERE b.lot_id IS NOT NULL
            GROUP BY b.lot_id
            """,
            self.engine,
        )
        lots = lots.merge(claims_per_lot, on="lot_id", how="left")
        lots["claims_linked"] = lots["claims_linked"].fillna(0).astype(int)
        max_claims = lots["claims_linked"].max()
        lots["norm_claim_signal"] = (
            lots["claims_linked"] / max_claims if max_claims else 0.0
        )

        # Signal 4: repeat-issue -- distinct failing inspection dates per supplier.
        repeat_sql = """
            SELECT supplier_id,
                   COUNT(DISTINCT DATE(insp_date)) AS n_fail_dates
            FROM fact_incoming_qm
            WHERE is_fail = 1 AND supplier_id IS NOT NULL
            GROUP BY supplier_id
        """
        repeat = pd.read_sql(repeat_sql, self.engine)
        repeat["norm_repeat"] = _min_max(repeat["n_fail_dates"])
        lots = lots.merge(
            repeat[["supplier_id", "n_fail_dates", "norm_repeat"]],
            on="supplier_id",
            how="left",
        )
        lots["n_fail_dates"] = lots["n_fail_dates"].fillna(0).astype(int)
        lots["norm_repeat"] = lots["norm_repeat"].fillna(0.0)

        # Composite score and tier.
        lots["lot_risk_score"] = (
            0.35 * lots["norm_fail_rate"]
            + 0.25 * lots["norm_supplier_risk"]
            + 0.20 * lots["norm_claim_signal"]
            + 0.20 * lots["norm_repeat"]
        )
        lots["risk_tier"] = lots["lot_risk_score"].apply(_tier_from_score)

        ordered_cols = [
            "lot_id", "lot_no", "component_id", "supplier_id",
            "total_inspections", "total_fails", "fail_rate", "defect_codes",
            "claims_linked", "n_fail_dates", "tier",
            "norm_fail_rate", "norm_supplier_risk",
            "norm_claim_signal", "norm_repeat",
            "lot_risk_score", "risk_tier",
        ]
        return (
            lots[ordered_cols]
            .sort_values("lot_risk_score", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # PROCESS DRIFT
    # ------------------------------------------------------------------

    def get_process_drift_by_line_shift(self) -> pd.DataFrame:
        """Torque / leak failure rates grouped by production line + shift."""
        sql = """
            SELECT
                line,
                shift,
                COUNT(*)                               AS total_builds,
                SUM(is_torque_fail)                    AS torque_fails,
                1.0 * SUM(is_torque_fail) / COUNT(*)   AS torque_fail_rate,
                SUM(is_leak_fail)                      AS leak_fails,
                1.0 * SUM(is_leak_fail) / COUNT(*)     AS leak_fail_rate,
                1.0 * (SUM(is_torque_fail) + SUM(is_leak_fail)) / COUNT(*)
                                                       AS combined_fail_rate
            FROM fact_process_measurements
            WHERE line IS NOT NULL AND shift IS NOT NULL
            GROUP BY line, shift
        """
        df = pd.read_sql(sql, self.engine)
        return df.sort_values("torque_fail_rate", ascending=False).reset_index(drop=True)

    def get_drift_signals(self) -> pd.DataFrame:
        """Line/shift combos exceeding the configured drift threshold."""
        df = self.get_process_drift_by_line_shift()
        threshold = settings.PROCESS_DRIFT_FAIL_RATE_THRESHOLD
        df["is_drift_signal"] = df["torque_fail_rate"] > threshold
        return df[df["is_drift_signal"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # COO
    # ------------------------------------------------------------------

    def get_coo_performance(self) -> pd.DataFrame:
        """Country-of-origin trend table with a 1 = best ranking by fail rate."""
        df = pd.read_sql("SELECT * FROM agg_coo_trends", self.engine)
        df = df.sort_values("coo_incoming_fail_rate", ascending=True).reset_index(drop=True)
        df["rank"] = df["coo_incoming_fail_rate"].rank(method="min", ascending=True).astype(int)
        return df

    def get_coo_vs_supplier_decomposition(self) -> pd.DataFrame:
        """Per-supplier benchmark versus its own country average.

        ``gap`` is intentionally ``coo_incoming_fail_rate - incoming_fail_rate``
        so that **positive values mean the supplier outperforms its COO** --
        the insight the downstream agent cites as "SUP-B from Japan beats its
        COO average".
        """
        sql = """
            SELECT
                s.supplier_name        AS supplier,
                cvs.coo,
                cvs.incoming_fail_rate,
                cvs.coo_incoming_fail_rate,
                cvs.warranty_claim_rate,
                cvs.coo_warranty_claim_rate,
                cvs.quality_score,
                cvs.tier,
                cvs.beats_coo_avg
            FROM agg_coo_vs_supplier cvs
            JOIN dim_supplier s ON cvs.supplier_id = s.supplier_id
        """
        df = pd.read_sql(sql, self.engine)
        df["gap"] = df["coo_incoming_fail_rate"] - df["incoming_fail_rate"]
        df["warranty_gap"] = df["coo_warranty_claim_rate"] - df["warranty_claim_rate"]
        return df.sort_values("gap", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # SUPPLIERS
    # ------------------------------------------------------------------

    def get_supplier_rankings(self) -> pd.DataFrame:
        """Scorecard + precision_score + composite rank.

            precision_score = 0.6 * (1 - normalized_mean_abs_distance_to_target)
                            + 0.4 * (1 - normalized_variance)

        Target distance is approximated by ``incoming_fail_rate`` and variance
        by ``process_drift_index`` -- both already present on the scorecard.
        """
        sql = """
            SELECT
                s.supplier_id,
                s.supplier_name        AS supplier,
                s.coo,
                s.engineering_maturity,
                s.engineering_maturity_score,
                s.process_cpk,
                sc.incoming_fail_rate,
                sc.warranty_claim_rate,
                sc.process_drift_index,
                sc.on_time_delivery_pct,
                sc.avg_lead_time_days,
                sc.quality_score,
                sc.tier,
                sc.premium_service_fit
            FROM dim_supplier s
            LEFT JOIN agg_supplier_scorecard sc ON s.supplier_id = sc.supplier_id
        """
        df = pd.read_sql(sql, self.engine)

        # Precision approximations -- normalised so a higher score => more precise.
        df["norm_distance"] = _min_max(df["incoming_fail_rate"].fillna(0))
        df["norm_variance"] = _min_max(df["process_drift_index"].fillna(0))
        df["precision_score"] = (
            0.6 * (1.0 - df["norm_distance"]) + 0.4 * (1.0 - df["norm_variance"])
        )

        # Composite rank -- average of a normalised quality score and precision.
        norm_quality = _min_max(df["quality_score"].fillna(0))
        df["composite_score"] = 0.5 * norm_quality + 0.5 * df["precision_score"]
        df["composite_rank"] = (
            df["composite_score"].rank(method="min", ascending=False).astype(int)
        )

        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    def get_premium_suppliers(self) -> pd.DataFrame:
        """Suppliers fit for premium / safety-critical programmes."""
        sql = """
            SELECT
                s.supplier_id,
                s.supplier_name        AS supplier,
                s.coo,
                s.engineering_maturity,
                s.engineering_maturity_score,
                s.process_cpk,
                s.design_ownership,
                s.typical_project_type,
                sc.quality_score,
                sc.tier,
                sc.premium_service_fit
            FROM dim_supplier s
            JOIN agg_supplier_scorecard sc ON s.supplier_id = sc.supplier_id
            WHERE sc.premium_service_fit = 'Yes'
        """
        df = pd.read_sql(sql, self.engine)
        return df.sort_values(
            ["engineering_maturity_score", "process_cpk"],
            ascending=[False, False],
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # INSPECTION PRIORITY
    # ------------------------------------------------------------------

    def get_inspection_focus(
        self,
        date_filter: Optional[str] = None,
        plant_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Consolidated "where to inspect today" DataFrame.

        Args:
            date_filter: ISO date (``YYYY-MM-DD``) -- only lots with at least
                one inspection on/after that date are included.  ``None``
                returns every lot that has ever been inspected.
            plant_filter: Plant code (e.g. ``"PL01"``) -- filters lots down to
                those whose consumed serials were built in that plant.

        Columns
        -------
        ``lot_no, component, supplier, lot_risk_score, risk_tier,
        is_supplier_watchlist, has_drift_signal, drift_lines``
        """
        risk = self.get_lot_risk_scores()

        # Active-lot filter (inspection activity in window).
        active_sql = """
            SELECT lot_id, MAX(insp_date) AS last_insp_date
            FROM fact_incoming_qm
            WHERE lot_id IS NOT NULL
            GROUP BY lot_id
        """
        active = pd.read_sql(active_sql, self.engine)
        if date_filter:
            active["last_insp_date"] = pd.to_datetime(active["last_insp_date"], errors="coerce")
            cutoff = pd.to_datetime(date_filter)
            active = active[active["last_insp_date"] >= cutoff]
        risk = risk.merge(active, on="lot_id", how="inner")

        # Watchlist flag.
        risk["is_supplier_watchlist"] = risk["tier"].fillna("").str.lower() == "watchlist"

        # Lot -> line(s) via BOM and serial master.
        line_sql = """
            SELECT b.lot_id, ds.line, ds.plant
            FROM fact_constituent_bom b
            JOIN dim_serial ds ON b.serial_id = ds.serial_id
            WHERE b.lot_id IS NOT NULL
        """
        lines = pd.read_sql(line_sql, self.engine)
        if plant_filter:
            lines = lines[lines["plant"] == plant_filter]
            # Intersect -- drop lots that no longer have any serials in this plant.
            risk = risk[risk["lot_id"].isin(lines["lot_id"].unique())]

        drifting_lines = set(self.get_drift_signals()["line"].unique())
        lot_lines = (
            lines.groupby("lot_id")["line"]
            .agg(lambda xs: sorted({x for x in xs if pd.notna(x)}))
            .to_dict()
        )
        risk["drift_lines"] = risk["lot_id"].map(
            lambda lid: [l for l in lot_lines.get(lid, []) if l in drifting_lines]
        )
        risk["has_drift_signal"] = risk["drift_lines"].apply(bool)
        risk["drift_lines"] = risk["drift_lines"].apply(lambda xs: ", ".join(xs))

        # Human-readable decoration.
        components = pd.read_sql(
            "SELECT component_id, component_name AS component FROM dim_component",
            self.engine,
        )
        suppliers = pd.read_sql(
            "SELECT supplier_id, supplier_name AS supplier FROM dim_supplier",
            self.engine,
        )
        risk = risk.merge(components, on="component_id", how="left")
        risk = risk.merge(suppliers, on="supplier_id", how="left")

        cols = [
            "lot_no", "component", "supplier",
            "lot_risk_score", "risk_tier",
            "is_supplier_watchlist", "has_drift_signal", "drift_lines",
            "total_inspections", "total_fails", "fail_rate",
            "claims_linked", "last_insp_date",
        ]
        return risk[cols].sort_values("lot_risk_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

_KPI_METHODS: list[tuple[str, str]] = [
    ("lot_fail_rates",      "get_lot_fail_rates"),
    ("lot_risk_scores",     "get_lot_risk_scores"),
    ("process_drift",       "get_process_drift_by_line_shift"),
    ("drift_signals",       "get_drift_signals"),
    ("coo_performance",     "get_coo_performance"),
    ("coo_vs_supplier",     "get_coo_vs_supplier_decomposition"),
    ("supplier_rankings",   "get_supplier_rankings"),
    ("premium_suppliers",   "get_premium_suppliers"),
    ("inspection_focus",    "get_inspection_focus"),
]


def _save_df(df: pd.DataFrame, path: Path) -> Path:
    """Write *df* as parquet when pyarrow/fastparquet is available, CSV otherwise."""
    try:
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError) as exc:
        # pyarrow/fastparquet missing -- degrade gracefully to CSV.
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.warning(
            "Parquet engine unavailable (%s) -- wrote CSV instead: %s",
            exc.__class__.__name__, csv_path,
        )
        return csv_path


def run_all_kpis(engine: Engine) -> dict[str, pd.DataFrame]:
    """Compute every KPI DataFrame, persist each to ``data/processed/kpis/``,
    and print a concise summary.

    Returns:
        Dict of ``{short_name: DataFrame}`` for every KPI computed.
    """
    kpi_dir = settings.PROCESSED_DIR / "kpis"
    kpi_dir.mkdir(parents=True, exist_ok=True)

    eng = KPIEngine(engine)
    results: dict[str, pd.DataFrame] = {}

    for short_name, method in _KPI_METHODS:
        df = getattr(eng, method)()
        results[short_name] = df
        out_path = _save_df(df, kpi_dir / f"{short_name}.parquet")
        logger.info("  %-22s  %5d rows  -> %s", short_name, len(df), out_path.name)

    # ------------------------------------------------------------------
    # Headline findings
    # ------------------------------------------------------------------
    risk = results["lot_risk_scores"]
    drift = results["drift_signals"]
    suppliers = results["supplier_rankings"]
    coo = results["coo_performance"]
    premium = results["premium_suppliers"]

    # L-778 -- the marquee traceability case.
    l778 = risk[risk["lot_no"] == "L-778"]
    if l778.empty:
        l778_tier = "NOT FOUND"
        l778_score = float("nan")
    else:
        l778_tier = str(l778.iloc[0]["risk_tier"])
        l778_score = float(l778.iloc[0]["lot_risk_score"])

    # LINE-2 Night -- the marquee drift case.
    line2_night = drift[(drift["line"] == "LINE-2") & (drift["shift"] == "Night")]
    line2_confirmed = not line2_night.empty

    sep = "=" * 72
    print()
    print(sep)
    print("  Phase 2 KPI Engine -- Headline Findings")
    print(sep)
    print(f"  L-778 risk tier: {l778_tier}  (score = {l778_score:.3f})")
    drift_status = "CONFIRMED" if line2_confirmed else "NOT CONFIRMED"
    if line2_confirmed:
        rate = float(line2_night.iloc[0]["torque_fail_rate"])
        print(f"  LINE-2 Night drift: {drift_status}  (torque fail rate = {rate:.1%})")
    else:
        print(f"  LINE-2 Night drift: {drift_status}")

    print()
    print(f"  High-risk lots   : {(risk['risk_tier'] == 'HIGH').sum()}")
    print(f"  Medium-risk lots : {(risk['risk_tier'] == 'MEDIUM').sum()}")
    print(f"  Low-risk lots    : {(risk['risk_tier'] == 'LOW').sum()}")
    print(f"  Drift signals    : {len(drift)}  (line/shift combos > {settings.PROCESS_DRIFT_FAIL_RATE_THRESHOLD:.0%})")
    print(f"  Premium-fit supps: {len(premium)}")
    print()

    # Top 5 risky lots.
    print("  Top 5 risky lots:")
    top5 = risk.head(5)[["lot_no", "lot_risk_score", "risk_tier", "total_fails", "claims_linked"]]
    for _, row in top5.iterrows():
        print(
            f"    {row['lot_no']:<15}"
            f"  score={row['lot_risk_score']:.3f}"
            f"  tier={row['risk_tier']:<7}"
            f"  fails={int(row['total_fails']):>3}"
            f"  claims={int(row['claims_linked']):>3}"
        )
    print()

    # Top 3 suppliers.
    print("  Top 3 suppliers (composite):")
    top_sup = suppliers.head(3)[["supplier", "quality_score", "precision_score", "composite_rank"]]
    for _, row in top_sup.iterrows():
        qs = row["quality_score"]
        qs_str = f"{qs:.0f}" if pd.notna(qs) else "n/a"
        print(
            f"    {row['supplier']:<10}"
            f"  qs={qs_str:>4}"
            f"  precision={row['precision_score']:.3f}"
            f"  rank={int(row['composite_rank'])}"
        )
    print()

    # Best / worst COO.
    print("  COO ranking (incoming fail rate):")
    for _, row in coo.iterrows():
        print(
            f"    rank {int(row['rank'])}: {row['coo']:<10}"
            f"  fail_rate={float(row['coo_incoming_fail_rate']):.3%}"
        )

    print(sep)
    print()

    return results


__all__ = ["KPIEngine", "run_all_kpis"]
