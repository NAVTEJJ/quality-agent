"""
Excel ingestion layer.

Reads every sheet from the source workbook and returns a normalised
dict of {sheet_name: DataFrame}.  Known data-quality quirks in the
source file are handled explicitly so downstream code sees clean frames.
"""
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-sheet handling rules
# ---------------------------------------------------------------------------

# Columns that should be silently dropped on load (trailing artefacts)
_COLUMNS_TO_DROP: Dict[str, list] = {
    "AI_Insights": ["Unnamed: 5", "Unnamed: 6"],
}

# Columns where NULLs are expected and should NOT trigger a warning.
# Values are (column, expected_null_count, explanation) tuples.
_EXPECTED_NULLS: Dict[str, list] = {
    "Incoming_QM": [
        ("DefectCode", 1431, "Most inspections pass — defect code is intentionally blank")
    ],
    "Warranty_Claims": [
        ("Region", 2, "Two claims have no region mapping — known source gap")
    ],
    "AI_Insights": [
        ("PatternDetected", None, "Insight rows are sparse by design — not every row has a pattern"),
        ("Evidence", None, "Insight rows are sparse by design"),
        ("RiskOrOpportunity", None, "Insight rows are sparse by design"),
        ("AI_Guidance", None, "Insight rows are sparse by design"),
        ("SuggestedActionables", None, "Actionables not populated for every insight row"),
    ],
}


def load_all_sheets(excel_path: Path) -> Dict[str, pd.DataFrame]:
    """Load every sheet from *excel_path* and return a ``{name: DataFrame}`` dict.

    Args:
        excel_path: Absolute or relative path to the ``.xlsx`` workbook.

    Returns:
        Mapping of sheet name → DataFrame with light cleaning applied.

    Raises:
        FileNotFoundError: If the workbook does not exist at the given path.
        ValueError: If the workbook contains no sheets.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Workbook not found: {excel_path}")

    logger.info("Opening workbook: %s", excel_path)

    xl = pd.ExcelFile(excel_path, engine="openpyxl")
    sheet_names = xl.sheet_names

    if not sheet_names:
        raise ValueError(f"Workbook has no sheets: {excel_path}")

    logger.info("Discovered %d sheet(s): %s", len(sheet_names), sheet_names)

    sheets: Dict[str, pd.DataFrame] = {}

    for name in sheet_names:
        logger.debug("Reading sheet '%s' …", name)
        df = xl.parse(name)

        # --- drop known trailing unnamed columns ---
        cols_to_drop = [
            c for c in _COLUMNS_TO_DROP.get(name, []) if c in df.columns
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(
                "[%s] Dropped artefact column(s): %s", name, cols_to_drop
            )

        # --- audit nulls ---
        null_summary = df.isnull().sum()
        non_zero_nulls = null_summary[null_summary > 0]

        expected_null_cols = {
            col for col, *_ in _EXPECTED_NULLS.get(name, [])
        }

        for col, count in non_zero_nulls.items():
            if col in expected_null_cols:
                logger.debug(
                    "[%s] Column '%s': %d null(s) — expected, skipping warning",
                    name, col, count,
                )
            else:
                logger.warning(
                    "[%s] Column '%s': %d unexpected null(s)", name, col, count
                )

        sheets[name] = df

        logger.info(
            "[%s] Loaded  shape=%s  columns=%s",
            name,
            df.shape,
            list(df.columns),
        )

    logger.info(
        "Ingestion complete — %d sheet(s) loaded from '%s'",
        len(sheets),
        excel_path.name,
    )
    return sheets
