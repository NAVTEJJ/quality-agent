"""
Central configuration for the Quality Agent.
All paths, DB URLs, and analytical thresholds live here.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

EXCEL_PATH = BASE_DIR / "AI_QM_Mechanical_Classy_Demo_MultiTab 1.xlsx"

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DICTIONARIES_DIR = DATA_DIR / "dictionaries"

# Ensure runtime directories exist
for _dir in (RAW_DIR, PROCESSED_DIR, DICTIONARIES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL = f"sqlite:///{PROCESSED_DIR / 'quality_agent.db'}"

# ---------------------------------------------------------------------------
# Analytical thresholds
# ---------------------------------------------------------------------------
# A lot whose defect rate exceeds this fraction is considered at-risk
LOT_RISK_FAIL_RATE_THRESHOLD = 0.05

# A process step whose rolling defect rate exceeds this fraction signals drift
PROCESS_DRIFT_FAIL_RATE_THRESHOLD = 0.10

# Composite risk score above which a lot is escalated as high-risk
HIGH_RISK_LOT_SCORE_THRESHOLD = 0.60

# Supplier scorecard rating (0–100) below which the supplier is non-preferred
PREFERRED_SUPPLIER_SCORE_THRESHOLD = 80

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
