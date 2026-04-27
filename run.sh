#!/usr/bin/env bash
set -euo pipefail

echo ""
echo " ============================================================"
echo "  AI Quality Inspection Copilot  |  Phase 4"
echo "  Powered by Claude AI + Streamlit"
echo " ============================================================"
echo ""

# Load .env if present
if [ -f .env ]; then
    echo "[INFO] Loading environment from .env ..."
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

# Check API key
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "[WARN] ANTHROPIC_API_KEY not set. Running in demo mode."
    echo "       Set it in .env to enable live Claude AI responses."
    echo ""
else
    echo "[OK]   ANTHROPIC_API_KEY detected - Claude AI enabled."
    echo ""
fi

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.10+."
    exit 1
fi

# Install deps if needed
if [ ! -f ".deps_installed" ]; then
    echo "[INFO] Installing dependencies ..."
    pip install -r requirements.txt --quiet
    touch .deps_installed
fi

# Free port 8501 if occupied
if command -v lsof &>/dev/null; then
    lsof -ti tcp:8501 | xargs kill -9 2>/dev/null || true
elif command -v fuser &>/dev/null; then
    fuser -k 8501/tcp 2>/dev/null || true
fi

echo "[INFO] Starting Streamlit on http://localhost:8501 ..."
echo "[INFO] Press Ctrl+C to stop."
echo ""

streamlit run app/frontend/streamlit_app.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false \
    --theme.base dark
