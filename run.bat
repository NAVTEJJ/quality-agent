@echo off
setlocal enabledelayedexpansion

title AI Quality Inspection Copilot

echo.
echo  ============================================================
echo   AI Quality Inspection Copilot  ^|  Phase 4
echo   Powered by Claude AI + Streamlit
echo  ============================================================
echo.

:: Check for Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: Load .env if present
if exist .env (
    echo [INFO] Loading environment from .env ...
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" if not "%%B"=="" (
            set "%%A=%%B"
        )
    )
)

:: Check API key
if "%ANTHROPIC_API_KEY%"=="" (
    echo [WARN] ANTHROPIC_API_KEY not set. Running in demo mode.
    echo        Set it in .env to enable live Claude AI responses.
    echo.
) else (
    echo [OK]   ANTHROPIC_API_KEY detected - Claude AI enabled.
    echo.
)

:: Install dependencies if needed
if not exist ".deps_installed" (
    echo [INFO] Installing dependencies ...
    pip install -r requirements.txt --quiet
    echo. > .deps_installed
)

:: Kill any existing Streamlit on port 8501
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501 "') do (
    taskkill /f /pid %%a >nul 2>&1
)

echo [INFO] Starting Streamlit on http://localhost:8501 ...
echo [INFO] Press Ctrl+C to stop.
echo.

streamlit run app/frontend/streamlit_app.py ^
    --server.port 8501 ^
    --server.headless false ^
    --browser.gatherUsageStats false ^
    --theme.base dark

endlocal
