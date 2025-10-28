@echo off
echo Starting MT5 Windows Bridge...

REM Set environment variables
set MT5_BRIDGE_TOKEN=changeme
set MT5_BRIDGE_HOST=0.0.0.0
set MT5_BRIDGE_PORT=8787

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Start the bridge
echo Starting MT5 Bridge on %MT5_BRIDGE_HOST%:%MT5_BRIDGE_PORT%
echo Token: %MT5_BRIDGE_TOKEN%
echo.
echo Make sure MetaTrader5 is running and logged in!
echo.
python bridge.py

pause