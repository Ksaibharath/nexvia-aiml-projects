@echo off
echo ============================================================
echo  Two-Stage Personalized Recommendation Engine - Windows Launcher
echo ============================================================
if not exist .venv (
    echo [1/3] Creating Python virtual environment .venv ...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo [2/3] Installing quickstart dependencies ...
pip install -q -r requirements-quickstart.txt
echo [3/3] Starting FastAPI + Interactive Web Dashboard at http://localhost:8080 ...
start http://localhost:8080
python -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8080
pause
