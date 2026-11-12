#!/usr/bin/env bash
set -e
echo "============================================================"
echo " Two-Stage Personalized Recommendation Engine - Local Launcher"
echo "============================================================"
if [ ! -d ".venv" ]; then
  echo "[1/3] Creating Python virtual environment .venv ..."
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "[2/3] Installing quickstart dependencies ..."
pip install -q -r requirements-quickstart.txt
echo "[3/3] Starting FastAPI + Interactive Web Dashboard at http://localhost:8080 ..."
python3 -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8080
