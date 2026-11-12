# How to Run Locally on Your System (Windows, macOS, or Linux)

Download **`recsys-two-stage.zip`**, unzip it onto your computer, and open a terminal (`PowerShell` / `CMD` on Windows, or `Terminal` on macOS/Linux) inside the `recsys-two-stage` folder.

You have **3 ways** to run it locally—pick whichever you want to test first!

---

## Option 1 — Instant Run in 10 Seconds (Pre-Trained Model Bundle Included!)

Because `deploy_bundle/bundle.json` (1.86 MB) already contains the trained Two-Tower, DCN-V2, and Wide & Deep weights + 1,852 MovieLens movies + evaluation metrics, you can launch the full Web App & API **immediately** without waiting to download the 250 MB dataset!

### Method 1A: Python FastAPI + Swagger `/docs` + 5-Tab Web App (Recommended)

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-quickstart.txt
python run_pipeline.py serve
```
*(Or simply double-click `start_windows.bat`!)*

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-quickstart.txt
python3 run_pipeline.py serve
```
*(Or run `bash start_mac_linux.sh`!)*

Now open in your browser:
* **Interactive 5-Tab Web App**: [http://localhost:8080](http://localhost:8080)
* **FastAPI Swagger OpenAPI Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)
* **Optional Streamlit UI** (open a second terminal with `.venv` active):
  ```bash
  python run_pipeline.py ui
  ```
  Then open [http://localhost:8501](http://localhost:8501).

### Method 1B: Zero-Install Node.js Mode (1 Second!)
If you have Node.js (v18+) installed, you don't even need `pip install`:
```bash
npm start
```
Then open [http://localhost:8080](http://localhost:8080).

---

## Option 2 — Train the Entire ML Pipeline from Scratch on Your Machine

When you want to download MovieLens 25M (~250 MB), run the chronological split, build features, train the Two-Tower neural net, build the FAISS HNSW index, train both DCN-V2 and Wide & Deep rankers, run the A/B replay simulation, and run all 20 `pytest` tests on your own CPU/GPU:

```bash
# Activate your .venv first, then install full ML packages (TensorFlow + FAISS + PyArrow + SciPy)
pip install -r requirements.txt

# Run the entire end-to-end offline pipeline + tests (takes ~3-4 minutes on sample_users: 5000)
python run_pipeline.py all

# Start the server (now automatically switches to Full FAISS + Parquet mode!)
python run_pipeline.py serve
```

> **Tip**: In `configs/config.yaml`, `sample_users: 5000` runs the whole pipeline in ~3 minutes on any laptop. Set `sample_users: null` if you have >=16 GB RAM and want to train on all 25,000,095 rows.

---

## Option 3 — Full Production Docker Compose Stack (`Redis + Triton + FastAPI + Streamlit`)

If you have Docker Desktop running (after running `python run_pipeline.py all` so `artifacts/` and `triton/model_repository/*/1/model.savedmodel` are populated on your machine):

```bash
docker compose up --build
```
* **Web App & FastAPI**: `http://localhost:8080`
* **Swagger Docs**: `http://localhost:8080/docs`
* **Streamlit Demo**: `http://localhost:8501`
* **NVIDIA Triton Health**: `http://localhost:8000/v2/health/ready`

---

## Troubleshooting on Your System

* **Windows PowerShell ExecutionPolicy error** when running `Activate.ps1`:
  Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use `.venv\Scripts\activate.bat` in Command Prompt.
* **Port 8080 already in use**:
  Run on port `8090` instead: `python -m uvicorn src.serving.api:app --port 8090` (or `PORT=8090 npm start`).
