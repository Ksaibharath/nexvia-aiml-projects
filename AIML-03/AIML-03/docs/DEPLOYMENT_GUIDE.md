# GitHub, Vercel, Netlify & Docker Deployment Guide

This repository is engineered so you can push once to **GitHub** and immediately deploy an interactive live demo to **Vercel** or **Netlify** (zero configuration required), as well as run the full **Docker Compose** stack (`Redis + NVIDIA Triton + FastAPI + Streamlit + Apache Airflow`) locally or on a cloud VM.

---

## Step 1 — Push the Completed Project to GitHub

1. Create a new **empty** repository on GitHub named `recsys-two-stage` (do **not** check "Add a README" or ".gitignore", as this repo already has clean commit history).
2. Run from inside `recsys-two-stage/`:

```bash
cd recsys-two-stage
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/recsys-two-stage.git
git branch -M main
git push -u origin main
```

### Why GitHub Push Succeeds Cleanly
* Large raw/processed datasets (`/data/`, `/artifacts/`, `/triton/model_repository/*/*/`) are excluded via `.gitignore` so no file exceeds GitHub's 100 MB limit.
* Meanwhile, `deploy_bundle/bundle.json` (**1.86 MB**) IS tracked in Git and stores the real trained Two-Tower, Wide & Deep, and DCN-V2 weights, 1,852 MovieLens items, 26 real user profiles, and evaluation benchmarks so cloud deployments work immediately after `git push`!

---

## Step 2A — Deploy on Vercel (Recommended — ~15 Seconds)

`vercel.json` and `api/index.mjs` are pre-configured at the root of this repository.

### Option 1: Via Vercel Web Dashboard (Easiest)
1. Go to [https://vercel.com/new](https://vercel.com/new) and sign in with GitHub.
2. Click **Import** next to your `recsys-two-stage` repository.
3. Leave all settings at their defaults (`Framework Preset: Other`, `Output Directory: public`).
4. Click **Deploy**.
5. Once deployed, open `https://<your-project>.vercel.app`:
   * Interactive 5-tab Web UI loads at `/`
   * Live Recommendation API runs at `/recommend?user_id=5&k=10`, `/api/compare?user_id=5`, `/api/metrics`, `/api/ab-results`, and `/health`!

### Option 2: Via Vercel CLI
```bash
npx vercel --prod
```

---

## Step 2B — Deploy on Netlify (~15 Seconds)

`netlify.toml` and `netlify/functions/api.mjs` are pre-configured at the root of this repository.

### Option 1: Via Netlify Web Dashboard (Easiest)
1. Go to [https://app.netlify.com](https://app.netlify.com) → **Add new site** → **Import an existing project**.
2. Connect **GitHub** and select `recsys-two-stage`.
3. Netlify automatically reads `netlify.toml` (`publish = "public"`, `functions = "netlify/functions"`).
4. Click **Deploy site**.
5. Open `https://<your-site>.netlify.app` — all 5 interactive tabs and `/recommend` + `/api/*` serverless endpoints work out of the box!

### Option 2: Via Netlify CLI
```bash
npx netlify-cli deploy --prod
```

---

## Step 3 — Full Local / VM Stack (Docker Compose: Redis + Triton + FastAPI + Streamlit)

To run the full container stack locally after generating artifacts:

```bash
# 1. Build sample dataset, features, models, FAISS index, and Cloud bundle
make all

# 2. Launch Redis + Triton Inference Server + FastAPI Gateway + Streamlit UI
docker compose up --build
```

* **Interactive Web App + FastAPI**: `http://localhost:8080`
* **Swagger OpenAPI Docs**: `http://localhost:8080/docs`
* **Streamlit Demo UI**: `http://localhost:8501`
* **Triton Health Check**: `curl http://localhost:8000/v2/health/ready`
* **Apache Airflow Weekly Retraining UI**: `cd airflow && docker compose up -d` → `http://localhost:8085`
