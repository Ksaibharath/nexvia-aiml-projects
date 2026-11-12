# 🏦 Nexvia Bank — Fraud Protection Prototype

**Banking app prototype with real-time fraud detection · IEEE-CIS · XGBoost + GraphSAGE blend · Feast · MLflow · Kafka · FastAPI · Evidently**

A bank/payment company must decide in milliseconds whether to approve, block, or review each card transaction. This platform scores every transaction as it arrives, finds fraud rings (accounts sharing devices/cards), shows analysts *why* each alert fired, learns from their decisions, and watches itself for drift.

> **Two deployable parts in one repo:**
> - **Root (`/`) — Nexvia Bank prototype (Home, Send Money, Transactions, Cards, Alerts) + scoring API** → deploy to **Vercel** (recommended) or **Netlify**. Runs anywhere, no Kafka/Postgres needed. Live stream, alert queue, ring view, model lab, drift checks.
> - **`backend/` — full Python MLOps stack** (Feast, MLflow, Kafka, FastAPI, Postgres, Streamlit, Minikube) → run locally with Docker Compose or on Kubernetes. Same scoring contract, real trained models.

---

## 🚀 Deploy in 3 minutes (GitHub → Vercel)

### 1. Push to GitHub
```bash
cd fraud-platform
git init
git add .
git commit -m "Real-time fraud detection platform"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/fraud-platform.git
git push -u origin main
```

### 2a. Vercel (recommended)
1. Go to [vercel.com](https://vercel.com) → **Add New → Project** → import `fraud-platform`.
2. Framework: **Next.js** (auto-detected). Build: `npm run build`. No env vars needed.
3. Click **Deploy** → you get a live URL like `fraud-platform.vercel.app`.

### 2b. Netlify (alternative)
1. Go to [app.netlify.com](https://app.netlify.com) → **Add new site → Import an existing project** → pick the repo.
2. Build command `npm run build`, publish `.next` (already in `netlify.toml`).
3. **Deploy site** → live URL.

### 3. Try it
- **Overview** — KPIs + EDA charts.
- **Live Stream** — ▶ Start stream (simulated Kafka replay). Alerts appear in real time.
- **Alert Queue** — open a case, see top-3 reasons + fraud-ring graph, click **Confirm fraud / False positive**.
- **Model Lab** — PR-AUC table + live recomputation.
- **Monitoring** — run drift check (toggle *Drifted data* to see retraining trigger).
- API: `POST /api/score`, `GET /api/stream?n=20`, `GET /api/metrics`, `GET /api/drift`, `GET /api/health`.

---

## 🏗️ Architecture

```
OFFLINE (training)                          ONLINE (every transaction)
IEEE-CIS CSVs ──► feature pipeline ──► Feast ──► XGBoost + MLflow (champion)
                        │                      Account graph + GraphSAGE ──► logit blend (α)
                        └──────────────────────► Kafka producer (replay test period)
                                                   │ topic: transactions
                                                   ▼
                                              FastAPI scorer ──► Postgres (scores/alerts)
                                                   │                    ▲
                                              Streamlit dashboard ─────┘ (analyst labels)
                                                   ▲
                                              Evidently drift ──► retrain ──► champion/challenger gate
```

**Web deploy twin:** the Next.js app implements the *same* flow in-browser/serverless — synthetic IEEE-CIS-like generator (`lib/model.ts`) → XGBoost-like scorer + GNN-like ring risk → logit blend (α=0.72, threshold=0.42) → alert queue → analyst feedback → drift check. API routes mirror the FastAPI contract 1:1.

---

## 📊 Results (headline)

| Model | Val PR-AUC | Test PR-AUC |
|---|---|---|
| Baseline (no weights) | 0.612 | 0.598 |
| Class weight | 0.641 | 0.627 |
| Class weight √ | 0.638 | 0.624 |
| SMOTE | 0.589 | 0.574 |
| GNN only | 0.412 | 0.398 |
| GNN no-graph ablation | 0.371 | 0.359 |
| **Blend XGB+GNN (champion)** | **0.664** | **0.651** |

- Fraud rate ≈ **3.5%** (1:28) — never use accuracy (always-legit = 96.5%).
- Threshold by net savings (review cost $5, ≤2% alerts). Latency p95 ≈ 28 ms (Feast ≈4 ms).
- Full tables: `backend/docs/RESULTS.md`. Live recomputation: **Model Lab** tab.

---

## 🐍 Full backend (local / Kubernetes)

```bash
cd backend
docker compose up -d   # Redis, Postgres, Kafka, MLflow
# Download data (accept Kaggle rules first; ~/.kaggle/kaggle.json)
kaggle competitions download -c ieee-fraud-detection -p data/raw
cd data/raw && unzip -o ieee-fraud-detection.zip train_transaction.csv train_identity.csv && cd ../..
python -m fraud.data.prepare
python -m fraud.features.build_features
cd feature_repo && PYTHONPATH=.. feast apply && PYTHONPATH=.. feast materialize 2017-11-30T00:00:00 2018-07-01T00:00:00 && cd ..
python -m fraud.models.train_xgb
python -m fraud.graph.train_gnn && python -m fraud.graph.train_gnn --no-graph
python -m fraud.models.blend
uvicorn fraud.serving.api:app --port 8080
python -m fraud.serving.producer --rate 50 --limit 5000
streamlit run fraud/dashboard/app.py
```
Kubernetes: `backend/k8s/*.yaml` (Minikube). Tests: `pytest -q`. Details: `backend/README.md`, `backend/docs/`.

---

## ⚠️ Limitations (stated openly)
- `uid` is a **proxy** account id (card1+addr1+start-day); IEEE-CIS has no real customer id.
- `TransactionDT` is seconds since an unknown reference; dates/hour are relative.
- Streaming is a **replay**; features are precomputed point-in-time correct.
- Analyst feedback in demo is simulated from ground truth unless labelled by hand; real labels arrive late.
- Single dataset, no adversarial adaptation; IEEE-CIS is anonymised (no PII).

## 📁 Repo map
```
fraud-platform/
  app/            Next.js pages + API routes (Vercel/Netlify)
  lib/            scoring engine, synthetic data, reference results
  components/     (charts inline in app/page.tsx)
  backend/        Python MLOps stack (Feast, XGBoost, GNN, Kafka, K8s, tests, docs)
  vercel.json · netlify.toml · package.json
```
