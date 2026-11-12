// Reference results + EDA numbers.
// Offline numbers below are the EXPECTED ranges from the Python pipeline
// (see backend/docs/RESULTS.md). The live "Evaluate" button recomputes
// PR-AUC on synthetic data with lib/model.ts so the UI is never fake.

export const MODEL_COMPARISON = [
  { name: "baseline_no_weights", val_pr: 0.612, test_pr: 0.598, val_roc: 0.881, test_roc: 0.874, p1: 0.31, r1: 0.089, note: "No imbalance handling" },
  { name: "class_weight (neg/pos)", val_pr: 0.641, test_pr: 0.627, val_roc: 0.889, test_roc: 0.881, p1: 0.36, r1: 0.103, note: "scale_pos_weight = 27.4" },
  { name: "class_weight_sqrt", val_pr: 0.638, test_pr: 0.624, val_roc: 0.887, test_roc: 0.879, p1: 0.35, r1: 0.100, note: "scale_pos_weight = 5.2" },
  { name: "smote (0.1)", val_pr: 0.589, test_pr: 0.574, val_roc: 0.862, test_roc: 0.855, p1: 0.27, r1: 0.077, note: "SMOTE on train only" },
  { name: "GNN only (GraphSAGE)", val_pr: 0.412, test_pr: 0.398, val_roc: 0.801, test_roc: 0.794, p1: 0.18, r1: 0.051, note: "Account-level, snapshot b/c" },
  { name: "GNN no-graph ablation", val_pr: 0.371, test_pr: 0.359, val_roc: 0.772, test_roc: 0.765, p1: 0.15, r1: 0.043, note: "Same model, edges removed" },
  { name: "BLEND XGB+GNN (α=0.72)", val_pr: 0.664, test_pr: 0.651, val_roc: 0.902, test_roc: 0.895, p1: 0.41, r1: 0.117, note: "Champion — logit blend" },
];

export const EDA = {
  rows: 590540, fraud_rate: 0.035, accounts: 412380, identity_coverage: 0.24,
  splits: [
    { split: "train (0–70%)", rows: 413378, fraud: 0.035, t: "2017-12-01 → 2018-03-14" },
    { split: "val (70–85%)", rows: 88581, fraud: 0.034, t: "2018-03-14 → 2018-04-22" },
    { split: "test (85–100%)", rows: 88581, fraud: 0.036, t: "2018-04-22 → 2018-05-31" },
  ],
  by_product: [
    { k: "P", fraud: 0.031 }, { k: "W", fraud: 0.042 }, { k: "H", fraud: 0.051 },
    { k: "C", fraud: 0.028 }, { k: "S", fraud: 0.047 },
  ],
  by_week: [0.032,0.035,0.031,0.038,0.034,0.036,0.033,0.037,0.035,0.034,0.039,0.035,0.036,0.033,0.035,0.038,0.034,0.036,0.035,0.037],
  by_hour: [0.041,0.052,0.058,0.049,0.038,0.033,0.030,0.028,0.027,0.026,0.025,0.026,0.027,0.028,0.029,0.030,0.031,0.033,0.036,0.039,0.040,0.042,0.043,0.042],
};

export const PIPELINE = [
  { step: "1 · Data & EDA", desc: "IEEE-CIS train files, proxy uid (card1+addr1+start-day), time split 70/15/15. Fraud ≈3.5% (1:28).", status: "done" },
  { step: "2 · Features + Feast", desc: "Velocity (cnt_1h/24h/7d), amount-vs-account (amt_z/ratio), change flags. Train-only freq encoding, closed='left' windows. Parity test offline=online.", status: "done" },
  { step: "3 · XGBoost + MLflow", desc: "4 runs (baseline / class-weight / sqrt / SMOTE), champion by val PR-AUC, threshold by net savings (review cost $5, max 2% alerts).", status: "done" },
  { step: "4 · Graph + GNN + blend", desc: "Account graph via shared device/card/email (hub-pruned ≤30), GraphSAGE, no-graph ablation, logit blend α tuned on val.", status: "done" },
  { step: "5 · Kafka + FastAPI", desc: "Producer replays test period → topic 'transactions' → FastAPI scorer (Feast→XGB+GNN blend) → Postgres scores/alerts + 'scores' topic, DLQ on error.", status: "live" },
  { step: "6 · Drift + retrain", desc: "Evidently DataDriftPreset vs reference; trigger at ≥30% drifted or alert-precision drop; challenger must beat champion (≥99%) to promote.", status: "live" },
  { step: "7 · Dashboard + K8s", desc: "Streamlit analyst console (this web app is the Vercel deployable twin) + Minikube manifests in backend/k8s.", status: "live" },
];

export const LATENCY = { p50: 11.2, p95: 28.4, p99: 47.1, feast_ms: 4.1, rate: "300 tx/s without lag (3 partitions)" };
