import json, os, numpy as np, pandas as pd, mlflow, mlflow.xgboost
from feast import FeatureStore
from sklearn.metrics import average_precision_score
from fraud.features.feature_list import FEATURES, TXN_REFS, GRAPH_REFS
from fraud.models.threshold import best_threshold
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
champ = json.load(open("artifacts/champion_run.json"))
model = mlflow.xgboost.load_model(f"models:/fraud-xgb/{champ['version']}")
store = FeatureStore(repo_path="feature_repo")
labels = pd.read_parquet("data/processed/labels.parquet")
labels = labels[labels.split.isin(["val", "test"])]
f = store.get_historical_features(entity_df=labels[["transaction_id", "uid", "event_timestamp"]],
                                  features=TXN_REFS + GRAPH_REFS).to_df()
df = labels.merge(f, on=["transaction_id", "uid", "event_timestamp"])
prior = float(df.gnn_score.mean())
df["gnn_p"] = df.gnn_score.fillna(prior).clip(1e-6, 1 - 1e-6)
df["xgb_p"] = model.predict_proba(df[FEATURES])[:, 1].clip(1e-6, 1 - 1e-6)
logit = lambda p: np.log(p / (1 - p))
def blend(xp, gp, a): z = a * logit(xp) + (1 - a) * logit(gp); return 1 / (1 + np.exp(-z))
va, te = df[df.split == "val"], df[df.split == "test"]
rows = []
for a in np.linspace(0, 1, 21):
    rows.append((a, average_precision_score(va.isFraud, blend(va.xgb_p, va.gnn_p, a))))
alpha = max(rows, key=lambda r: r[1])[0]
print("alpha (weight on XGBoost) =", alpha)
for name, d in [("val", va), ("test", te)]:
    print(name, "XGB", round(average_precision_score(d.isFraud, d.xgb_p), 4), "| GNN-only",
          round(average_precision_score(d.isFraud, d.gnn_p), 4), "| BLEND",
          round(average_precision_score(d.isFraud, blend(d.xgb_p, d.gnn_p, alpha)), 4))
pv = blend(va.xgb_p.values, va.gnn_p.values, alpha)
thr, stats = best_threshold(va.isFraud.values, pv, va.TransactionAmt.values, review_cost=5.0)
pt = blend(te.xgb_p.values, te.gnn_p.values, alpha); flag = pt >= thr
print("validation stats", stats)
print("TEST at validation threshold: alert_rate", flag.mean(), "precision", te.isFraud.values[flag].mean(),
      "recall", te.isFraud.values[flag].sum() / te.isFraud.sum())
cfg = {"features": FEATURES, "alpha": float(alpha), "threshold": float(thr), "prior": prior, "review_cost": 5.0}
with mlflow.start_run(run_id=champ["run_id"]):
    mlflow.log_dict(cfg, "serving_config.json"); mlflow.log_metrics({"blend_val_threshold": thr})
json.dump(cfg, open("artifacts/serving_config.json", "w"))
