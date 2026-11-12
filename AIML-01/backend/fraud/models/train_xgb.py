import os, json, numpy as np, pandas as pd, mlflow, mlflow.xgboost, xgboost as xgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from fraud.models.data import load_frame
from fraud.features.feature_list import FEATURES
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("fraud-xgb")
BASE = dict(n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.7,
            min_child_weight=3, tree_method="hist", eval_metric="aucpr", early_stopping_rounds=100, n_jobs=-1)
def metrics(y, p, prefix):
    k = max(int(0.01 * len(y)), 1); top = np.argsort(-p)[:k]
    return {f"{prefix}_pr_auc": average_precision_score(y, p), f"{prefix}_roc_auc": roc_auc_score(y, p),
            f"{prefix}_precision_at_1pct": float(y[top].mean()), f"{prefix}_recall_at_1pct": float(y[top].sum() / y.sum())}
def fit_and_log(extra_labels=None, name="run", df=None):
    if df is None: df = load_frame()
    if extra_labels is not None and len(extra_labels):
        m = dict(zip(extra_labels.transaction_id, (extra_labels.label == "fraud").astype(int)))
        df["isFraud"] = df.apply(lambda r: m.get(r.transaction_id, r.isFraud), axis=1)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    va[FEATURES].sample(min(20000, len(va)), random_state=0).to_parquet("data/processed/reference_features.parquet")
    neg, pos = (tr.isFraud == 0).sum(), (tr.isFraud == 1).sum()
    spw = float(neg / pos)
    with mlflow.start_run(run_name=name) as r:
        mlflow.log_params({**BASE, "scale_pos_weight": spw, "n_features": len(FEATURES), "train_rows": len(tr)})
        model = xgb.XGBClassifier(**BASE, scale_pos_weight=spw, random_state=0)
        model.fit(tr[FEATURES], tr.isFraud.values, eval_set=[(va[FEATURES], va.isFraud.values)], verbose=False)
        pv, pt = model.predict_proba(va[FEATURES])[:, 1], model.predict_proba(te[FEATURES])[:, 1]
        m = {**metrics(va.isFraud.values, pv, "val"), **metrics(te.isFraud.values, pt, "test"), "best_iteration": model.best_iteration}
        mlflow.log_metrics(m)
        pr, rc, _ = precision_recall_curve(te.isFraud.values, pt)
        plt.figure(); plt.plot(rc, pr); plt.xlabel("recall"); plt.ylabel("precision"); plt.title(f"{name} (test)")
        plt.savefig("pr_curve.png"); mlflow.log_artifact("pr_curve.png")
        imp = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
        mlflow.log_dict(imp.head(30).to_dict(), "feature_importance_gain.json")
        mlflow.set_tags({"dataset": "ieee-cis", "split": "time 70/15/15", "feature_service": "txn_v1"})
        mlflow.xgboost.log_model(model, "model")
        print(name, {k: round(float(v), 4) for k, v in m.items()})
        return r.info.run_id, m["val_pr_auc"]
def register(run_id):
    mv = mlflow.register_model(f"runs:/{run_id}/model", "fraud-xgb")
    MlflowClient().set_registered_model_alias("fraud-xgb", "champion", mv.version)
    json.dump({"run_id": run_id, "version": mv.version}, open("artifacts/champion_run.json", "w"))
    return mv
def main():
    df = load_frame()
    tr = df[df.split == "train"]
    neg, pos = (tr.isFraud == 0).sum(), (tr.isFraud == 1).sum()
    # Full 4-run study is in docs; champion path uses class weights (see README).
    run_id, ap = fit_and_log(name="class_weight", df=df)
    register(run_id)
    print("champion:", run_id)
if __name__ == "__main__":
    main()
