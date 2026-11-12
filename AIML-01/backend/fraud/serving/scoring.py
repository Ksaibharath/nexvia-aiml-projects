import json, os, time, numpy as np, xgboost as xgb, mlflow, mlflow.xgboost
from feast import FeatureStore
from mlflow.tracking import MlflowClient
from fraud.features.feature_list import FEATURES, TXN_REFS, GRAPH_REFS
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
logit = lambda p: float(np.log(p / (1 - p)))
class Scorer:
    def __init__(self):
        self.store = FeatureStore(repo_path=os.getenv("FEAST_REPO", "feature_repo"))
        self.version = None; self.load_champion()
    def load_champion(self):
        c = MlflowClient(); mv = c.get_model_version_by_alias("fraud-xgb", "champion")
        if mv.version == self.version: return False
        m = mlflow.xgboost.load_model(f"models:/fraud-xgb/{mv.version}")
        path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="serving_config.json")
        self.cfg = json.load(open(path)); self.booster = m.get_booster(); self.version = mv.version
        return True
    def score(self, tid, uid, shift=None):
        t0 = time.perf_counter()
        r = self.store.get_online_features(features=TXN_REFS + GRAPH_REFS,
                                           entity_rows=[{"transaction_id": int(tid), "uid": uid}]).to_dict()
        feast_ms = (time.perf_counter() - t0) * 1000
        x = {f: (np.nan if r[f][0] is None else float(r[f][0])) for f in FEATURES}
        if shift and os.getenv("ALLOW_SIM") == "1":
            for f, delta in shift.items(): x[f] = x.get(f, 0.0) + delta
        d = xgb.DMatrix(np.array([[x[f] for f in FEATURES]], dtype="float32"), feature_names=FEATURES)
        p_x = float(np.clip(self.booster.predict(d)[0], 1e-6, 1 - 1e-6))
        g = r["gnn_score"][0]; p_g = float(np.clip(self.cfg["prior"] if g is None else g, 1e-6, 1 - 1e-6))
        a = self.cfg["alpha"]; z = a * logit(p_x) + (1 - a) * logit(p_g); final = 1 / (1 + np.exp(-z))
        flagged = bool(final >= self.cfg["threshold"]); reasons = None
        if flagged:
            contrib = self.booster.predict(d, pred_contribs=True)[0][:-1]
            top = np.argsort(-np.abs(contrib))[:3]
            reasons = [{"feature": FEATURES[i], "value": None if np.isnan(x[FEATURES[i]]) else round(x[FEATURES[i]], 3),
                        "impact": round(float(contrib[i]), 3)} for i in top]
        return {"transaction_id": int(tid), "uid": uid, "xgb": p_x, "gnn": p_g, "final": float(final), "flagged": flagged,
              "ring_id": r["ring_id"][0], "ring_size": r["ring_size"][0], "reasons": json.dumps(reasons),
              "feast_ms": round(feast_ms, 2), "total_ms": round((time.perf_counter() - t0) * 1000, 2),
              "features_json": json.dumps({k: (None if np.isnan(v) else v) for k, v in x.items()}),
              "model_version": str(self.version)}
