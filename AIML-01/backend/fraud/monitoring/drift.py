import os, tempfile, time, pandas as pd, sqlalchemy as sa
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from fraud.serving.db import engine
from fraud.features.feature_list import FEATURES
from fraud.monitoring.retrain import retrain
SHARE = float(os.getenv("DRIFT_SHARE", "0.30")); MIN_ROWS = int(os.getenv("DRIFT_MIN_ROWS", "500"))
def current_window(n=5000):
    with engine.connect() as c:
        rows = [r[0] for r in c.execute(sa.text("SELECT features FROM scores ORDER BY scored_at DESC LIMIT :n"), {"n": n})]
    return pd.DataFrame(rows)[FEATURES].astype("float64") if rows else pd.DataFrame(columns=FEATURES)
def alert_precision(last=200):
    with engine.connect() as c:
        r = c.execute(sa.text("SELECT label FROM alerts WHERE label IS NOT NULL ORDER BY updated_at DESC LIMIT :n"), {"n": last}).fetchall()
    return (sum(x[0] == "fraud" for x in r) / len(r), len(r)) if len(r) >= 50 else (None, len(r))
def run_once():
    cur = current_window()
    if len(cur) < MIN_ROWS: print("not enough live rows yet:", len(cur)); return
    ref = pd.read_parquet("data/processed/reference_features.parquet")[FEATURES].astype("float64")
    ref = ref.sample(min(len(ref), 2 * len(cur)), random_state=0)
    rep = Report(metrics=[DataDriftPreset()]); rep.run(reference_data=ref, current_data=cur)
    res = rep.as_dict()["metrics"][0]["result"]; share = float(res["share_of_drifted_columns"])
    prec, n_lab = alert_precision()
    reason = None
    if share >= SHARE: reason = f"data drift: {share:.0%} of features drifted"
    elif prec is not None and prec < float(os.getenv("MIN_ALERT_PRECISION", "0.20")): reason = f"alert precision {prec:.0%} on {n_lab} reviewed"
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f: path = f.name
    rep.save_html(path); html = open(path, encoding="utf-8").read()
    with engine.begin() as c:
        c.execute(sa.text("INSERT INTO drift_reports (share_drifted, decision, html) VALUES (:s, :d, :h)"),
                  {"s": share, "d": reason or "no action", "h": html})
    print(f"drift share {share:.2f} | alert precision {prec} | decision: {reason or 'no action'}")
    if reason: retrain(reason)
if __name__ == "__main__":
    loop = int(os.getenv("DRIFT_LOOP_SECONDS", "0"))
    while True:
        run_once()
        if not loop: break
        time.sleep(loop)
