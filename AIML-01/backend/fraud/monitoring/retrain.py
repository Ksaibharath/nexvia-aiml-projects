import os, sqlalchemy as sa
from mlflow.tracking import MlflowClient
from fraud.serving.db import engine
def load_feedback():
    import pandas as pd
    with engine.connect() as c:
        return pd.read_sql(sa.text("SELECT transaction_id, label FROM alerts WHERE label IS NOT NULL"), c)
def retrain(reason):
    from fraud.models import train_xgb as T
    fb = load_feedback()
    run_id, challenger_ap = T.fit_and_log(extra_labels=fb, name="auto_retrain")
    client = MlflowClient()
    champ = client.get_model_version_by_alias("fraud-xgb", "champion")
    champ_ap = float(client.get_run(champ.run_id).data.metrics["test_pr_auc"])
    promoted = challenger_ap >= 0.99 * champ_ap
    if promoted:
        mv = T.register(run_id)
        client.set_registered_model_alias("fraud-xgb", "champion", mv.version)
    with engine.begin() as c:
        c.execute(sa.text("INSERT INTO retrain_log (reason, champion_ap, challenger_ap, promoted) VALUES (:r, :c, :h, :p)"),
                  {"r": reason, "c": champ_ap, "h": challenger_ap, "p": promoted})
    print("retrain:", reason, "| champion", champ_ap, "| challenger", challenger_ap, "| promoted", promoted)
