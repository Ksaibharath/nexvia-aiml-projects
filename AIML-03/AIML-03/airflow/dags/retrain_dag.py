"""Weekly Airflow retraining DAG with automated NDCG@10 and Recall@300 quality gate."""
import json
import pathlib
from datetime import datetime, timedelta

PROJ = "/opt/project"


def quality_gate(proj_dir: str = PROJ) -> bool:
    """Publish only if the new model is at least 98% as strong as production on NDCG@10 and Recall@300."""
    metrics_file = pathlib.Path(proj_dir) / "artifacts" / "metrics.json"
    prod_file = pathlib.Path(proj_dir) / "artifacts" / "metrics_prod.json"
    new = json.loads(metrics_file.read_text(encoding="utf-8"))
    if not prod_file.exists():
        return True
    prod = json.loads(prod_file.read_text(encoding="utf-8"))
    return (
        float(new["ndcg@10"]) >= 0.98 * float(prod["ndcg@10"])
        and float(new["recall@300"]) >= 0.98 * float(prod["recall@300"])
    )


try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import ShortCircuitOperator

    def sh(task_id: str, target: str):
        return BashOperator(task_id=task_id, bash_command=f"cd {PROJ} && make {target}")

    with DAG(
        "recsys_weekly_retrain",
        start_date=datetime(2026, 1, 1),
        schedule="@weekly",
        catchup=False,
        default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
        tags=["recsys"],
    ) as dag:
        split = sh("clean_and_split", "split")
        features = sh("spark_features", "features")
        retrieval = sh("train_two_tower", "retrieval")
        index = sh("build_faiss", "index")
        ranking = sh("train_rankers", "ranking")
        evaluate = sh("evaluate", "evaluate")
        gate = ShortCircuitOperator(task_id="quality_gate", python_callable=quality_gate)
        publish = sh("publish_to_triton", "publish")

        split >> features >> retrieval >> index >> ranking >> evaluate >> gate >> publish
except ImportError:
    # Allows importing quality_gate() in lightweight pytest environments without apache-airflow installed
    dag = None
