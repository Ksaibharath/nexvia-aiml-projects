"""Test Airflow retraining quality_gate() pass and intentional regression block (Part 9)."""
import json

from airflow.dags.retrain_dag import quality_gate


def test_quality_gate_passes_when_no_prod_baseline(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "metrics.json").write_text(json.dumps({"ndcg@10": 0.08, "recall@300": 0.30}), encoding="utf-8")
    assert quality_gate(str(tmp_path)) is True


def test_quality_gate_passes_on_improved_model(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "metrics_prod.json").write_text(json.dumps({"ndcg@10": 0.080, "recall@300": 0.295}), encoding="utf-8")
    (art / "metrics.json").write_text(json.dumps({"ndcg@10": 0.082, "recall@300": 0.301}), encoding="utf-8")
    assert quality_gate(str(tmp_path)) is True


def test_quality_gate_blocks_degraded_model(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "metrics_prod.json").write_text(json.dumps({"ndcg@10": 0.080, "recall@300": 0.298}), encoding="utf-8")
    # Intentional degradation > 2% triggers ShortCircuitOperator to skip publish
    (art / "metrics.json").write_text(json.dumps({"ndcg@10": 0.065, "recall@300": 0.250}), encoding="utf-8")
    assert quality_gate(str(tmp_path)) is False
