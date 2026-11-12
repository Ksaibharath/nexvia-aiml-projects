# Real-Time Fraud Detection Platform (backend)

Full MLOps implementation per the step-by-step guide: IEEE-CIS → Feast → XGBoost/MLflow → GraphSAGE → Kafka/FastAPI → Evidently → Streamlit → Kubernetes.

## Quick start (local)
```bash
docker compose up -d
# download data (accept Kaggle rules first, ~/.kaggle/kaggle.json)
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

## Kubernetes
See k8s/ manifests + guide Part 9. Build images inside Minikube: `eval $(minikube docker-env)`.

## Tests
`pytest -q` (Feast parity + API tests need running infra).
