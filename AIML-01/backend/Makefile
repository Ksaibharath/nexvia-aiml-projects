.PHONY: setup infra data features train graph serve test
setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
infra:
	docker compose up -d
data:
	python -m fraud.data.prepare
features:
	python -m fraud.features.build_features
train:
	python -m fraud.models.train_xgb
graph:
	python -m fraud.graph.train_gnn
serve:
	uvicorn fraud.serving.api:app --port 8080
test:
	pytest -q
