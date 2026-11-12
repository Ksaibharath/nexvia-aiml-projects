.PHONY: download split features retrieval index ranking evaluate ab_sim bundle publish test serve ui all

download:
	python3 src/data/download.py

split:
	python3 src/data/clean_split.py

features:
	python3 src/spark/build_features.py

retrieval:
	python3 -m src.retrieval.train

index:
	python3 src/index/build_faiss.py

ranking:
	python3 -m src.ranking.build_ranking_data && python3 -m src.ranking.train_all

evaluate:
	python3 -m src.evaluation.eval_retrieval && python3 -m src.evaluation.eval_ranking

ab_sim:
	python3 -m src.evaluation.ab_sim

bundle:
	python3 -m src.serving.bundle_export

publish:
	python3 -m src.serving.publish

test:
	pytest -q

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8080

ui:
	streamlit run src/serving/ui.py --server.port 8501 --server.address 0.0.0.0

all: download split features retrieval index ranking evaluate bundle test
