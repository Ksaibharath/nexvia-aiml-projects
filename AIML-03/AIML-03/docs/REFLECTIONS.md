# Individual Reflections & Oral Defense Q&A Guide

---

## Part A — Individual Reflection Documents (1 per Role)

### Intern A — Data Engineering & Spark Features (`Parts 1, 2, 3`)
1. **What I owned**: Repository scaffolding, `src/data/download.py`, `src/data/clean_split.py`, `src/spark/build_features.py`, `tests/test_split.py`, and `docs/DATA_DICTIONARY.md`. Produced 6 leak-free Parquet tables (`588,429` train rows, `73,554` val rows, `73,554` test rows).
2. **Technical decision I am proud of**: Designing a dual-engine feature pipeline (PySpark + chunked PyArrow fallback) that reserves integer ID `0` exclusively for sequence padding and post-train cold-start entities, while computing all popularity and genre-affinity aggregations strictly on `split == 'train'`.
3. **What went wrong and how I debugged it**: Initially, writing `data/` without a leading slash in `.gitignore` silently ignored `src/data/` during `git add`. I caught it via `git status`, anchored root ignore patterns (`/data/`, `/artifacts/`), and verified clean staging.
4. **What I learned**: *Technical*: Spark window frame boundaries (`rowsBetween(-20, -1)`) prevent subtle target leakage. *Teamwork*: Locking down Parquet schemas early unblocked Retrieval and Ranking in parallel. *Process*: Writing `test_split.py` immediately after `clean_split.py` caught boundary bugs on day one.
5. **What I would improve next**: Add incremental daily Parquet compaction and integrate Apache Iceberg / Feast so online and offline stores share point-in-time joins natively.
6. **Contribution to demo, docs & presentation**: Authored `DATA_DICTIONARY.md`, built the Split Integrity panel in Tab 4, and prepared Slides 1 & 4.

---

### Intern B — Two-Tower Retrieval, FAISS & Cold-Start (`Parts 4, 5, 7`)
1. **What I owned**: `src/retrieval/{data.py, model.py, train.py, export.py}`, `src/index/build_faiss.py`, `src/serving/cold_start.py`, `src/evaluation/eval_retrieval.py`, and `tests/test_cold_start.py`.
2. **Technical decision I am proud of**: Adding **LogQ frequency correction** (`logits - 0.45 * log(p_item)`) and accidental duplicate-positive masking to in-batch softmax training. This stopped popular items from being over-penalized and pushed Two-Tower `Recall@300` to `0.2979` (beating the strong MovieLens Popularity baseline `0.2694` by `+10.6%`).
3. **What went wrong and how I debugged it**: Monitoring validation top-K accuracy for 9 epochs caused `user_emb` to overfit pre-2016 training histories, hurting 2018 test recall. Switching `EarlyStopping` back to `val_total_loss` restored the optimal Epoch 1–2 generalization weights.
4. **What I learned**: *Technical*: Off-by-one alignment (`item_idx = faiss_pos + 1`) and L2-normalizing both query and index vectors are non-negotiable for inner-product cosine search. *Process*: Always benchmark against Popularity before moving to Stage 2.
5. **What I would improve next**: Upgrade the masked-mean `UserTower` to a causal Transformer (`SASRec`) to model sequential item transitions.
6. **Contribution to demo, docs & presentation**: Built the Cold-Start Onboarding Lab (Tab 2) and FAISS HNSW scaling benchmark (`99.98%` catalog recall, `25.63x` QPS speedup at 300k vectors), and prepared Slides 2 & 3.

---

### Intern C — Deep Ranking (DCN-V2 / Wide & Deep) & A/B Analysis (`Parts 6, 10`)
1. **What I owned**: `src/ranking/{features.py, build_ranking_data.py, model.py, train.py, train_all.py, export.py}`, `src/evaluation/{metrics.py, eval_ranking.py, ab_sim.py}`, `tests/test_metrics.py`, and `docs/RESULTS.md`.
2. **Technical decision I am proud of**: Centralizing all 50 user, item, retrieval, and genre-cross features in a single `build_dense()` function shared identically across offline ranker training and live API serving, eliminating training-serving skew.
3. **What went wrong and how I debugged it**: In my first ranking dataset draft, positives came from all `val` likes (many outside top-120 with low cosine similarity) while negatives were sampled exclusively from FAISS top-120 (high cosine similarity). Despite a `0.806` validation AUC, test `NDCG@10` collapsed to `0.0205` because the ranker learned a *negative* weight on retrieval cosine score! Rebuilding `build_ranking_data.py` around true Stage-1 top-300 session impressions jumped `NDCG@10` to **`0.0840`** (`+25.2%` above retrieval-only).
4. **What I learned**: *Technical*: Selection-bias alignment between candidate generation and ranker training matters more than model depth. *Statistics*: Always check Sample Ratio Mismatch (`SRM p = 0.6729`) and report 95% bootstrap confidence intervals honestly.
5. **What I would improve next**: Train with pairwise/listwise LambdaRank loss directly optimizing NDCG rather than pointwise binary cross-entropy.
6. **Contribution to demo, docs & presentation**: Authored `RESULTS.md`, generated `ab_test_results.png`, built the A/B Ranker Arena (Tab 3), and prepared Slides 6, 7 & 8.

---

### Intern D — Serving Platform, Triton, Redis, Airflow & Cloud Deploy (`Parts 8, 9, 11`)
1. **What I owned**: `src/serving/{api.py, cache.py, triton_client.py, publish.py, ui.py, bundle_export.py}`, `airflow/dags/retrain_dag.py`, `Dockerfile`, `docker-compose.yml`, `public/`, `lib/recsysEngine.mjs`, `vercel.json`, `netlify.toml`, `tests/test_api.py`, and `tests/test_quality_gate.py`.
2. **Technical decision I am proud of**: Exporting a compact `1.86 MB` float16 model + catalog bundle (`deploy_bundle/bundle.json`) paired with a zero-dependency ES-module inference engine (`lib/recsysEngine.mjs`), enabling the exact same repository to run full Docker/Triton/Redis locally AND deploy in 5 seconds on both **Vercel** and **Netlify** without hitting 250 MB Python Lambda limits.
3. **What went wrong and how I debugged it**: When Redis was offline during standalone testing, attempting a TCP socket connection on every `get()` and `set()` added ~80 ms per request. Adding a 15-second monotonic circuit-breaker + in-memory TTL fallback reduced warm miss latency to `2.53 ms` and cache hits to `0.06 ms`.
4. **What I learned**: *Technical*: Embedding `model_version` inside cache keys (`rec:v1:...`) enables zero-downtime model rollouts without `FLUSHALL`. *DevOps*: Automated quality gates (`>= 98%` of prod NDCG & Recall) prevent silent regressions.
5. **What I would improve next**: Add Prometheus latency histograms and Evidently feature-drift alerts wired to Triton's `/metrics` endpoint.
6. **Contribution to demo, docs & presentation**: Built the 5-tab Web UI, Streamlit UI, `API.md`, `DEPLOYMENT_GUIDE.md`, `airflow_dag_run.png`, and Slides 5, 9 & 10.

---

## Part B — Prepared Answers for Review Questions (Part 12.2)

1. **Why a time split and not a random split?**
   A random split lets the model train on a user's 2019 ratings and predict their 2015 ratings, leaking future item popularity bursts and future taste shifts. Production systems only ever predict the future, so a strict chronological `80/10/10` split mirrors reality.
2. **Why approximate search (FAISS HNSW) and what accuracy do we lose?**
   Exhaustive dot-product scales linearly $O(N \cdot d)$, which becomes a bottleneck at tens of millions of items. On our catalog, HNSW retains **99.98%** recall vs exact Flat search, and at 300k vectors delivers a **25.63x QPS speedup** while retaining **98.9%** recall.
3. **How do we prevent training-serving skew?**
   Both `build_ranking_data.py` and `src/serving/api.py` import the exact same `build_dense()` and `load_feature_arrays()` functions from `src/ranking/features.py`, and `pad()` from `src/retrieval/data.py`.
4. **What happens when Redis or Triton is down?**
   `Cache` catches `redis.RedisError`, activates a 15-second circuit-breaker, and falls back to an in-process TTL cache without failing requests. Similarly, `triton_client.py` falls back to exported local SavedModel weights if Triton HTTP is unreachable.
5. **How do we handle a brand-new user or brand-new movie?**
   Brand-new users with zero info get 90-day global popularity; users picking genres get round-robin genre popularity; users picking 1–5 onboarding films get content-based nearest neighbours via averaged Two-Tower item embeddings. Brand-new items appear via genre features in `ItemTower` and cold-start exploration slots.
6. **Is our A/B test real?**
   Honestly: it is an offline **counterfactual replay simulation** over held-out MovieLens likes (`rating >= 4.0`) routed through the real `/recommend` API and MD5 splitter. A live production A/B test would capture real impression position bias, novelty discovery, and session dwell time.
7. **What would we do with 100x more items and users?**
   Shard the FAISS index with `IndexIVFPQ` (Product Quantization) across GPU/CPU replicas, move offline feature pipelines to distributed Spark on EMR/Databricks backed by a Redis/Feast online feature store, and run Triton with dynamic GPU batching + TensorRT/ONNX quantization.
