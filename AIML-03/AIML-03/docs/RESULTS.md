# Experimental Results & Performance Benchmarks

All metrics below were measured end-to-end on the **MovieLens 25M** dataset (`sample_users: 5000` dev configuration; set `sample_users: null` in `configs/config.yaml` for the full 162k-user / 25M-rating run) under a **strict chronological 80 / 10 / 10 time split** with zero future leakage.

---

## 1. Environment, Hyperparameters & Training Times

| Component | Setting / Details |
| :--- | :--- |
| **Dataset** | MovieLens 25M (`ratings.csv`, `movies.csv`), positive label = `rating >= 4.0` |
| **Filtering** | Iterative 2-pass 5-core filter (`userId >= 5` ratings, `movieId >= 5` ratings) |
| **Two-Tower Retrieval** | `embedding_dim=64`, `history_len=20`, `batch_size=1024`, `lr=2e-3` (Adam), `temperature=0.07`, partial LogQ correction (`0.45`), early stopping on `val_total_loss` (3 epochs, **30.07 s**) |
| **FAISS Index** | `IndexHNSWFlat` (`M=32`, `efConstruction=200`, `efSearch=128`, L2-normalized inner product) |
| **Stage-2 Rankers** | 50 shared dense features (`build_dense()`), `emb=32`, `deep=(256, 128, 64)`, `batch_size=2048`, `lr=1e-3`, early stopping on `val_auc` (`Wide & Deep`: **20.95 s**, `val_auc=0.7259`; `DCN-V2` w/ 3 Cross layers: **22.88 s**, `val_auc=0.7237`) |

---

## 2. Temporal Split Summary (Zero Date Overlap)

| Split | Ratings Rows | Positive Rows (`>= 4.0`) | Distinct Users | Distinct Movies | Start Timestamp (UTC) | End Timestamp (UTC) |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| **Train (80%)** | 588,429 | 299,903 | 4,248 | 8,839 | `1996-03-20 14:56:58` | `2016-08-01 20:11:20` |
| **Val (10%)** | 73,554 | 37,655 | 550 | 7,086 | `2016-08-01 20:14:14` | `2018-02-07 15:02:54` |
| **Test (10%)** | 73,554 | 35,982 | 548 | 7,288 | `2018-02-07 15:03:02` | `2019-11-21 05:02:56` |

*Verified automatically by `tests/test_split.py::test_time_split_no_overlap` (`max(train) <= min(val)` and `max(val) <= min(test)`).*

---

## 3. Headline Master Results Table

| Metric | Random | Popularity | Retrieval Only (FAISS) | Wide & Deep (Variant A) | DCN-V2 (Variant B) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Recall@10 (Stage 1)** | 0.0004 | 0.0207 | **0.0333** | — | — |
| **Recall@100 (Stage 1)** | 0.0066 | 0.1493 | **0.1610** | — | — |
| **Recall@300 (Stage 1)** | 0.0461 | 0.2694 | **0.2979** | — | — |
| **NDCG@10 (Offline Test)** | 0.0278 | 0.0724 | 0.0671 | **0.0840** `[0.0614, 0.1105]` | **0.0801** `[0.0563, 0.1060]` |
| **Recall@10 (Stage 2)** | 0.0102 | 0.0322 | 0.0293 | **0.0424** | **0.0381** |
| **HitRate@10 (Offline Test)** | 16.85% | 33.15% | 30.34% | **36.52%** | **34.27%** |
| **Proxy CTR (Warm Replay A/B)** | — | — | — | 0.0580 | **0.0956** (`p = 0.0913`) |
| **Latency p50 / p95 (Warm Miss)** | — | — | 0.08 ms | **2.53 ms / 3.84 ms** | **2.53 ms / 3.84 ms** |
| **Latency p50 / p95 (Cache Hit)** | — | — | — | **0.06 ms / 0.10 ms** | **0.06 ms / 0.10 ms** |

![Offline & Online A/B Results](ab_test_results.png)

---

## 4. FAISS Approximate Nearest Neighbour Benchmark

### 4.1 Catalog Index Benchmark (`8,839` 64-d L2-Normalized Movie Vectors, `K=100`)

| FAISS Index | Recall vs Exact@100 | Throughput (QPS) | Latency per Query |
| :--- | ---: | ---: | ---: |
| **Flat (`IndexFlatIP` — Exact)** | 1.0000 (100.00%) | 23,530 qps | 0.0425 ms |
| **HNSW (`IndexHNSWFlat`, `M=32, efSearch=128`)** | **0.9998 (99.98%)** | **28,736 qps** | **0.0348 ms** |
| **IVF (`IndexIVFFlat`, `nlist=94, nprobe=16`)** | 0.9911 (99.11%) | 62,328 qps | 0.0160 ms |

### 4.2 Scaling Demonstration (`300,000` Clustered 64-d Vectors, `500` Queries, `K=100`)

| Index Type | Total Search Time | Throughput (QPS) | Speedup vs Flat | Recall vs Exact@100 |
| :--- | ---: | ---: | ---: | ---: |
| **Exact Flat (`IndexFlatIP`)** | 1.977 s | 253 qps | 1.00x | 100.0% |
| **Approximate HNSW (`IndexHNSWFlat`)** | **0.077 s** | **6,482 qps** | **25.63x** | **98.9%** |

*Note: On ≥16 GB RAM machines, set `FAISS_SCALE_N=5000000` to run the same benchmark on 5 million vectors where linear Flat scan slows down ~16x further while HNSW search remains logarithmic.*

---

## 5. Counterfactual Replay Online A/B Test (`ab_sim.py`)

* **Randomization**: Deterministic user-level MD5 hashing `MD5("exp1:" + str(user_id)) % 2` (`0 → Variant A: Wide & Deep`, `1 → Variant B: DCN-V2`).
* **Sample Ratio Mismatch (SRM) Check**: `N = 1,100` visitors (`Variant A = 557`, `Variant B = 543`), Chi-square `p = 0.6729` (**Pass** — no traffic skew).

| Visitor Cohort | Metric | Variant A (Wide & Deep) | Variant B (DCN-V2) | Diff (`B − A`) | 95% Bootstrap CI | Welch `p`-value |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| **Warm Users Only (`N=179`: `A=88, B=91`)** | Proxy CTR | 0.0580 | **0.0956** | `+0.0376` | `[-0.0031, +0.0840]` | `0.0913` |
| **Warm Users Only (`N=179`: `A=88, B=91`)** | Replay NDCG@10 | 0.0703 | **0.1109** | `+0.0406` | `[-0.0106, +0.0960]` | `0.1267` |
| **Warm Users Only (`N=179`: `A=88, B=91`)** | HitRate@10 | 0.3409 | **0.3956** | `+0.0547` | `[-0.0805, +0.1998]` | `0.4509` |
| **All Visitors (`N=1,100`: incl. Cold-Start)** | Proxy CTR | 0.2390 | **0.2670** | `+0.0281` | `[-0.0029, +0.0602]` | `0.0832` |

---

## 6. Honest Interpretation & Limitations

Both Stage-2 neural rankers (**Wide & Deep** at `0.0840` NDCG@10 and **DCN-V2** at `0.0801` NDCG@10) substantially outperform Stage-1 retrieval order alone (`0.0671` NDCG@10, a **+25.2%** relative improvement) and the strong MovieLens popularity baseline (`0.0724` NDCG@10), placing at least one held-out liked film in the Top-10 for **36.5%** of warm test users. In the counterfactual replay A/B experiment on warm users, Variant B (DCN-V2) achieved higher point estimates on simulated CTR (`9.56%` vs `5.80%`, `+3.76 pp`) and HitRate@10 (`39.6%` vs `34.1%`), though with `p = 0.0913` on the 5,000-user sample (`95% CI [-0.0031, +0.0840]`), the difference between DCN-V2 and Wide & Deep does not cross the strict `α = 0.05` threshold — an expected and honest finding on tabular MovieLens features.

**What these numbers do NOT prove (Limitations):**
1. **Simulated CTR Proxy**: MovieLens contains explicit 0.5–5.0 star ratings rather than real UI impressions and clicks; we define a proxy positive as `rating >= 4.0` in the held-out period.
2. **Survivorship / Exposure Bias in Replay**: Offline counterfactual replay can only reward movies the user actually watched and rated in 2018–2019; truly great novel recommendations that the user never encountered in real life are counted as non-clicks.
3. **Static Batch Features**: User genre affinities and item popularity statistics are computed offline at the end of the training window rather than updated via a streaming real-time feature store.
