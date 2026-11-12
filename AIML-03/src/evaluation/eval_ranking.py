"""Evaluate all 5 ranking methods on held-out Test period (NDCG@10, Recall@10, HitRate@10 + 95% bootstrap CI)."""
import json
import pathlib
import faiss
import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k
from src.index.build_faiss import retrieve
from src.ranking.export import infer_ranker_numpy
from src.ranking.features import build_dense, load_feature_arrays
from src.retrieval.data import pad
from src.retrieval.export import infer_user_tower_numpy


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    clean = vals[~np.isnan(vals)]
    if len(clean) == 0:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    means = [float(np.mean(rng.choice(clean, size=len(clean), replace=True))) for _ in range(n_boot)]
    return (round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4))


def evaluate_ranking() -> dict:
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    min_hist = int(cfg.get("min_history_for_warm", 5))
    ret_k = int(cfg.get("retrieval_k", 300))

    state = pd.read_parquet("data/processed/user_state_val_end")
    state = state[state["history"].apply(len) >= min_hist].reset_index(drop=True)

    test_df = pd.read_parquet("data/processed/interactions", filters=[("split", "==", "test")])
    test_df = test_df[test_df["item_idx"] > 0]
    test_by_user = test_df.groupby("userId")["item_idx"].apply(list)

    eval_users = state[state["userId"].isin(test_by_user.index)].copy()
    if len(eval_users) > 5000:
        eval_users = eval_users.sample(5000, random_state=0, replace=False)
    eval_users = eval_users.reset_index(drop=True)

    ut_weights = dict(np.load("artifacts/user_tower_weights.npz"))
    wd_weights = dict(np.load("artifacts/ranker_wd_weights.npz"))
    dcn_weights = dict(np.load("artifacts/ranker_weights.npz"))

    UF, IF, n_g = load_feature_arrays()
    items_df = pd.read_parquet("data/processed/items")
    n_pos_map = dict(zip(items_df["item_idx"].astype(int), items_df["n_pos"].astype(float)))

    index = faiss.read_index("artifacts/items_hnsw.faiss")
    vecs = infer_user_tower_numpy(
        ut_weights,
        eval_users["user_idx"].values.astype("int32"),
        pad(eval_users["history"].values),
    )

    seen_col = "seen_items" if "seen_items" in eval_users.columns else "history"
    seen_sets = [set(int(x) for x in row) for row in eval_users[seen_col].values]
    cands_list = retrieve(vecs, seen_sets, k=ret_k, index=index)

    methods = ["random", "popularity", "retrieval_only", "wide_and_deep", "dcn_v2"]
    per_user = {m: {"ndcg@10": [], "recall@10": [], "hit_rate@10": []} for m in methods}

    rng = np.random.RandomState(42)
    for row_idx, row in eval_users.iterrows():
        uid = int(row["userId"])
        u_idx = int(row["user_idx"])
        hist_len = len(row["history"])
        seen = seen_sets[row_idx]
        rel = [i for i in test_by_user[uid] if i not in seen]
        if not rel:
            continue

        cand = cands_list[row_idx]
        if not cand:
            continue

        ids = np.array([c[0] for c in cand], dtype="int32")
        rs = np.array([c[1] for c in cand], dtype="float32")

        # 1) Random order of candidates
        rand_ids = ids[rng.permutation(len(ids))].tolist()

        # 2) Popularity order of candidates
        pop_scores = np.array([n_pos_map.get(int(i), 0.0) for i in ids])
        pop_ids = ids[np.argsort(-pop_scores)].tolist()

        # 3) Retrieval-only cosine score order
        ret_ids = ids.tolist()

        # Shared dense feature vector for Stage-2 models
        u_arr = np.full(len(ids), u_idx, dtype="int32")
        hl_arr = np.full(len(ids), hist_len, dtype="float32")
        dense = build_dense(u_arr, ids, rs, hl_arr, UF, IF, n_g)

        # 4) Wide & Deep (Variant A)
        wd_scores = infer_ranker_numpy(wd_weights, u_arr, ids, dense).ravel()
        wd_ids = ids[np.argsort(-wd_scores)].tolist()

        # 5) DCN-V2 (Variant B)
        dcn_scores = infer_ranker_numpy(dcn_weights, u_arr, ids, dense).ravel()
        dcn_ids = ids[np.argsort(-dcn_scores)].tolist()

        for name, ranked in [
            ("random", rand_ids),
            ("popularity", pop_ids),
            ("retrieval_only", ret_ids),
            ("wide_and_deep", wd_ids),
            ("dcn_v2", dcn_ids),
        ]:
            per_user[name]["ndcg@10"].append(ndcg_at_k(ranked, rel, 10))
            per_user[name]["recall@10"].append(recall_at_k(ranked, rel, 10))
            per_user[name]["hit_rate@10"].append(hit_rate_at_k(ranked, rel, 10))

    summary = {}
    print(f"\nStage-2 Ranking Offline Evaluation ({len(per_user['dcn_v2']['ndcg@10'])} warm test users)")
    print(f"{'Method':<18} {'NDCG@10':>10} {'95% CI':>18} {'Recall@10':>11} {'HitRate@10':>12}")
    print("-" * 74)
    for m in methods:
        ndcg_arr = np.array(per_user[m]["ndcg@10"], dtype="float64")
        rec_arr = np.array(per_user[m]["recall@10"], dtype="float64")
        hr_arr = np.array(per_user[m]["hit_rate@10"], dtype="float64")
        ci_lo, ci_hi = _bootstrap_ci(ndcg_arr)
        summary[m] = {
            "ndcg@10": round(float(np.nanmean(ndcg_arr)), 4),
            "ndcg@10_ci95": [ci_lo, ci_hi],
            "recall@10": round(float(np.nanmean(rec_arr)), 4),
            "hit_rate@10": round(float(np.nanmean(hr_arr)), 4),
        }
        print(
            f"{m:<18} {summary[m]['ndcg@10']:>10.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
            f"{summary[m]['recall@10']:>11.4f} {summary[m]['hit_rate@10']:>12.4f}"
        )

    pathlib.Path("artifacts").mkdir(parents=True, exist_ok=True)
    metrics_path = pathlib.Path("artifacts/metrics.json")
    existing = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    existing["ranking"] = summary
    existing["ndcg@10"] = summary["dcn_v2"]["ndcg@10"]
    existing["ndcg@10_wd"] = summary["wide_and_deep"]["ndcg@10"]
    existing["hit_rate@10"] = summary["dcn_v2"]["hit_rate@10"]
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    evaluate_ranking()
