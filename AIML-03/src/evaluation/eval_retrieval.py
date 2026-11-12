"""Evaluate Stage-1 Retrieval (Two-Tower + FAISS HNSW) vs Popularity and Random baselines on Test split."""
import json
import pathlib
import faiss
import numpy as np
import pandas as pd

from src.evaluation.metrics import mean_metric, recall_at_k
from src.index.build_faiss import retrieve
from src.retrieval.data import pad
from src.retrieval.export import infer_user_tower_numpy


def evaluate_retrieval() -> dict:
    state = pd.read_parquet("data/processed/user_state_val_end")
    state = state[state["user_idx"] > 0].reset_index(drop=True)

    test_df = pd.read_parquet("data/processed/interactions", filters=[("split", "==", "test")])
    test_df = test_df[test_df["item_idx"] > 0]
    test_by_user = test_df.groupby("userId")["item_idx"].apply(list)

    eval_users = state[state["userId"].isin(test_by_user.index)].copy()
    if len(eval_users) > 10_000:
        eval_users = eval_users.sample(10_000, random_state=0, replace=False)
    eval_users = eval_users.reset_index(drop=True)

    weights = dict(np.load("artifacts/user_tower_weights.npz"))
    vecs = infer_user_tower_numpy(
        weights,
        eval_users["user_idx"].values.astype("int32"),
        pad(eval_users["history"].values),
    )

    seen_col = "seen_items" if "seen_items" in eval_users.columns else "history"
    seen_sets = [set(int(x) for x in row) for row in eval_users[seen_col].values]

    index = faiss.read_index("artifacts/items_hnsw.faiss")
    cands_tt = retrieve(vecs, seen_sets, k=300, index=index)
    tt_lists = [[i for i, _ in c] for c in cands_tt]

    items_df = pd.read_parquet("data/processed/items").sort_values("n_pos", ascending=False)
    pop_order = items_df[items_df["item_idx"] > 0]["item_idx"].astype(int).tolist()
    all_item_ids = items_df[items_df["item_idx"] > 0]["item_idx"].astype(int).values

    pop_lists = []
    rand_lists = []
    rng = np.random.RandomState(42)
    for seen in seen_sets:
        pop_lists.append([i for i in pop_order if i not in seen][:300])
        pool = [int(i) for i in rng.choice(all_item_ids, size=min(450, len(all_item_ids)), replace=False) if int(i) not in seen]
        rand_lists.append(pool[:300])

    relevant = [[i for i in test_by_user[u] if i not in seen] for u, seen in zip(eval_users["userId"].values, seen_sets)]
    # Keep users who have at least 1 unseen positive item in test
    valid_mask = [len(r) > 0 for r in relevant]
    tt_lists = [x for x, m in zip(tt_lists, valid_mask) if m]
    pop_lists = [x for x, m in zip(pop_lists, valid_mask) if m]
    rand_lists = [x for x, m in zip(rand_lists, valid_mask) if m]
    relevant = [x for x, m in zip(relevant, valid_mask) if m]

    results = {"n_eval_users": len(relevant), "two_tower": {}, "popularity": {}, "random": {}}
    print(f"Evaluating retrieval on {len(relevant)} warm test users...")
    print(f"{'K':>5}  {'Random':>10}  {'Popularity':>12}  {'Two-Tower':>12}")
    print("-" * 46)
    for K in (10, 50, 100, 300):
        r_rand = mean_metric(recall_at_k, rand_lists, relevant, K)
        r_pop = mean_metric(recall_at_k, pop_lists, relevant, K)
        r_tt = mean_metric(recall_at_k, tt_lists, relevant, K)
        results["random"][f"recall@{K}"] = round(r_rand, 4)
        results["popularity"][f"recall@{K}"] = round(r_pop, 4)
        results["two_tower"][f"recall@{K}"] = round(r_tt, 4)
        print(f"{K:>5}  {r_rand:>10.4f}  {r_pop:>12.4f}  {r_tt:>12.4f}")

    pathlib.Path("artifacts").mkdir(parents=True, exist_ok=True)
    metrics_path = pathlib.Path("artifacts/metrics.json")
    existing = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    existing["retrieval"] = results
    existing["recall@100"] = results["two_tower"]["recall@100"]
    existing["recall@300"] = results["two_tower"]["recall@300"]
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    evaluate_retrieval()
