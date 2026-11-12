"""Build Stage-2 ranking dataset from realistic Stage-1 top-300 candidate impressions in the val split.

Key insight (preventing selection-bias inversion):
At serving time, Stage 2 re-ranks the top-300 candidates returned by FAISS. If positives come from
arbitrary items outside top-300 while negatives come exclusively from top-100, E[cosine | y=1] < E[cosine | y=0]
and the ranker learns an inverted weight on retrieval score. Instead, we generate true Stage-1 top-300
candidate lists at chronologically ordered val checkpoints, label retrieved future val positives as y=1,
and sample unliked candidates from the SAME top-300 pool as y=0. Test data remains 100% untouched.
"""
import pathlib
import faiss
import numpy as np
import pandas as pd

from src.index.build_faiss import retrieve
from src.ranking.features import build_dense, load_feature_arrays
from src.retrieval.data import pad
from src.retrieval.export import infer_user_tower_numpy


def main():
    rng = np.random.RandomState(42)
    val_df = pd.read_parquet("data/processed/interactions", filters=[("split", "==", "val")])
    val_df = val_df[val_df["item_idx"] > 0].sort_values(["userId", "timestamp"]).reset_index(drop=True)

    train_state = pd.read_parquet("data/processed/user_state_train_end").set_index("userId")
    UF, IF, n_g = load_feature_arrays()
    index = faiss.read_index("artifacts/items_hnsw.faiss")
    ut_weights = dict(np.load("artifacts/user_tower_weights.npz"))

    sessions = []
    # Build checkpoints per val user:
    # 1) End-of-train state predicting val positives (for pre-existing warm users)
    # 2) Rolling intra-val checkpoints (after 5, 10, 20 val likes) predicting remaining val positives
    for uid, group in val_df.groupby("userId"):
        uid_int = int(uid)
        items_seq = group["item_idx"].values.astype("int32").tolist()
        ts_seq = group["timestamp"].values.astype("int64").tolist()
        u_idx = int(group["user_idx"].iloc[0])

        if uid_int in train_state.index:
            row = train_state.loc[uid_int]
            base_hist = [int(x) for x in row["history"]]
            seen_col = "seen_items" if "seen_items" in train_state.columns else "history"
            base_seen = set(int(x) for x in row[seen_col])
            if len(base_hist) >= 1 and len(items_seq) >= 1:
                sessions.append((uid_int, u_idx, base_hist[-20:], base_seen, set(items_seq), ts_seq[0]))
        else:
            base_hist = []
            base_seen = set()

        # Rolling checkpoints inside val so ranker sees abundant realistic candidate impressions
        for ckpt in (5, 10, 18, 30, 45):
            if len(items_seq) > ckpt:
                cur_hist = (base_hist + items_seq[:ckpt])[-20:]
                cur_seen = base_seen | set(items_seq[:ckpt])
                future_pos = set(items_seq[ckpt:]) - cur_seen
                if future_pos:
                    sessions.append((uid_int, u_idx, cur_hist, cur_seen, future_pos, ts_seq[ckpt]))

    # Sort sessions chronologically by checkpoint timestamp so last 20% is strict future validation
    sessions.sort(key=lambda x: x[5])

    u_idxs = np.array([s[1] for s in sessions], dtype="int32")
    hists = [s[2] for s in sessions]
    seen_sets = [s[3] for s in sessions]

    q_vecs = infer_user_tower_numpy(ut_weights, u_idxs, pad(hists))
    cand_lists = retrieve(q_vecs, seen_sets, k=300, index=index)

    rows_u, rows_i, rows_s, rows_hl, rows_y, rows_sess = [], [], [], [], [], []
    n_neg_per_pos = 6

    for sess_i, (_, u_idx, hist, _, future_pos, _) in enumerate(sessions):
        cands = cand_lists[sess_i]
        if not cands:
            continue
        pos_cands = [(item_id, sc) for item_id, sc in cands if item_id in future_pos]
        neg_cands = [(item_id, sc) for item_id, sc in cands if item_id not in future_pos]
        if not pos_cands or not neg_cands:
            continue

        hl = float(len(hist))
        for item_id, sc in pos_cands:
            rows_u.append(u_idx)
            rows_i.append(item_id)
            rows_s.append(sc)
            rows_hl.append(hl)
            rows_y.append(1.0)
            rows_sess.append(sess_i)

        n_sample = min(len(neg_cands), len(pos_cands) * n_neg_per_pos)
        chosen_idx = rng.choice(len(neg_cands), size=n_sample, replace=False)
        for ci in chosen_idx:
            item_id, sc = neg_cands[ci]
            rows_u.append(u_idx)
            rows_i.append(item_id)
            rows_s.append(sc)
            rows_hl.append(hl)
            rows_y.append(0.0)
            rows_sess.append(sess_i)

    out_u = np.array(rows_u, dtype="int32")
    out_i = np.array(rows_i, dtype="int32")
    out_s = np.array(rows_s, dtype="float32")
    out_hl = np.array(rows_hl, dtype="float32")
    out_y = np.array(rows_y, dtype="float32")

    dense = build_dense(out_u, out_i, out_s, out_hl, UF, IF, n_g)

    cutoff_idx = int(len(out_y) * 0.80)
    pathlib.Path("artifacts").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        "artifacts/ranking_data.npz",
        tr_user_idx=out_u[:cutoff_idx],
        tr_item_idx=out_i[:cutoff_idx],
        tr_dense=dense[:cutoff_idx],
        tr_y=out_y[:cutoff_idx],
        va_user_idx=out_u[cutoff_idx:],
        va_item_idx=out_i[cutoff_idx:],
        va_dense=dense[cutoff_idx:],
        va_y=out_y[cutoff_idx:],
        n_users=np.array([len(UF) - 1], dtype="int32"),
        n_items=np.array([len(IF) - 1], dtype="int32"),
    )
    print(
        f"Built ranking dataset from {len(sessions):,d} val sessions: "
        f"train={cutoff_idx:,d} rows, val={len(out_y) - cutoff_idx:,d} rows, "
        f"pos_rate={out_y.mean():.3f}, n_dense={dense.shape[1]}"
    )


if __name__ == "__main__":
    main()
