"""Shared dense feature builder imported by BOTH offline ranker training and online API serving."""
import numpy as np
import pandas as pd


def load_feature_arrays():
    items = pd.read_parquet("data/processed/items")
    items = items[items["item_idx"] > 0]
    gc = sorted([c for c in items.columns if c.startswith("g_")])

    users = pd.read_parquet("data/processed/user_features")
    users = users[users["user_idx"] > 0]
    uac = ["ua_" + c for c in gc]

    IF = np.zeros((int(items["item_idx"].max()) + 1, 5 + len(gc)), dtype="float32")
    IF[items["item_idx"].values] = np.column_stack([
        np.log1p(items["n_ratings"].values),
        items["avg_rating"].values,
        np.log1p(items["n_pos"].values),
        np.log1p(items["n_pos_recent"].values),
        (items["year"].fillna(1990).values - 1990.0) / 30.0,
        items[gc].values,
    ]).astype("float32")

    UF = np.zeros((int(users["user_idx"].max()) + 1, 2 + len(gc)), dtype="float32")
    UF[users["user_idx"].values] = np.column_stack([
        np.log1p(users["u_n"].values),
        users["u_avg"].values,
        users[uac].values,
    ]).astype("float32")

    return UF, IF, len(gc)


def build_dense(u_idx, i_idx, scores, hist_len, UF, IF, n_g) -> np.ndarray:
    """ONE function, used by training AND serving. All array inputs have equal length N."""
    u_idx = np.clip(np.asarray(u_idx, dtype="int32"), 0, len(UF) - 1)
    i_idx = np.clip(np.asarray(i_idx, dtype="int32"), 0, len(IF) - 1)
    u = UF[u_idx]
    it = IF[i_idx]
    cross = (u[:, -n_g:] * it[:, -n_g:]).sum(axis=1, keepdims=True)  # genre match
    return np.hstack([
        u,
        it,
        np.asarray(scores, dtype="float32").reshape(-1, 1),
        np.log1p(np.asarray(hist_len, dtype="float32")).reshape(-1, 1),
        cross,
    ]).astype("float32")
