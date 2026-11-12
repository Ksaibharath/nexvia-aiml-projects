import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
NODE_COLS = ["n_txn", "amt_sum", "amt_mean", "amt_max", "n_dev", "n_cards", "n_addr", "age_days", "recency_days"]
def build_snapshot(tx, cutoff, horizon_end, max_group=30):
    past = tx[tx.TransactionDT <= cutoff]
    n = past.groupby("uid").agg(n_txn=("TransactionID", "size"), amt_sum=("TransactionAmt", "sum"),
            amt_mean=("TransactionAmt", "mean"), amt_max=("TransactionAmt", "max"),
            n_dev=("device_sig", "nunique"), n_cards=("card_key", "nunique"), n_addr=("addr1", "nunique"),
            first=("TransactionDT", "min"), last=("TransactionDT", "max"))
    n["age_days"] = (cutoff - n["first"]) / 86400.0
    n["recency_days"] = (cutoff - n["last"]) / 86400.0
    n = n.drop(columns=["first", "last"])
    fut = tx[(tx.TransactionDT > cutoff) & (tx.TransactionDT <= horizon_end)].groupby("uid").isFraud.max()
    n["active"] = n.index.isin(fut.index)
    n["label"] = n.index.map(fut).fillna(0).astype(int)
    idx = pd.Series(np.arange(len(n)), index=n.index)
    parts = []
    for col, tag in [("device_sig", "dev"), ("card_key", "card"), ("email_key", "mail")]:
        e = past[["uid", col]].dropna().drop_duplicates().rename(columns={col: "key"})
        e["key"] = tag + ":" + e.key.astype(str); parts.append(e)
    ent = pd.concat(parts)
    size = ent.groupby("key").uid.transform("nunique")
    ent = ent[(size >= 2) & (size <= max_group)]
    src, dst = [], []
    for _, accts in ent.groupby("key").uid:
        ids = idx[accts.values].values
        i, j = np.triu_indices(len(ids), k=1)
        src.append(ids[i]); dst.append(ids[j])
    if src:
        s, d = np.concatenate(src), np.concatenate(dst)
        pairs = np.unique(np.stack([np.minimum(s, d), np.maximum(s, d)], 1), axis=0)
        edge_index = np.concatenate([pairs.T, pairs.T[::-1]], axis=1)
    else:
        edge_index = np.zeros((2, 0), dtype=int)
    A = coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(len(n), len(n)))
    _, comp = connected_components(A, directed=False)
    size_c = np.bincount(comp)
    n["ring_id"] = np.where(size_c[comp] >= 3, comp, -1)
    n["ring_size"] = np.where(size_c[comp] >= 3, size_c[comp], 0)
    print(f"snapshot nodes={len(n)} edges={edge_index.shape[1]//2} rings={(n.ring_id>=0).sum()}")
    return {"nodes": n, "edge_index": edge_index}
