import json, numpy as np, pandas as pd
from fraud.features.feature_list import CAT, M_COLS, FEATURES

def add_velocity(tx):
    # Per-account rolling windows (closed='left' excludes current row).
    # Done group-by-group so index alignment is exact across pandas versions.
    tx = tx.copy()
    tx["_orig_idx"] = tx.index
    parts = []
    for _, g in tx.groupby("uid", sort=False):
        g = g.sort_values("event_timestamp").set_index("event_timestamp")
        for name in ["1h", "24h", "7d"]:
            r = g["TransactionAmt"].rolling(name, closed="left")
            g[f"cnt_{name}"] = r.count().fillna(0)
            if name == "24h":
                g["sum_24h"] = r.sum().fillna(0)
        parts.append(g.reset_index())
    out = pd.concat(parts).sort_values("_orig_idx")
    out = out.set_index("_orig_idx")
    out.index.name = tx.index.name
    return out

def build(tx):
    train = tx.split == "train"
    for c in CAT:
        vc = tx.loc[train, c].value_counts(normalize=True)
        tx[c + "_freq"] = tx[c].map(vc).fillna(0).astype("float32")
    for m in M_COLS:
        tx[m] = tx[m].map({"T": 1.0, "F": 0.0}).astype("float32")
    tx["amt_log"] = np.log1p(tx.TransactionAmt)
    tx["hour"] = (tx.TransactionDT // 3600) % 24
    tx["dow"] = (tx.TransactionDT // 86400) % 7
    tx = tx.sort_values(["uid", "TransactionDT"])
    g = tx.groupby("uid")
    tx["n_past"] = g.cumcount()
    tx["time_since_last"] = tx.TransactionDT - g.TransactionDT.shift(1)
    a = tx.TransactionAmt.astype("float64")
    cs = a.groupby(tx.uid).cumsum() - a
    cs2 = (a ** 2).groupby(tx.uid).cumsum() - a ** 2
    n = tx.n_past.replace(0, np.nan)
    mean = cs / n
    std = np.sqrt(np.clip(cs2 / n - mean ** 2, 0, None))
    tx["amt_mean_past"], tx["amt_std_past"] = mean, std
    tx["amt_z"] = (a - mean) / (std + 1.0)
    tx["amt_ratio"] = a / (mean + 1e-6)
    dev = tx.device_sig.fillna("none")
    prev_dev = dev.groupby(tx.uid).shift(1)
    tx["device_changed"] = (prev_dev.notna() & (prev_dev != dev)).astype("float32")
    addr = tx.addr1.fillna(-1)
    prev_addr = addr.groupby(tx.uid).shift(1)
    tx["addr_changed"] = (prev_addr.notna() & (prev_addr != addr)).astype("float32")
    em = tx.P_emaildomain.fillna("none")
    prev_em = em.groupby(tx.uid).shift(1)
    tx["email_changed"] = (prev_em.notna() & (prev_em != em)).astype("float32")
    first_real = (~pd.DataFrame({"uid": tx.uid, "dev": dev}).duplicated()) & (dev != "none")
    tx["n_devices"] = first_real.astype(int).groupby(tx.uid).cumsum()
    tx = add_velocity(tx)
    return tx

def main():
    tx = pd.read_parquet("data/interim/transactions.parquet")
    tx = build(tx)
    out = tx[["TransactionID", "event_timestamp"] + FEATURES].rename(columns={"TransactionID": "transaction_id"})
    out[FEATURES] = out[FEATURES].astype("float32")
    out["transaction_id"] = out.transaction_id.astype("int64")
    out.sort_values("event_timestamp").to_parquet("feature_repo/data/txn_features.parquet", index=False)
    labels = tx[["TransactionID", "uid", "isFraud", "TransactionDT", "TransactionAmt", "event_timestamp", "split"]]
    labels = labels.rename(columns={"TransactionID": "transaction_id"}).sort_values("TransactionDT")
    labels.to_parquet("data/processed/labels.parquet", index=False)
    print(out.shape, out[FEATURES].isna().mean().sort_values(ascending=False).head(8).to_dict())

if __name__ == "__main__":
    main()
