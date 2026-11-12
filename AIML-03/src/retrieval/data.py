"""Data loading and sequence padding utilities for the Two-Tower retrieval model."""
import numpy as np
import pandas as pd

HIST = 20


def pad(seqs, L: int = HIST) -> np.ndarray:
    out = np.zeros((len(seqs), L), dtype=np.int32)  # 0 = padding
    for i, s in enumerate(seqs):
        s_list = [int(x) for x in list(s) if int(x) > 0][-L:]
        if s_list:
            out[i, -len(s_list):] = s_list
    return out


def load_items():
    items = pd.read_parquet("data/processed/items").sort_values("item_idx")
    items = items[items["item_idx"] > 0].reset_index(drop=True)
    gcols = [c for c in items.columns if c.startswith("g_")]
    max_idx = int(items["item_idx"].max())
    G = np.zeros((max_idx + 1, len(gcols)), dtype="float32")  # row 0 = zeros
    G[items["item_idx"].values] = items[gcols].values.astype("float32")
    return items, G


def load_split(split: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet("data/processed/interactions", filters=[("split", "==", split)])
    df = df[(df["user_idx"] > 0) & (df["item_idx"] > 0)].reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=0).reset_index(drop=True)
    return df


def to_dataset(df: pd.DataFrame, batch: int = 2048, shuffle: bool = False):
    import tensorflow as tf

    d = {
        "user_idx": df["user_idx"].values.astype("int32"),
        "item_idx": df["item_idx"].values.astype("int32"),
        "history": pad(df["history"].values),
    }
    ds = tf.data.Dataset.from_tensor_slices(d)
    if shuffle:
        ds = ds.shuffle(min(200_000, max(len(df), 1)), seed=0, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)
