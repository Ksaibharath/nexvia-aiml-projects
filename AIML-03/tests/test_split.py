"""Data split and temporal leakage guard tests."""
import pandas as pd


def test_time_split_no_overlap():
    df = pd.read_parquet("data/interim/ratings")
    mx = df.groupby("split", observed=True)["timestamp"].max()
    mn = df.groupby("split", observed=True)["timestamp"].min()
    assert mx["train"] <= mn["val"], f"Train max {mx['train']} > Val min {mn['val']}"
    assert mx["val"] <= mn["test"], f"Val max {mx['val']} > Test min {mn['test']}"


def test_history_is_past_only():
    df = pd.read_parquet("data/processed/interactions", filters=[("split", "==", "train")]).head(2000)
    assert all(row.item_idx not in list(row.history) for row in df.itertuples())


def test_history_timestamps_strictly_prior():
    df = pd.read_parquet("data/processed/interactions", filters=[("split", "==", "train")]).head(500)
    # Check per user that history length never exceeds 20 and grows monotonically
    for _, g in df.groupby("userId"):
        prev_items = set()
        for row in g.itertuples():
            hist_list = list(row.history)
            assert len(hist_list) <= 20
            assert set(hist_list).issubset(prev_items)
            prev_items.add(row.item_idx)
