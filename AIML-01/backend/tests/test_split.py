import pandas as pd
def test_time_split_no_overlap():
    d = pd.read_parquet("data/interim/transactions.parquet")
    mx, mn = d.groupby("split").TransactionDT.max(), d.groupby("split").TransactionDT.min()
    assert mx["train"] <= mn["val"] and mx["val"] <= mn["test"]
