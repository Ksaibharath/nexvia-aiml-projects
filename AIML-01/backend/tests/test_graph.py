import pandas as pd, numpy as np
from fraud.graph.build_graph import build_snapshot
def test_no_hub_edges():
    n = 103
    uids = [f"u{i}" for i in range(n)]
    rows = []
    for i, u in enumerate(uids):
        dev = "hub" if i < 100 else "rare"
        rows.append({"TransactionID": i, "uid": u, "TransactionDT": i, "TransactionAmt": 10.0,
                     "device_sig": dev, "card_key": f"c{i}", "email_key": f"m{i}", "addr1": 1, "isFraud": 0})
    tx = pd.DataFrame(rows)
    snap = build_snapshot(tx, cutoff=50, horizon_end=103, max_group=30)
    # hub (100 accounts) must be pruned: edges only from the 3-account group
    assert snap["edge_index"].shape[1] <= 6
