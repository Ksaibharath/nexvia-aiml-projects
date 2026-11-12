import pandas as pd
from fraud.features import build_features
def test_velocity_counts_exclude_current():
    t0 = pd.Timestamp("2018-01-01 10:00")
    tx = pd.DataFrame({"TransactionID": [1, 2, 3], "uid": ["a", "a", "a"],
                       "event_timestamp": [t0, t0 + pd.Timedelta("30min"), t0 + pd.Timedelta("90min")],
                       "TransactionAmt": [10.0, 20.0, 30.0]})
    out = build_features.add_velocity(tx)
    assert out.sort_values("TransactionID").cnt_1h.tolist() == [0, 1, 1]
