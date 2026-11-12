import pandas as pd
from feast import FeatureStore
from fraud.features.feature_list import FEATURES, TXN_REFS
def test_offline_equals_online():
    store = FeatureStore(repo_path="feature_repo")
    labels = pd.read_parquet("data/processed/labels.parquet")
    s = labels[labels.split == "test"].sample(200, random_state=0)
    off = (store.get_historical_features(entity_df=s[["transaction_id", "event_timestamp"]], features=TXN_REFS)
           .to_df().set_index("transaction_id").sort_index()[FEATURES])
    rows = [{"transaction_id": int(t)} for t in s.transaction_id]
    on = (store.get_online_features(features=TXN_REFS, entity_rows=rows)
          .to_df().set_index("transaction_id").sort_index()[FEATURES])
    pd.testing.assert_frame_equal(off.astype("float64"), on.astype("float64"), check_dtype=False, atol=1e-5)
