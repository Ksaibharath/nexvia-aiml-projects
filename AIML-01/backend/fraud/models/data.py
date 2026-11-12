import pandas as pd
from feast import FeatureStore
from fraud.features.feature_list import FEATURES, TXN_REFS
def load_frame(cache="data/processed/frame.parquet", refresh=False):
    import os
    if os.path.exists(cache) and not refresh: return pd.read_parquet(cache)
    store = FeatureStore(repo_path="feature_repo")
    labels = pd.read_parquet("data/processed/labels.parquet")
    feats = store.get_historical_features(entity_df=labels[["transaction_id", "event_timestamp"]],
                                          features=TXN_REFS).to_df()
    df = labels.merge(feats, on=["transaction_id", "event_timestamp"], how="left")
    df.to_parquet(cache, index=False)
    return df
