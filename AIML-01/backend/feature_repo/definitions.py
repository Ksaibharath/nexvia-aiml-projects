from datetime import timedelta
from feast import Entity, FeatureView, FeatureService, Field, FileSource
from feast.types import Float32, Int64
from fraud.features.feature_list import FEATURES
txn = Entity(name="transaction", join_keys=["transaction_id"])
txn_src = FileSource(path="data/txn_features.parquet", timestamp_field="event_timestamp")
txn_fv = FeatureView(name="txn_features", entities=[txn], ttl=timedelta(days=400),
                     schema=[Field(name=f, dtype=Float32) for f in FEATURES], source=txn_src)
txn_v1 = FeatureService(name="txn_v1", features=[txn_fv])
account = Entity(name="account", join_keys=["uid"])
graph_src = FileSource(path="data/account_graph.parquet", timestamp_field="event_timestamp")
graph_fv = FeatureView(name="account_graph", entities=[account], ttl=timedelta(days=400),
    schema=[Field(name="gnn_score", dtype=Float32), Field(name="ring_id", dtype=Int64),
            Field(name="ring_size", dtype=Int64)], source=graph_src)
