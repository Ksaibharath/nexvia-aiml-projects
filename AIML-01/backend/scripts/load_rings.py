import pandas as pd
from fraud.serving.db import engine
pd.read_parquet("data/processed/rings.parquet").to_sql("rings", engine, if_exists="replace", index=False)
pd.read_parquet("data/processed/ring_edges.parquet").to_sql("ring_edges", engine, if_exists="replace", index=False)
print("rings loaded")
