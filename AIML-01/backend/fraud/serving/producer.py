import argparse, json, time, pandas as pd
from confluent_kafka import Producer
ap = argparse.ArgumentParser()
ap.add_argument("--rate", type=float, default=50)
ap.add_argument("--limit", type=int, default=20000)
ap.add_argument("--bootstrap", default="localhost:9092")
ap.add_argument("--drift-after", type=int, default=-1)
a = ap.parse_args()
df = pd.read_parquet("data/processed/labels.parquet")
df = df[df.split == "test"].sort_values("TransactionDT").head(a.limit)
p = Producer({"bootstrap.servers": a.bootstrap, "linger.ms": 5})
gap = 1.0 / a.rate; t_next = time.perf_counter()
for i, r in enumerate(df.itertuples(), 1):
    msg = {"transaction_id": int(r.transaction_id), "uid": r.uid, "event_timestamp": r.event_timestamp.isoformat(),
           "amount": float(r.TransactionAmt)}
    if 0 <= a.drift_after < i: msg["sim_shift"] = {"amt_log": 1.5, "amt_z": 3.0, "sum_24h": 800.0}
    p.produce("transactions", key=r.uid, value=json.dumps(msg)); p.poll(0)
    t_next += gap; time.sleep(max(0, t_next - time.perf_counter()))
    if i % 1000 == 0: print("sent", i)
p.flush(); print("done")
