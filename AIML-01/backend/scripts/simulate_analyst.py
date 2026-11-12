# Simulate analyst labels from ground truth for demo (real labels arrive late).
import pandas as pd, sqlalchemy as sa, os
engine = sa.create_engine(os.getenv("DB_URL", "postgresql+psycopg2://fraud:fraud@localhost:5432/fraud"))
labels = pd.read_parquet("data/processed/labels.parquet").set_index("transaction_id")
with engine.begin() as c:
    rows = c.execute(sa.text("SELECT transaction_id FROM alerts WHERE label IS NULL")).fetchall()
    for (tid,) in rows:
        lab = "fraud" if int(labels.loc[tid, "isFraud"]) == 1 else "legit"
        c.execute(sa.text("UPDATE alerts SET label=:l, status='confirmed_fraud' WHERE transaction_id=:t"), {"l": lab, "t": tid})
print("simulated", len(rows))
