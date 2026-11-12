# Export scores/alerts to CSV for reporting.
import pandas as pd, sqlalchemy as sa, os
engine = sa.create_engine(os.getenv("DB_URL", "postgresql+psycopg2://fraud:fraud@localhost:5432/fraud"))
with engine.connect() as c:
    pd.read_sql(sa.text("SELECT * FROM scores ORDER BY scored_at DESC LIMIT 100000"), c).to_csv("artifacts/scores_export.csv", index=False)
    pd.read_sql(sa.text("SELECT * FROM alerts ORDER BY final_score DESC"), c).to_csv("artifacts/alerts_export.csv", index=False)
print("exported")
