import os, sqlalchemy as sa
engine = sa.create_engine(os.getenv("DB_URL", "postgresql+psycopg2://fraud:fraud@localhost:5432/fraud"),
                          pool_size=5, pool_pre_ping=True)
def save(res):
    with engine.begin() as c:
        c.execute(sa.text("""INSERT INTO scores (transaction_id, uid, event_ts, amount, xgb_score, gnn_score,
              final_score, flagged, feast_ms, total_ms, features, model_version)
              VALUES (:transaction_id, :uid, :event_ts, :amount, :xgb, :gnn, :final, :flagged, :feast_ms,
              :total_ms, CAST(:features AS JSONB), :model_version) ON CONFLICT (transaction_id) DO NOTHING"""),
              {**res, "features": res["features_json"]})
        if res["flagged"]:
            c.execute(sa.text("""INSERT INTO alerts (transaction_id, uid, event_ts, amount, xgb_score, gnn_score,
                  final_score, ring_id, ring_size, reasons) VALUES (:transaction_id, :uid, :event_ts, :amount, :xgb, :gnn,
                            :final, :ring_id, :ring_size, CAST(:reasons AS JSONB)) ON CONFLICT (transaction_id) DO NOTHING"""), res)
