CREATE TABLE IF NOT EXISTS scores (
  transaction_id BIGINT PRIMARY KEY, uid TEXT, event_ts TIMESTAMP, amount REAL,
  xgb_score REAL, gnn_score REAL, final_score REAL, flagged BOOLEAN,
  feast_ms REAL, total_ms REAL, features JSONB, model_version TEXT, scored_at TIMESTAMP DEFAULT now());
CREATE INDEX IF NOT EXISTS scores_time ON scores (scored_at DESC);
CREATE TABLE IF NOT EXISTS alerts (
  transaction_id BIGINT PRIMARY KEY, uid TEXT, event_ts TIMESTAMP, amount REAL,
  xgb_score REAL, gnn_score REAL, final_score REAL, ring_id BIGINT, ring_size INT, reasons JSONB,
  status TEXT DEFAULT 'new', label TEXT, analyst TEXT, notes TEXT, created_at TIMESTAMP DEFAULT now(), updated_at TIMESTAMP DEFAULT now());
CREATE INDEX IF NOT EXISTS alerts_status ON alerts (status, final_score DESC);
CREATE TABLE IF NOT EXISTS rings (ring_id BIGINT, uid TEXT, ring_size INT, gnn_score REAL);
CREATE TABLE IF NOT EXISTS ring_edges (src TEXT, dst TEXT);
CREATE TABLE IF NOT EXISTS drift_reports (id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT now(), share_drifted REAL, decision TEXT, html TEXT);
CREATE TABLE IF NOT EXISTS retrain_log (id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT now(), reason TEXT, champion_ap REAL, challenger_ap REAL, promoted BOOLEAN);
