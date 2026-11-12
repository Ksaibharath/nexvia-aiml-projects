import os, json, pandas as pd, sqlalchemy as sa, streamlit as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, networkx as nx
import streamlit.components.v1 as components
engine = sa.create_engine(os.getenv("DB_URL", "postgresql+psycopg2://fraud:fraud@localhost:5432/fraud"))
st.set_page_config(page_title="Fraud analyst console", layout="wide")
def q(sql, **p):
    with engine.connect() as c: return pd.read_sql(sa.text(sql), c, params=p)
def run(sql, **p):
    with engine.begin() as c: c.execute(sa.text(sql), p)
page = st.sidebar.radio("Screen", ["Alert queue", "Monitoring"]); analyst = st.sidebar.text_input("Your name", "analyst1")
if page == "Alert queue":
    k = q("""SELECT count(*) FILTER (WHERE status='new') AS new_alerts,
             count(*) FILTER (WHERE label='fraud') AS confirmed, count(*) FILTER (WHERE label='legit') AS false_pos,
             coalesce(sum(amount) FILTER (WHERE label='fraud'),0) AS fraud_amount FROM alerts""").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("New alerts", int(k.new_alerts)); c2.metric("Confirmed fraud", int(k.confirmed))
    c3.metric("False positives", int(k.false_pos))
    c4.metric("Review precision", f"{k.confirmed / max(k.confirmed + k.false_pos, 1):.0%}")
    status = st.selectbox("Status", ["new", "in_review", "confirmed_fraud", "false_positive", "escalated"])
    df = q("SELECT transaction_id, uid, amount, final_score, xgb_score, gnn_score, ring_size, created_at FROM alerts WHERE status = :s ORDER BY final_score DESC LIMIT 200", s=status)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if len(df):
        tid = st.selectbox("Open case (transaction_id)", df.transaction_id)
        a = q("SELECT * FROM alerts WHERE transaction_id = :t", t=int(tid)).iloc[0]
        left, right = st.columns(2)
        with left:
            st.subheader(f"Case {tid}"); st.write(f"Account `{a.uid}` | amount {a.amount:.2f} | score **{a.final_score:.3f}**")
            st.write(f"XGBoost {a.xgb_score:.3f} | GNN {a.gnn_score:.3f} | ring size {a.ring_size}")
            st.write("**Why flagged (top contributions):**")
            st.table(pd.DataFrame(a.reasons if isinstance(a.reasons, list) else json.loads(a.reasons)))
            notes = st.text_area("Notes", a.notes or "")
            b1, b2, b3, b4 = st.columns(4)
            for col, label, stt, lab in [(b1, "Start review", "in_review", None), (b2, "Confirm fraud", "confirmed_fraud", "fraud"),
                                         (b3, "False positive", "false_positive", "legit"), (b4, "Escalate", "escalated", None)]:
                if col.button(label):
                    run("UPDATE alerts SET status=:s, label=coalesce(:l, label), analyst=:a, notes=:n, updated_at=now() WHERE transaction_id=:t", s=stt, l=lab, a=analyst, n=notes, t=int(tid)); st.rerun()
        with right:
            st.subheader("Fraud ring")
            if a.ring_id is not None and a.ring_id >= 0:
                m = q("SELECT uid, gnn_score FROM rings WHERE ring_id = :r ORDER BY gnn_score DESC LIMIT 40", r=int(a.ring_id))
                e = q("SELECT src, dst FROM ring_edges WHERE src IN (SELECT uid FROM rings WHERE ring_id=:r)", r=int(a.ring_id))
                G = nx.Graph(); G.add_nodes_from(m.uid); G.add_edges_from([(s, d) for s, d in zip(e.src, e.dst) if s in G and d in G])
                fig, ax = plt.subplots(figsize=(5, 4)); nx.draw(G, ax=ax, with_labels=False, node_size=60); st.pyplot(fig)
            else: st.write("No ring for this account.")
else:
    st.subheader("Monitoring")
    lat = q("SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY total_ms) AS p50, percentile_cont(0.95) WITHIN GROUP (ORDER BY total_ms) AS p95, percentile_cont(0.99) WITHIN GROUP (ORDER BY total_ms) AS p99 FROM scores")
    st.write(lat)
    st.dataframe(q("SELECT id, created_at, share_drifted, decision FROM drift_reports ORDER BY id DESC LIMIT 10"), use_container_width=True, hide_index=True)
    st.dataframe(q("SELECT * FROM retrain_log ORDER BY id DESC LIMIT 10"), use_container_width=True, hide_index=True)
    rep = q("SELECT html FROM drift_reports ORDER BY id DESC LIMIT 1")
    if len(rep): components.html(rep.iloc[0].html, height=600, scrolling=True)
