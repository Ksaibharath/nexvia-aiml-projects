import json, os, threading, time, logging, pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from confluent_kafka import Consumer, Producer
from fraud.serving.scoring import Scorer
from fraud.serving import db
log = logging.getLogger("fraud"); BOOT = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
state = {"scorer": None, "ready": False}; stop = threading.Event()
def consume():
    c = Consumer({"bootstrap.servers": BOOT, "group.id": "fraud-scorers", "auto.offset.reset": "earliest",
                  "enable.auto.commit": False})
    c.subscribe(["transactions"]); out = Producer({"bootstrap.servers": BOOT}); last_reload = time.time()
    while not stop.is_set():
        msg = c.poll(1.0)
        if time.time() - last_reload > 60:
            try:
                if state["scorer"].load_champion(): log.info("loaded new champion %s", state["scorer"].version)
            except Exception: log.exception("champion check failed")
            last_reload = time.time()
        if msg is None: continue
        if msg.error(): log.error(msg.error()); continue
        try:
            tx = json.loads(msg.value())
            res = state["scorer"].score(tx["transaction_id"], tx["uid"], tx.get("sim_shift"))
            res.update(event_ts=tx["event_timestamp"], amount=tx["amount"])
            db.save(res)
            out.produce("scores", key=str(res["transaction_id"]),
                        value=json.dumps({k: res[k] for k in ("transaction_id", "final", "flagged", "total_ms")}))
            out.poll(0); c.commit(msg)
        except Exception as e:
            log.exception("scoring failed"); out.produce("transactions.dlq", value=msg.value()); out.poll(0); c.commit(msg)
@asynccontextmanager
async def lifespan(app):
    state["scorer"] = Scorer(); state["ready"] = True
    t = threading.Thread(target=consume, daemon=True); t.start()
    yield
    stop.set()
app = FastAPI(title="Fraud scoring API", lifespan=lifespan)
class Tx(BaseModel):
    transaction_id: int
    uid: str
@app.get("/live")
def live(): return {"status": "alive"}
@app.get("/health")
def health():
    return {"status": "ok" if state["ready"] else "starting", "model_version": getattr(state["scorer"], "version", None)}
@app.post("/score")
def score(tx: Tx):
    r = state["scorer"].score(tx.transaction_id, tx.uid)
    return {k: r[k] for k in ("transaction_id", "xgb", "gnn", "final", "flagged", "reasons", "feast_ms", "total_ms", "model_version")}
