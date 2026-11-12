import json, sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import average_precision_score
from fraud.graph.build_graph import build_snapshot, NODE_COLS
NO_GRAPH = "--no-graph" in sys.argv
torch.manual_seed(0); np.random.seed(0)
cut = json.load(open("artifacts/cutoffs.json")); start = pd.Timestamp(cut["start"])
tx = pd.read_parquet("data/interim/transactions.parquet")
SNAP = {"a": (cut["t40"], cut["t70"]), "b": (cut["t70"], cut["t85"]), "c": (cut["t85"], cut["t100"])}
S = {k: build_snapshot(tx, c, h) for k, (c, h) in SNAP.items()}
def feats(nodes):
    X = nodes[NODE_COLS].astype("float64").copy()
    for c in ["n_txn", "amt_sum", "amt_mean", "amt_max", "n_dev", "n_cards", "n_addr"]: X[c] = np.log1p(X[c])
    return X.values
Xa = feats(S["a"]["nodes"]); mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
def tensors(k):
    s = S[k]; x = torch.tensor((feats(s["nodes"]) - mu) / sd, dtype=torch.float32)
    ei = torch.zeros((2, 0), dtype=torch.long) if NO_GRAPH else torch.tensor(s["edge_index"], dtype=torch.long)
    y = torch.tensor(s["nodes"].label.values, dtype=torch.float32)
    m = torch.tensor(s["nodes"].active.values)
    return x, ei, y, m
T = {k: tensors(k) for k in S}
class SAGE(torch.nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.c1, self.c2 = SAGEConv(d, h), SAGEConv(h, h); self.out = torch.nn.Linear(h, 1)
        self.skip = torch.nn.Linear(d, h)
    def forward(self, x, ei):
        h = F.relu(self.c1(x, ei) + self.skip(x)); h = F.dropout(h, 0.3, self.training)
        h = F.relu(self.c2(h, ei)); return self.out(h).squeeze(-1)
model = SAGE(Xa.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
x, ei, y, m = T["a"]
pw = torch.tensor(float((y[m] == 0).sum() / max((y[m] == 1).sum(), 1)) ** 0.5)
def ap(k):
    model.eval()
    with torch.no_grad(): p = torch.sigmoid(model(T[k][0], T[k][1]))
    mk = T[k][3]; return average_precision_score(T[k][2][mk].numpy(), p[mk].numpy())
best, best_state = -1, None
for ep in range(1, 301):
    model.train(); opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(model(x, ei)[m], y[m], pos_weight=pw); loss.backward(); opt.step()
    if ep % 5 == 0:
        v = ap("b")
        if v > best: best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
        if ep % 25 == 0: print(f"epoch {ep} loss {loss.item():.4f} val_AP(b) {v:.4f}")
model.load_state_dict(best_state)
print(f"{'NO-GRAPH' if NO_GRAPH else 'GRAPH'} val AP (snapshot b) = {best:.4f} | test AP (snapshot c) = {ap('c'):.4f}")
if NO_GRAPH: sys.exit()
rows = []
model.eval()
for k in S:
    with torch.no_grad(): p = torch.sigmoid(model(T[k][0], T[k][1])).numpy()
    n = S[k]["nodes"]
    rows.append(pd.DataFrame({"uid": n.index, "gnn_score": p.astype("float32"), "ring_id": n.ring_id.astype("int64").values,
                              "ring_size": n.ring_size.astype("int64").values,
                              "event_timestamp": start + pd.to_timedelta(SNAP[k][0], unit="s"), "snapshot": k}))
allg = pd.concat(rows); allg.to_parquet("data/processed/account_graph_all.parquet", index=False)
allg[allg.snapshot.isin(["b", "c"])].drop(columns="snapshot").to_parquet("feature_repo/data/account_graph.parquet", index=False)
sc = S["c"]; nodes = sc["nodes"].assign(gnn_score=rows[2].gnn_score.values)
rings = nodes[nodes.ring_id >= 0].reset_index().rename(columns={"index": "uid"})[["uid", "ring_id", "ring_size", "gnn_score"]]
rings.to_parquet("data/processed/rings.parquet", index=False)
ei_c = sc["edge_index"]; keep = np.isin(ei_c[0], np.where(nodes.ring_id.values >= 0)[0])
uids = nodes.index.values
pd.DataFrame({"src": uids[ei_c[0][keep]], "dst": uids[ei_c[1][keep]]}).to_parquet("data/processed/ring_edges.parquet", index=False)
torch.save(model.state_dict(), "artifacts/gnn.pt"); print("rings:", rings.ring_id.nunique())
