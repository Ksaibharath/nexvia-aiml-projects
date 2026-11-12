"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { scoreTransaction, generateTransaction, rng, evaluate, SERVING } from "@/lib/model";
import { MODEL_COMPARISON, EDA, PIPELINE } from "@/lib/data";

type TxnRow = {
  transaction_id: number; uid: string; amount: number; xgb: number; gnn: number; final: number;
  flagged: boolean; ring_id: number | null; ring_size: number;
  reasons: { feature: string; value: any; impact: number }[];
  total_ms: number; model_version: string; isFraud?: number;
  merchant: string; category: string; event_timestamp?: string;
  status: "approved" | "review" | "blocked"; acked?: boolean;
};

const NAV = [
  { id: "home", icon: "🏠", label: "Home" },
  { id: "pay", icon: "💸", label: "Send Money" },
  { id: "txns", icon: "🧾", label: "Transactions" },
  { id: "cards", icon: "💳", label: "Cards" },
  { id: "alerts", icon: "🔔", label: "Alerts & Security" },
  { id: "how", icon: "💡", label: "How It Works" },
] as const;
type PageId = (typeof NAV)[number]["id"];

const MERCHANTS = ["Swiggy", "Amazon", "Zomato", "Netflix", "Uber", "DMart", "IRCTC", "Electricity Bill", "Mobile Recharge", "Salary Credit", "UPI Transfer", "ATM Withdrawal"];
const CATS = ["Food", "Shopping", "Travel", "Bills", "Transfer", "Cash"];

function fmt(n: number) { return n.toLocaleString("en-US"); }
function money(n: number) { return "₹" + n.toLocaleString("en-US", { maximumFractionDigits: 2 }); }
function Bar({ v, color }: { v: number; color: string }) {
  return <div className="bar"><i style={{ width: `${Math.min(100, v * 100)}%`, background: color }} /></div>;
}
function riskBadge(final: number) {
  if (final >= 0.85) return <span className="badge red">High risk</span>;
  if (final >= 0.65) return <span className="badge amber">Review</span>;
  if (final >= 0.35) return <span className="badge blue">Check</span>;
  return <span className="badge green">Safe</span>;
}
function statusBadge(s: string) {
  if (s === "blocked") return <span className="badge red">⛔ Blocked</span>;
  if (s === "review") return <span className="badge amber">⏳ Under review</span>;
  return <span className="badge green">✓ Approved</span>;
}
function plainReason(f: string) {
  const map: Record<string, string> = {
    amt_z: "Amount is much higher than your usual spending",
    amt_ratio: "Amount is much higher than your usual spending",
    cnt_1h: "Many transactions in a short time (possible card misuse)",
    cnt_24h: "Unusual number of transactions today",
    device_changed: "New device / phone used",
    P_emaildomain_freq: "Unusual email domain",
    n_past: "New account with little history",
    sum_24h: "High total spending in last 24 hours",
    addr_changed: "New location / address",
    email_changed: "Email changed recently",
    hour: "Transaction at an unusual hour",
    dist1: "Unusual distance from usual location",
    C1: "Unusual card activity pattern",
    D1: "Unusual time gap between transactions",
    TransactionAmt: "Large transaction amount",
  };
  return map[f] ?? `Unusual pattern: ${f}`;
}

export default function Page() {
  const [page, setPage] = useState<PageId>("home");
  const [txns, setTxns] = useState<TxnRow[]>([]);
  const [seed, setSeed] = useState(1);
  const [streaming, setStreaming] = useState(false);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [selected, setSelected] = useState<TxnRow | null>(null);
  const [cardFrozen, setCardFrozen] = useState(false);
  const [evalRes, setEvalRes] = useState<any>(null);
  const [driftRes, setDriftRes] = useState<any>(null);
  const [driftMode, setDriftMode] = useState(false);
  const timer = useRef<any>(null);

  // Send-money form
  const [to, setTo] = useState("Rahul Sharma");
  const [amount, setAmount] = useState("2500");
  const [note, setNote] = useState("Dinner split");
  const [payResult, setPayResult] = useState<any>(null);
  const [payBusy, setPayBusy] = useState(false);

  const makeRow = useCallback((rand: () => number, id: number, drift = false): TxnRow => {
    const t = generateTransaction(rand, id, drift);
    const s = scoreTransaction(t);
    const merchant = MERCHANTS[Math.floor(rand() * MERCHANTS.length)];
    const category = CATS[Math.floor(rand() * CATS.length)];
    const status = s.final >= 0.85 ? "blocked" : s.final >= 0.65 ? "review" : "approved";
    return { ...s, isFraud: t.isFraud, merchant, category, event_timestamp: t.event_timestamp, status };
  }, []);

  const genBatch = useCallback((n = 10) => {
    setSeed((s) => {
      const rand = rng(s * 7919 + 13);
      const batch: TxnRow[] = [];
      for (let i = 0; i < n; i++) batch.push(makeRow(rand, s * 100000 + i, driftMode));
      setTxns((prev) => [...batch, ...prev].slice(0, 120));
      return s + 1;
    });
  }, [makeRow, driftMode]);

  useEffect(() => { genBatch(24); }, []); // initial history
  useEffect(() => {
    if (streaming) timer.current = setInterval(() => genBatch(4), 2500);
    return () => clearInterval(timer.current);
  }, [streaming, genBatch]);

  const alerts = useMemo(() => txns.filter((t) => t.status !== "approved" && !t.acked), [txns]);
  const blocked = useMemo(() => txns.filter((t) => t.status === "blocked"), [txns]);
  const spent = useMemo(() => txns.filter((t) => t.status === "approved").reduce((s, t) => s + t.amount, 0), [txns]);
  const balance = 84250 - spent;
  const lat = useMemo(() => { const a = txns.map((t) => t.total_ms).sort((x, y) => x - y); return a.length ? a[Math.floor(0.95 * a.length)] : 0; }, [txns]);

  const filtered = useMemo(() => {
    let list = txns;
    if (filter === "flagged") list = list.filter((t) => t.status !== "approved");
    if (filter === "blocked") list = list.filter((t) => t.status === "blocked");
    if (filter === "approved") list = list.filter((t) => t.status === "approved");
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter((t) => t.merchant.toLowerCase().includes(q) || String(t.transaction_id).includes(q) || t.uid.toLowerCase().includes(q));
    }
    return list;
  }, [txns, filter, search]);

  async function sendMoney(e: React.FormEvent) {
    e.preventDefault();
    setPayBusy(true); setPayResult(null);
    const amt = parseFloat(amount) || 0;
    // Score via real API (same contract as FastAPI /score)
    const r = await fetch("/api/score", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ seed: Math.floor(Math.random() * 1e9), drift: driftMode }) }).then((x) => x.json()).catch(() => null);
    const final = r?.final ?? 0.1;
    const status = final >= 0.85 ? "blocked" : final >= 0.65 ? "review" : "approved";
    const row: TxnRow = {
      transaction_id: r?.transaction_id ?? Math.floor(1000000 + Math.random() * 9000000),
      uid: r?.uid ?? "acct_self", amount: amt, xgb: r?.xgb ?? 0, gnn: r?.gnn ?? 0, final,
      flagged: r?.flagged ?? false, ring_id: r?.ring_id ?? null, ring_size: r?.ring_size ?? 0,
      reasons: r?.reasons ?? [], total_ms: r?.total_ms ?? 8, model_version: r?.model_version ?? "3",
      merchant: `To ${to}`, category: "Transfer", status,
    };
    setTxns((prev) => [row, ...prev]);
    setPayResult({ ...row, note });
    setPayBusy(false);
  }

  async function runDrift() {
    const r = await fetch(`/api/drift?drift=${driftMode ? 1 : 0}`).then((x) => x.json());
    setDriftRes(r);
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="logo"><div className="logo-badge">🏦</div><div><b>Nexvia Bank</b><small>Fraud Protection Prototype</small></div></div>
        <nav className="nav">
          {NAV.map((n) => (
            <button key={n.id} className={page === n.id ? "on" : ""} onClick={() => setPage(n.id)}>
              <span>{n.icon}</span><span className="lbl">{n.label}</span>
              {n.id === "alerts" && alerts.length > 0 && <span className="count">{alerts.length}</span>}
            </button>
          ))}
        </nav>
        <div className="side-user"><div className="avatar">AR</div><div><b style={{ fontSize: 13 }}>Aarav Reddy</b><div className="small" style={{ color: "#93c5fd" }}>Savings ••4521</div></div></div>
      </aside>

      <div className="main">
        <div className="topbar">
          <h1>{NAV.find((n) => n.id === page)?.label}</h1>
          <div className="row">
            <input className="search" placeholder="🔍 Search transactions…" value={search} onChange={(e) => { setSearch(e.target.value); if (page !== "txns") setPage("txns"); }} />
            <button className="icon-btn" onClick={() => setPage("alerts")}>🔔{alerts.length > 0 && <span className="dot">{alerts.length}</span>}</button>
            <span className="badge green">🛡️ Protected</span>
          </div>
        </div>

        <div className="content">
          {page === "home" && (
            <div className="grid">
              <div className="grid two">
                <div className="card balance-card">
                  <div className="mut small">Total balance · Savings account</div>
                  <div className="big">{money(balance)}</div>
                  <div className="small" style={{ color: "#bfdbfe" }}>Aarav Reddy · ••4521 · Every payment is fraud-checked in milliseconds</div>
                  <div className="row" style={{ marginTop: 14 }}>
                    <button className="btn primary" onClick={() => setPage("pay")}>💸 Send money</button>
                    <button className="btn" onClick={() => setPage("txns")}>View transactions</button>
                  </div>
                </div>
                <div className="card">
                  <h3>🛡️ Fraud protection status</h3>
                  <p className="small mut">Your account is protected by AI that checks every transaction before it completes.</p>
                  <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    <div><div className="small mut">Blocked fraud</div><b style={{ fontSize: 22 }}>{blocked.length}</b></div>
                    <div><div className="small mut">Under review</div><b style={{ fontSize: 22 }}>{alerts.length}</b></div>
                    <div><div className="small mut">Check time (p95)</div><b style={{ fontSize: 22 }}>{lat.toFixed(1)} ms</b></div>
                    <div><div className="small mut">Model</div><b style={{ fontSize: 22 }}>v3 · champion</b></div>
                  </div>
                  <button className="btn" style={{ marginTop: 10 }} onClick={() => setPage("alerts")}>View alerts →</button>
                </div>
              </div>

              <div className="card">
                <h3>Quick actions</h3>
                <div className="quick">
                  <button onClick={() => setPage("pay")}><span className="ic">💸</span>Send money</button>
                  <button onClick={() => setPage("txns")}><span className="ic">🧾</span>Transactions</button>
                  <button onClick={() => setPage("cards")}><span className="ic">💳</span>My cards</button>
                  <button onClick={() => setPage("alerts")}><span className="ic">🔔</span>Alerts</button>
                </div>
              </div>

              <div className="grid two">
                <div className="card">
                  <div className="row" style={{ justifyContent: "space-between" }}><h3>Recent transactions</h3><button className="btn" onClick={() => setPage("txns")}>See all</button></div>
                  {txns.slice(0, 6).map((t) => (
                    <div className="txn" key={t.transaction_id} onClick={() => setSelected(t)} style={{ cursor: "pointer" }}>
                      <div className="txn-ic" style={{ background: t.status === "blocked" ? "#fee2e2" : t.status === "review" ? "#fef3c7" : "#dcfce7" }}>{t.status === "blocked" ? "⛔" : t.status === "review" ? "⏳" : "✓"}</div>
                      <div style={{ flex: 1 }}><b>{t.merchant}</b><div className="small mut">{t.category} · {t.event_timestamp?.slice(0, 10)}</div></div>
                      <div style={{ textAlign: "right" }}><b>{money(t.amount)}</b><div>{statusBadge(t.status)}</div></div>
                    </div>
                  ))}
                  {txns.length === 0 && <p className="mut">Loading transactions…</p>}
                </div>
                <div className="card">
                  <h3>🔴 Live activity</h3>
                  <p className="small mut">Simulated live stream (like Kafka replay). New transactions arrive every few seconds.</p>
                  <div className="row">
                    <button className={`btn ${streaming ? "danger" : "primary"}`} onClick={() => setStreaming(!streaming)}>{streaming ? "⏸ Pause live" : "▶ Start live stream"}</button>
                    <button className="btn" onClick={() => genBatch(10)}>+ Add 10 now</button>
                  </div>
                  <p className="small mut" style={{ marginTop: 10 }}>{fmt(txns.length)} transactions shown · {blocked.length} blocked · {alerts.length} need attention</p>
                  <Bar v={Math.min(1, alerts.length / 20)} color="#2563eb" />
                  <p className="small mut">Alert load (lower is better)</p>
                </div>
              </div>
            </div>
          )}

          {page === "pay" && (
            <div className="grid two">
              <div className="card">
                <h3>💸 Send money</h3>
                <p className="small mut">Every payment is checked by our fraud AI <b>before</b> money moves. Try a normal payment, then toggle “Simulate suspicious” to see a block.</p>
                <form onSubmit={sendMoney}>
                  <div className="field"><label>To</label><input value={to} onChange={(e) => setTo(e.target.value)} placeholder="Name / UPI ID" /></div>
                  <div className="field"><label>Amount (₹)</label><input type="number" min="1" value={amount} onChange={(e) => setAmount(e.target.value)} /></div>
                  <div className="field"><label>Note</label><input value={note} onChange={(e) => setNote(e.target.value)} /></div>
                  <label className="row small" style={{ marginBottom: 12 }}><input type="checkbox" style={{ width: "auto" }} checked={driftMode} onChange={(e) => setDriftMode(e.target.checked)} /> Simulate suspicious payment (unusual amount + new device)</label>
                  <button className="btn primary" disabled={payBusy} style={{ width: "100%" }}>{payBusy ? "Checking fraud risk…" : "Send securely →"}</button>
                </form>
              </div>
              <div className="card">
                <h3>Result</h3>
                {!payResult && <p className="mut">Your fraud-check result will appear here with a plain-English explanation.</p>}
                {payResult && (
                  <div>
                    <div style={{ fontSize: 40 }}>{payResult.status === "blocked" ? "⛔" : payResult.status === "review" ? "⏳" : "✅"}</div>
                    <h3>{payResult.status === "blocked" ? "Payment blocked" : payResult.status === "review" ? "Sent for review" : "Payment approved"}</h3>
                    <p className="small">{money(payResult.amount)} to <b>{to}</b> · Risk score <b>{payResult.final.toFixed(3)}</b> · Checked in {payResult.total_ms.toFixed(1)} ms</p>
                    <div className="scorebar"><Bar v={payResult.final} color={payResult.final >= 0.65 ? "#dc2626" : "#16a34a"} /><span className="small">{Math.round(payResult.final * 100)}% risk</span></div>
                    {payResult.reasons?.length > 0 ? <>
                      <h4>Why?</h4>
                      <ul className="small">{payResult.reasons.map((r: any) => <li key={r.feature}>{plainReason(r.feature)} <span className="mut">({r.feature})</span></li>)}</ul>
                    </> : <p className="small mut">No warning signs — amount, device and behaviour look normal.</p>}
                    {payResult.status !== "approved" && <p className="small">We blocked/held this to protect you. If this was really you, please verify in Alerts.</p>}
                  </div>
                )}
              </div>
            </div>
          )}

          {page === "txns" && (
            <div className="card">
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h3 style={{ margin: 0 }}>Transactions</h3>
                <div className="row">
                  <select value={filter} onChange={(e) => setFilter(e.target.value)} style={{ width: "auto" }}>
                    <option value="all">All</option><option value="approved">Approved</option><option value="flagged">Flagged</option><option value="blocked">Blocked</option>
                  </select>
                  <button className={`btn ${streaming ? "danger" : "primary"}`} onClick={() => setStreaming(!streaming)}>{streaming ? "⏸ Pause" : "▶ Live"}</button>
                </div>
              </div>
              <table><thead><tr><th>Merchant</th><th>Amount</th><th>Risk</th><th>Status</th><th>Date</th></tr></thead>
                <tbody>{filtered.slice(0, 40).map((t) => (
                  <tr key={t.transaction_id} onClick={() => setSelected(t)} style={{ cursor: "pointer" }}>
                    <td><b>{t.merchant}</b><div className="small mut">{t.category} · #{t.transaction_id}</div></td>
                    <td>{money(t.amount)}</td>
                    <td><div className="scorebar"><span>{t.final.toFixed(2)}</span><Bar v={t.final} color={t.final >= 0.65 ? "#dc2626" : "#2563eb"} /></div></td>
                    <td>{statusBadge(t.status)}</td><td className="small">{t.event_timestamp?.slice(0, 10)}</td>
                  </tr>))}
                </tbody></table>
              {filtered.length === 0 && <p className="mut">No transactions match. Try clearing search or starting live stream.</p>}
            </div>
          )}

          {page === "cards" && (
            <div className="grid two">
              <div className="card" style={{ background: "linear-gradient(135deg,#0b1b3a,#2563eb)", color: "#fff", border: 0 }}>
                <div className="small" style={{ color: "#bfdbfe" }}>NEXVIA BANK · DEBIT</div>
                <div style={{ fontSize: 22, letterSpacing: 2, margin: "18px 0" }}>•••• •••• •••• 4521</div>
                <div className="row" style={{ justifyContent: "space-between" }}><span>AARAV REDDY</span><span>12/28</span></div>
                <div className="row" style={{ marginTop: 14 }}><span className={`badge ${cardFrozen ? "red" : "green"}`}>{cardFrozen ? "❄️ Frozen" : "● Active"}</span><span className="small" style={{ color: "#bfdbfe" }}>Daily limit ₹50,000</span></div>
              </div>
              <div className="card">
                <h3>Card controls</h3>
                <p className="small mut">If you lose your card or see fraud, freeze it instantly. Our AI also watches card payments for misuse.</p>
                <div className="row">
                  <button className={`btn ${cardFrozen ? "primary" : "danger"}`} onClick={() => setCardFrozen(!cardFrozen)}>{cardFrozen ? "🔓 Unfreeze card" : "❄️ Freeze card"}</button>
                  <button className="btn" onClick={() => setPage("alerts")}>View card alerts</button>
                </div>
                <h4 style={{ marginTop: 16 }}>Spending this month</h4>
                <Bar v={Math.min(1, spent / 50000)} color="#2563eb" />
                <p className="small mut">{money(spent)} of ₹50,000 daily limit</p>
              </div>
            </div>
          )}

          {page === "alerts" && (
            <div className="grid two">
              <div className="card">
                <h3>🔔 Fraud alerts</h3>
                {alerts.length === 0 && <p className="mut">All clear! No pending alerts. Blocked payments appear here for your review.</p>}
                {alerts.map((t) => (
                  <div className="txn" key={t.transaction_id}>
                    <div className="txn-ic" style={{ background: t.status === "blocked" ? "#fee2e2" : "#fef3c7" }}>{t.status === "blocked" ? "⛔" : "⏳"}</div>
                    <div style={{ flex: 1 }}><b>{t.merchant}</b> · {money(t.amount)}<div className="small mut">{t.reasons?.[0] ? plainReason(t.reasons[0].feature) : "Unusual activity"}</div></div>
                    <div className="row">
                      <button className="btn" onClick={() => setSelected(t)}>Details</button>
                      <button className="btn primary" onClick={() => setTxns((prev) => prev.map((x) => x.transaction_id === t.transaction_id ? { ...x, acked: true } : x))}>This was me ✓</button>
                    </div>
                  </div>
                ))}
                <h4 style={{ marginTop: 16 }}>Blocked payments</h4>
                {blocked.length === 0 && <p className="small mut">Nothing blocked yet.</p>}
                {blocked.slice(0, 5).map((t) => <div className="txn" key={t.transaction_id}><div style={{ flex: 1 }}><b>{t.merchant}</b><div className="small mut">Risk {t.final.toFixed(2)} · {money(t.amount)}</div></div>{statusBadge(t.status)}</div>)}
              </div>
              <div className="card">
                <h3>🔒 Security centre</h3>
                <ul className="small">
                  <li>✅ Real-time fraud check on every payment</li>
                  <li>✅ New-device & unusual-amount detection</li>
                  <li>✅ Fraud-ring detection (linked accounts)</li>
                  <li>✅ Instant card freeze</li>
                  <li>✅ Drift monitoring — model retrains if behaviour changes</li>
                </ul>
                <h4>Drift check (is the model still healthy?)</h4>
                <div className="row">
                  <button className="btn primary" onClick={runDrift}>Run drift check</button>
                  <label className="small row"><input type="checkbox" style={{ width: "auto" }} checked={driftMode} onChange={(e) => setDriftMode(e.target.checked)} /> Simulate behaviour change</label>
                </div>
                {driftRes && <p className="small">Drifted features: <b>{Math.round(driftRes.share * 100)}%</b> — {driftRes.decision}</p>}
              </div>
            </div>
          )}

          {page === "how" && (
            <div className="grid">
              <div className="card">
                <h3>💡 How fraud protection works (simple version)</h3>
                <p className="small mut">This prototype uses the same ideas as a real bank’s fraud system. Here’s the flow in plain words:</p>
                <div className="progress-steps">
                  <div className="step"><b>1. You pay</b><span>Amount, device, location & history are collected.</span></div>
                  <div className="step"><b>2. AI checks in ms</b><span>XGBoost scores the payment; a graph model checks linked accounts.</span></div>
                  <div className="step"><b>3. Decide</b><span>Safe → approve. Risky → review or block with reasons.</span></div>
                  <div className="step"><b>4. Learn</b><span>Your feedback + drift checks retrain the model safely.</span></div>
                </div>
              </div>
              <div className="grid two">
                <div className="card">
                  <div className="row" style={{ justifyContent: "space-between" }}><h3 style={{ margin: 0 }}>Model results (IEEE-CIS)</h3><button className="btn primary" onClick={() => setEvalRes(evaluate(4000))}>Recompute live</button></div>
                  {evalRes && <p className="small">Live demo: PR-AUC <b>{evalRes.xgb_pr_auc.toFixed(3)}</b> · precision {evalRes.precision.toFixed(3)} · recall {evalRes.recall.toFixed(3)} · net savings {money(evalRes.net_savings)}</p>}
                  <table><thead><tr><th>Model</th><th>PR-AUC (test)</th></tr></thead>
                    <tbody>{MODEL_COMPARISON.map((m) => <tr key={m.name}><td>{m.name}</td><td>{m.test_pr}</td></tr>)}</tbody></table>
                  <p className="small mut">Fraud is rare (~3.5%), so we use PR-AUC — never accuracy. Champion chosen on validation only.</p>
                </div>
                <div className="card">
                  <h3>🏗️ System architecture</h3>
                  <pre>{`You pay → Kafka stream → Feast features
  → XGBoost + GraphSAGE blend → Approve / Review / Block
  → Alerts + analyst feedback → Evidently drift → Retrain
  → Kubernetes (Minikube) in production`}</pre>
                  <p className="small mut">Full code: <code>/backend</code> (Feast, MLflow, Kafka, FastAPI, Postgres, K8s). This web app is the deployable prototype twin for Vercel.</p>
                  <h4>Data facts</h4>
                  <p className="small">590,540 transactions · ~3.5% fraud · Time split 70/15/15 · Proxy account id (limitation stated openly).</p>
                </div>
              </div>
              <div className="card">
                <h3>🚀 Deploy this prototype</h3>
                <pre>{`git push to GitHub → vercel.com → Import → Deploy
(No env vars needed. Netlify alternative included.)`}</pre>
              </div>
            </div>
          )}
        </div>
      </div>

      {selected && (
        <div className="modal-bg" onClick={() => setSelected(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="row" style={{ justifyContent: "space-between" }}><h3 style={{ margin: 0 }}>Transaction details</h3><button className="btn" onClick={() => setSelected(null)}>✕</button></div>
            <p><b>{selected.merchant}</b> · {money(selected.amount)} · {statusBadge(selected.status)}</p>
            <p className="small mut">ID #{selected.transaction_id} · Account {selected.uid} · {selected.event_timestamp?.slice(0, 19).replace("T", " ")}</p>
            <div className="scorebar"><span>Risk</span><Bar v={selected.final} color={selected.final >= 0.65 ? "#dc2626" : "#2563eb"} /><b>{selected.final.toFixed(3)}</b></div>
            <p className="small">XGBoost {selected.xgb.toFixed(3)} · Graph {selected.gnn.toFixed(3)} · Checked in {selected.total_ms.toFixed(1)} ms · Model v{selected.model_version}</p>
            {selected.reasons?.length > 0 ? <>
              <h4>Why was this flagged?</h4>
              <ul className="small">{selected.reasons.map((r) => <li key={r.feature}>{plainReason(r.feature)} <span className="mut">({r.feature} = {String(r.value ?? "—")})</span></li>)}</ul>
            </> : <p className="small mut">No warning signs detected.</p>}
            {selected.ring_size > 0 && <p className="small">🔗 Part of a linked group (ring size {selected.ring_size}) — accounts sharing devices/cards.</p>}
            <div className="row" style={{ marginTop: 12 }}>
              <button className="btn primary" onClick={() => { setTxns((prev) => prev.map((x) => x.transaction_id === selected.transaction_id ? { ...x, acked: true } : x)); setSelected(null); }}>This was me ✓</button>
              <button className="btn" onClick={() => setSelected(null)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
