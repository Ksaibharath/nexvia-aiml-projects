// Core fraud scoring engine — JS port of the Python XGBoost + GNN blend.
// Mirrors backend/fraud/features/feature_list.py FEATURES order and
// backend/fraud/models/blend.py logit blending.

export const FEATURES = [
  "TransactionAmt","dist1","dist2","addr1","addr2","card1","card2","card3","card5",
  "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14",
  "D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15",
  "ProductCD_freq","card4_freq","card6_freq","P_emaildomain_freq","R_emaildomain_freq","DeviceType_freq",
  "M1","M2","M3","M4","M5","M6","M7","M8","M9",
  "amt_log","hour","dow","n_past","time_since_last","amt_mean_past","amt_std_past",
  "amt_z","amt_ratio","cnt_1h","cnt_24h","cnt_7d","sum_24h",
  "device_changed","addr_changed","email_changed","n_devices",
] as const;

export type Txn = {
  transaction_id: number; uid: string; event_timestamp: string;
  TransactionAmt: number; ProductCD: string; card4: string; DeviceType: string;
  isFraud?: number; // ground truth — NEVER sent to scorer in production
  // engineered / raw features (subset used by scorer)
  [k: string]: any;
};

export type Scored = {
  transaction_id: number; uid: string; amount: number;
  xgb: number; gnn: number; final: number; flagged: boolean;
  ring_id: number | null; ring_size: number;
  reasons: { feature: string; value: number | null; impact: number }[];
  feast_ms: number; total_ms: number; model_version: string;
  features: Record<string, number | null>;
};

// Seeded PRNG (mulberry32) so demo data is reproducible
export function rng(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const EMAILS = ["gmail.com","yahoo.com","hotmail.com","outlook.com","proton.me","mail.ru","tempmail.io","guerrillamail.com"];
const DEVICES = ["Windows","MacOS","Linux","Android","iOS","unknown"];
const CARDS = ["visa","mastercard","amex","discover"];

// Generate one synthetic IEEE-CIS-like transaction.
// Fraud rows are generated with burst velocity, amount spikes, new device,
// rare email domain — the patterns the Python features capture.
export function generateTransaction(rand: () => number, id: number, drift = false): Txn {
  const isFraud = rand() < 0.035 ? 1 : 0;
  const uid = `acct_${Math.floor(rand() * 800)}`;
  const baseAmt = 5 + Math.exp(rand() * 4.2); // lognormal-ish
  let amt = baseAmt;
  let cnt1h = Math.floor(rand() * 3);
  let cnt24h = Math.floor(rand() * 8);
  let amtZ = (rand() - 0.5) * 1.2;
  let deviceChanged = rand() < 0.08 ? 1 : 0;
  let emailIdx = Math.floor(rand() * 6); // common domains
  let nPast = Math.floor(rand() * 40);
  // Overlap: fraud often looks mild, legit sometimes looks suspicious —
  // otherwise PR-AUC would be a fake 1.0. Only ~55% of fraud shows the full pattern.
  const strength = rand(); // 0..1 fraud signal strength
  if (isFraud) {
    if (strength < 0.55) {
      amt = baseAmt * (2.5 + rand() * 4);
      cnt1h = 3 + Math.floor(rand() * 8);
      cnt24h = 8 + Math.floor(rand() * 20);
      amtZ = 1.5 + rand() * 3;
      deviceChanged = rand() < 0.55 ? 1 : 0;
      emailIdx = 6 + Math.floor(rand() * 2); // rare domains
      nPast = Math.floor(rand() * 6); // new-ish accounts
    } else if (strength < 0.85) {
      // partial pattern: 1-2 signals only
      amt = baseAmt * (1.5 + rand() * 2);
      cnt1h = Math.floor(rand() * 3);
      cnt24h = 3 + Math.floor(rand() * 6);
      amtZ = 0.5 + rand() * 1.2;
      deviceChanged = rand() < 0.3 ? 1 : 0;
      emailIdx = rand() < 0.4 ? 6 + Math.floor(rand() * 2) : Math.floor(rand() * 6);
      nPast = Math.floor(rand() * 15);
    } else {
      // stealth fraud: looks almost legit (hard cases)
      amt = baseAmt * (1.1 + rand() * 0.8);
      cnt1h = Math.floor(rand() * 2);
      cnt24h = Math.floor(rand() * 5);
      amtZ = (rand() - 0.3) * 0.8;
      deviceChanged = rand() < 0.15 ? 1 : 0;
      nPast = Math.floor(rand() * 25);
    }
  } else if (rand() < 0.09) {
    // legit but suspicious-looking (false-positive pressure)
    amt = baseAmt * (2 + rand() * 2);
    cnt1h = 2 + Math.floor(rand() * 4);
    cnt24h = 5 + Math.floor(rand() * 8);
    amtZ = 1 + rand() * 1.5;
    deviceChanged = rand() < 0.4 ? 1 : 0;
  }
  if (drift) { amt *= 1.8; amtZ += 2.2; cnt24h += 6; }
  const hour = Math.floor(rand() * 24);
  const dow = Math.floor(rand() * 7);
  const t = new Date(Date.UTC(2017, 11, 1) + id * 137000 + Math.floor(rand() * 60000));
  return {
    transaction_id: 1000000 + id, uid, event_timestamp: t.toISOString(),
    TransactionAmt: Math.round(amt * 100) / 100,
    ProductCD: ["P","W","H","C","S"][Math.floor(rand() * 5)],
    card4: CARDS[Math.floor(rand() * 4)],
    DeviceType: DEVICES[Math.floor(rand() * 6)],
    isFraud,
    dist1: Math.round(rand() * 2000 * 100) / 100,
    dist2: Math.round(rand() * 500 * 100) / 100,
    addr1: Math.floor(rand() * 500), addr2: Math.floor(rand() * 100),
    card1: Math.floor(rand() * 1000), card2: Math.floor(rand() * 100),
    card3: Math.floor(rand() * 1000), card5: Math.floor(rand() * 100),
    C1: Math.floor(rand() * 20), C2: Math.floor(rand() * 10), C3: Math.floor(rand() * 30),
    C4: Math.floor(rand() * 5), C5: Math.floor(rand() * 8), C6: Math.floor(rand() * 4),
    C7: Math.floor(rand() * 6), C8: Math.floor(rand() * 12), C9: Math.floor(rand() * 7),
    C10: Math.floor(rand() * 9), C11: Math.floor(rand() * 15), C12: Math.floor(rand() * 3),
    C13: Math.floor(rand() * 11), C14: Math.floor(rand() * 6),
    D1: Math.floor(rand() * 200), D2: Math.floor(rand() * 100), D3: Math.floor(rand() * 50),
    D4: Math.floor(rand() * 80), D5: Math.floor(rand() * 40), D6: Math.floor(rand() * 30),
    D7: Math.floor(rand() * 25), D8: Math.floor(rand() * 20), D9: Math.floor(rand() * 18),
    D10: Math.floor(rand() * 15), D11: Math.floor(rand() * 12), D12: Math.floor(rand() * 10),
    D13: Math.floor(rand() * 9), D14: Math.floor(rand() * 8), D15: Math.floor(rand() * 7),
    ProductCD_freq: 0.2, card4_freq: 0.25, card6_freq: 0.3,
    P_emaildomain_freq: isFraud ? (strength < 0.55 ? 0.01 : strength < 0.85 ? 0.08 : 0.22) : 0.25,
    R_emaildomain_freq: isFraud ? (strength < 0.55 ? 0.02 : 0.18) : 0.2,
    DeviceType_freq: 0.16,
    M1: rand() < 0.5 ? 1 : 0, M2: rand() < 0.5 ? 1 : 0, M3: rand() < 0.5 ? 1 : 0,
    M4: rand() < 0.5 ? 1 : 0, M5: rand() < 0.5 ? 1 : 0, M6: rand() < 0.5 ? 1 : 0,
    M7: rand() < 0.5 ? 1 : 0, M8: rand() < 0.5 ? 1 : 0, M9: rand() < 0.5 ? 1 : 0,
    amt_log: Math.log1p(amt), hour, dow, n_past: nPast,
    time_since_last: Math.floor(rand() * 86400),
    amt_mean_past: Math.round(baseAmt * 100) / 100,
    amt_std_past: Math.round(baseAmt * 0.5 * 100) / 100,
    amt_z: Math.round(amtZ * 100) / 100,
    amt_ratio: Math.round((amt / (baseAmt + 1e-6)) * 100) / 100,
    cnt_1h: cnt1h, cnt_24h: cnt24h, cnt_7d: cnt24h * 3,
    sum_24h: Math.round(cnt24h * baseAmt * 100) / 100,
    device_changed: deviceChanged,
    addr_changed: rand() < 0.1 ? 1 : 0,
    email_changed: rand() < 0.08 ? 1 : 0,
    n_devices: 1 + Math.floor(rand() * 3),
    email_domain: EMAILS[emailIdx],
  };
}

const clamp = (x: number, lo = 1e-6, hi = 1 - 1e-6) => Math.min(hi, Math.max(lo, x));
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const logit = (p: number) => Math.log(p / (1 - p));

// XGBoost-like scorer: weighted logit with interactions (stand-in for the
// trained booster; the Python backend loads the real model from MLflow).
export function xgbScore(t: Txn): { p: number; contrib: Record<string, number> } {
  const c: Record<string, number> = {};
  const add = (k: string, v: number) => { c[k] = v; };
  add("amt_z", 0.55 * Math.max(0, (t.amt_z ?? 0)));
  add("cnt_1h", 0.32 * Math.min(10, t.cnt_1h ?? 0));
  add("cnt_24h", 0.14 * Math.min(20, t.cnt_24h ?? 0));
  add("device_changed", 0.7 * (t.device_changed ?? 0));
  add("P_emaildomain_freq", t.P_emaildomain_freq < 0.05 ? 1.1 : 0.1);
  add("amt_ratio", 0.25 * Math.min(8, t.amt_ratio ?? 1));
  add("n_past", (t.n_past ?? 0) < 5 ? 0.5 : 0);
  add("sum_24h", 0.002 * Math.min(2000, t.sum_24h ?? 0));
  add("addr_changed", 0.35 * (t.addr_changed ?? 0));
  add("email_changed", 0.3 * (t.email_changed ?? 0));
  add("hour", (t.hour === 2 || t.hour === 3) ? 0.4 : 0);
  add("dist1", 0.0002 * Math.min(2000, t.dist1 ?? 0));
  add("C1", 0.03 * Math.min(20, t.C1 ?? 0));
  add("D1", 0.004 * Math.min(200, t.D1 ?? 0));
  add("TransactionAmt", 0.15 * Math.log1p(t.TransactionAmt ?? 0));
  let z = -4.2; // prior ~1.5% at zero signal; fraud rate 3.5% overall
  for (const v of Object.values(c)) z += v;
  // interaction: burst + new device is extra risky
  if ((t.cnt_1h ?? 0) >= 3 && (t.device_changed ?? 0) === 1) z += 0.8;
  return { p: clamp(sigmoid(z)), contrib: c };
}

// GNN-like account risk: accounts in rings / with many neighbours score higher.
// Deterministic from uid hash so ring membership is stable.
export function gnnScore(uid: string): { p: number; ring_id: number | null; ring_size: number } {
  let h = 0; for (let i = 0; i < uid.length; i++) h = (h * 31 + uid.charCodeAt(i)) >>> 0;
  const r = (h % 1000) / 1000;
  if (r < 0.06) return { p: clamp(0.35 + r * 4), ring_id: h % 120, ring_size: 3 + (h % 9) };
  if (r < 0.14) return { p: clamp(0.12 + r), ring_id: null, ring_size: 0 };
  return { p: clamp(0.02 + r * 0.05), ring_id: null, ring_size: 0 };
}

export const SERVING = { alpha: 0.72, threshold: 0.65, prior: 0.08, review_cost: 5.0, model_version: "3" };

export function scoreTransaction(t: Txn, opts?: { drift?: boolean }): Scored {
  const t0 = performance.now();
  const feastMs = 2 + Math.random() * 6; // simulated Feast/Redis lookup
  const { p: px, contrib } = xgbScore(t);
  const g = gnnScore(t.uid);
  const a = SERVING.alpha;
  const final = sigmoid(a * logit(px) + (1 - a) * logit(g.p));
  const flagged = final >= SERVING.threshold;
  const reasons = Object.entries(contrib)
    .sort((x, y) => Math.abs(y[1]) - Math.abs(x[1]))
    .slice(0, 3)
    .map(([feature, impact]) => ({ feature, value: (t as any)[feature] ?? null, impact: Math.round(impact * 1000) / 1000 }));
  const features: Record<string, number | null> = {};
  for (const f of FEATURES) features[f] = (t as any)[f] ?? null;
  return {
    transaction_id: t.transaction_id, uid: t.uid, amount: t.TransactionAmt,
    xgb: Math.round(px * 10000) / 10000, gnn: Math.round(g.p * 10000) / 10000,
    final: Math.round(final * 10000) / 10000, flagged,
    ring_id: g.ring_id, ring_size: g.ring_size,
    reasons: flagged ? reasons : [],
    feast_ms: Math.round(feastMs * 100) / 100,
    total_ms: Math.round((performance.now() - t0 + feastMs + 1.5) * 100) / 100,
    model_version: SERVING.model_version, features,
  };
}

// Offline evaluation helpers (PR-AUC via average precision)
export function averagePrecision(y: number[], p: number[]): number {
  const idx = y.map((v, i) => [v, p[i]] as const).sort((a, b) => b[1] - a[1]);
  const P = y.reduce((s, v) => s + v, 0); if (!P) return 0;
  let tp = 0, ap = 0;
  idx.forEach(([v], i) => { if (v === 1) { tp++; ap += tp / (i + 1); } });
  return ap / P;
}

export function evaluate(n = 4000, seed = 42) {
  const rand = rng(seed);
  const y: number[] = [], px: number[] = [], pf: number[] = [], amt: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = generateTransaction(rand, i);
    const s = scoreTransaction(t);
    y.push(t.isFraud ?? 0); px.push(s.xgb); pf.push(s.final); amt.push(t.TransactionAmt);
  }
  const thr = SERVING.threshold;
  const flagged = pf.map((p) => p >= thr);
  const tp = y.filter((v, i) => v === 1 && flagged[i]).length;
  const fp = y.filter((v, i) => v === 0 && flagged[i]).length;
  const caught = amt.filter((_, i) => y[i] === 1 && flagged[i]).reduce((s, v) => s + v, 0);
  const alerts = tp + fp;
  return {
    n, fraud_rate: y.reduce((s, v) => s + v, 0) / n,
    xgb_pr_auc: averagePrecision(y, px), blend_pr_auc: averagePrecision(y, pf),
    alert_rate: alerts / n, precision: alerts ? tp / alerts : 0,
    recall: tp / Math.max(1, y.reduce((s, v) => s + v, 0)),
    caught_amount: Math.round(caught * 100) / 100,
    net_savings: Math.round((caught - SERVING.review_cost * alerts) * 100) / 100,
    threshold: thr, alpha: SERVING.alpha,
  };
}
