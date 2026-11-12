import { NextResponse } from "next/server";
import { scoreTransaction, generateTransaction, rng } from "@/lib/model";

// POST /api/score — mirrors FastAPI POST /score (backend/fraud/serving/api.py)
// Body: { transaction_id?, uid?, drift? } or a full transaction object.
// If no transaction supplied, a synthetic one is generated (demo mode).
export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  let txn: any;
  if (body.TransactionAmt !== undefined) {
    txn = body;
  } else {
    const seed = body.seed ?? Math.floor(Math.random() * 1e9);
    txn = generateTransaction(rng(seed), body.transaction_id ?? seed % 100000, !!body.drift);
    if (body.uid) txn.uid = body.uid;
    if (body.transaction_id) txn.transaction_id = body.transaction_id;
  }
  const r = scoreTransaction(txn);
  return NextResponse.json({
    transaction_id: r.transaction_id, uid: r.uid, xgb: r.xgb, gnn: r.gnn,
    final: r.final, flagged: r.flagged, reasons: r.reasons,
    feast_ms: r.feast_ms, total_ms: r.total_ms, model_version: r.model_version,
    ring_id: r.ring_id, ring_size: r.ring_size,
  });
}
