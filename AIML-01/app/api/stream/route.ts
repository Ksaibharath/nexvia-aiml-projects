import { NextResponse } from "next/server";
import { scoreTransaction, generateTransaction, rng } from "@/lib/model";

// GET /api/stream?n=20&seed=1&drift=0 — batch replay (Kafka producer equivalent)
export async function GET(req: Request) {
  const u = new URL(req.url);
  const n = Math.min(200, Math.max(1, parseInt(u.searchParams.get("n") ?? "20")));
  const seed = parseInt(u.searchParams.get("seed") ?? "1");
  const drift = u.searchParams.get("drift") === "1";
  const rand = rng(seed);
  const rows = [];
  for (let i = 0; i < n; i++) {
    const t = generateTransaction(rand, seed * 100000 + i, drift);
    const s = scoreTransaction(t);
    rows.push({ ...s, isFraud: t.isFraud, amount: t.TransactionAmt, event_timestamp: t.event_timestamp });
  }
  return NextResponse.json({ count: rows.length, drift, rows });
}
