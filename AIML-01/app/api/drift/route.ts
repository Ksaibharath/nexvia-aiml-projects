import { NextResponse } from "next/server";
import { scoreTransaction, generateTransaction, rng, FEATURES } from "@/lib/model";

// GET /api/drift?drift=0|1 — Evidently-style drift check (reference vs current)
export async function GET(req: Request) {
  const u = new URL(req.url);
  const drift = u.searchParams.get("drift") === "1";
  const refRand = rng(7), curRand = rng(drift ? 99 : 7);
  const mean = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length;
  const cols = ["TransactionAmt","amt_z","cnt_24h","sum_24h","amt_log","P_emaildomain_freq","device_changed","n_past"];
  const ref: Record<string, number[]> = {}, cur: Record<string, number[]> = {};
  for (const c of cols) { ref[c] = []; cur[c] = []; }
  for (let i = 0; i < 1500; i++) {
    const a = generateTransaction(refRand, i, false);
    const b = generateTransaction(curRand, i, drift);
    for (const c of cols) { ref[c].push((a as any)[c] ?? 0); cur[c].push((b as any)[c] ?? 0); }
  }
  const perFeature = cols.map((c) => {
    const m0 = mean(ref[c]), m1 = mean(cur[c]);
    const s0 = Math.sqrt(mean(ref[c].map((v) => (v - m0) ** 2))) || 1;
    const shift = Math.abs(m1 - m0) / s0;
    return { feature: c, ref_mean: Math.round(m0 * 1000) / 1000, cur_mean: Math.round(m1 * 1000) / 1000, shift: Math.round(shift * 1000) / 1000, drifted: shift > 0.25 };
  });
  const share = perFeature.filter((f) => f.drifted).length / perFeature.length;
  const decision = share >= 0.3 ? "RETRAIN triggered — challenger must beat champion (≥99% PR-AUC)" : "No action — within drift tolerance";
  return NextResponse.json({ share: Math.round(share * 100) / 100, decision, perFeature, features_checked: FEATURES.length });
}
