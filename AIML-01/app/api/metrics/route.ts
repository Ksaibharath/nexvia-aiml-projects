import { NextResponse } from "next/server";
import { evaluate } from "@/lib/model";
import { MODEL_COMPARISON, EDA, LATENCY } from "@/lib/data";

export async function GET(req: Request) {
  const u = new URL(req.url);
  const n = parseInt(u.searchParams.get("n") ?? "4000");
  const live = evaluate(Math.min(8000, Math.max(500, n)));
  return NextResponse.json({ live, reference: MODEL_COMPARISON, eda: EDA, latency: LATENCY });
}
