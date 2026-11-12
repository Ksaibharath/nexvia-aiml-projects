import { NextResponse } from "next/server";
export async function GET() {
  return NextResponse.json({ status: "ok", model_version: "3", alias: "champion", uptime: process.uptime?.() ?? 0, framework: "Next.js (Vercel) + Python MLOps backend in /backend" });
}
