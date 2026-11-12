"""Counterfactual Replay A/B Simulation + SRM Check + Welch's t-test + Latency Benchmark (Part 10)."""
import json
import os
import pathlib
import numpy as np
import pandas as pd
import requests
from fastapi.testclient import TestClient
from scipy import stats

from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k
from src.serving.api import app, cache


def _bootstrap_diff_ci(a_vals: np.ndarray, b_vals: np.ndarray, n_boot: int = 1000, seed: int = 42) -> list[float]:
    a_clean = a_vals[~np.isnan(a_vals)]
    b_clean = b_vals[~np.isnan(b_vals)]
    if len(a_clean) == 0 or len(b_clean) == 0:
        return [0.0, 0.0]
    rng = np.random.RandomState(seed)
    diffs = [
        float(np.mean(rng.choice(b_clean, size=len(b_clean), replace=True)) - np.mean(rng.choice(a_clean, size=len(a_clean), replace=True)))
        for _ in range(n_boot)
    ]
    return [round(float(np.percentile(diffs, 2.5)), 5), round(float(np.percentile(diffs, 97.5)), 5)]


def run_ab_simulation() -> dict:
    use_http = os.getenv("AB_USE_HTTP", "0") == "1"
    api_url = os.getenv("API_URL", "http://localhost:8080")
    client = None if use_http else TestClient(app)

    def fetch_rec(uid: int, k: int = 10, liked: str = "", bypass_cache: bool = False) -> dict:
        params = {"user_id": int(uid), "k": k, "liked": liked, "bypass_cache": bypass_cache}
        if use_http:
            return requests.get(f"{api_url}/recommend", params=params, timeout=10).json()
        return client.get("/recommend", params=params).json()

    test_df = pd.read_parquet("data/processed/interactions", filters=[("split", "==", "test")])
    test_df = test_df[test_df["item_idx"] > 0].sort_values("timestamp")
    test_by_user = test_df.groupby("userId")["item_idx"].apply(list).to_dict()

    # Build >= 1,000 held-out visitor sessions:
    # 1) All true Test-split users (warm Stage-2 + pure cold-start)
    # 2) Held-out Val onboarding sessions (first 3 likes -> content-based /recommend?liked=... -> held-out future likes)
    visitors: list[tuple[int, str, set[int]]] = []
    for u, items_l in sorted(test_by_user.items()):
        visitors.append((int(u), "", set(int(x) for x in items_l)))

    if len(visitors) < 1050:
        val_df = pd.read_parquet("data/processed/interactions", filters=[("split", "==", "val")])
        val_df = val_df[val_df["item_idx"] > 0].sort_values("timestamp")
        for uid, grp in val_df.groupby("userId"):
            seq = grp["item_idx"].astype(int).tolist()
            if len(seq) >= 6:
                seed_liked = ",".join(str(x) for x in seq[:3])
                held = set(seq[3:]) - set(seq[:3])
                if held:
                    visitors.append((9_000_000 + int(uid), seed_liked, held))
            if len(visitors) >= 1100:
                break

    if len(visitors) < 1050:
        tr_df = pd.read_parquet("data/processed/interactions", filters=[("split", "==", "train")])
        tr_df = tr_df[tr_df["item_idx"] > 0].sort_values("timestamp")
        for uid, grp in tr_df.groupby("userId"):
            seq = grp["item_idx"].astype(int).tolist()
            if len(seq) >= 8:
                seed_liked = ",".join(str(x) for x in seq[:3])
                held = set(seq[-5:]) - set(seq[:3])
                if held:
                    visitors.append((8_000_000 + int(uid), seed_liked, held))
            if len(visitors) >= 1100:
                break

    rng = np.random.RandomState(1)
    if len(visitors) > 2000:
        idx_sel = rng.choice(len(visitors), size=2000, replace=False)
        visitors = [visitors[i] for i in idx_sel]

    cache.clear_local()
    log = []
    for u, liked_str, rel in visitors:
        r = fetch_rec(u, k=10, liked=liked_str, bypass_cache=True)
        shown = [int(x["item_idx"]) for x in r["items"]]
        clicks = sum(1 for i in shown if i in rel)
        log.append({
            "user": int(u),
            "variant": r["variant"],
            "source": r["source"],
            "impressions": len(shown),
            "clicks": clicks,
            "ctr": clicks / max(len(shown), 1),
            "ndcg10": float(ndcg_at_k(shown, rel, 10)),
            "hit10": float(hit_rate_at_k(shown, rel, 10)),
            "latency_ms": float(r["latency_ms"]),
        })

    df = pd.DataFrame(log)
    pathlib.Path("artifacts").mkdir(parents=True, exist_ok=True)
    df.to_csv("artifacts/ab_log.csv", index=False)

    n = df["variant"].value_counts()
    n_a, n_b = int(n.get("A", 0)), int(n.get("B", 0))
    srm_p = float(stats.chisquare([n_a, n_b]).pvalue)
    print(f"Total visitors: {len(df)}  |  Variant A: {n_a}  |  Variant B: {n_b}")
    print(f"SRM p-value: {srm_p:.4f} (should be > 0.05 — PASS: {srm_p > 0.05})")

    def compare_slice(sub_df: pd.DataFrame, label: str) -> dict:
        a = sub_df[sub_df["variant"] == "A"]
        b = sub_df[sub_df["variant"] == "B"]
        out_m = {"n_A": int(len(a)), "n_B": int(len(b))}
        print(f"\n--- {label} (N={len(sub_df)}: A={len(a)}, B={len(b)}) ---")
        for m in ("ctr", "ndcg10", "hit10"):
            a_arr = a[m].dropna().values
            b_arr = b[m].dropna().values
            t_res = stats.ttest_ind(b_arr, a_arr, equal_var=False)
            diff = float(np.mean(b_arr) - np.mean(a_arr))
            ci = _bootstrap_diff_ci(a_arr, b_arr)
            out_m[m] = {
                "A_mean": round(float(np.mean(a_arr)), 4),
                "B_mean": round(float(np.mean(b_arr)), 4),
                "diff_B_minus_A": round(diff, 4),
                "ci95_diff": ci,
                "p_value": round(float(t_res.pvalue), 4),
                "significant_at_05": bool(t_res.pvalue < 0.05),
            }
            print(
                f"  {m:7s}: A={out_m[m]['A_mean']:.4f}  B={out_m[m]['B_mean']:.4f}  "
                f"diff={diff:+.4f}  95%CI={ci}  p={t_res.pvalue:.4f}"
            )
        return out_m

    all_stats = compare_slice(df, "All Simulated Visitors")
    warm_df = df[df["source"].isin(["warm", "warm_blend"])]
    warm_stats = compare_slice(warm_df, "Warm Users Only (Stage-2 Active)")
    cold_df = df[~df["source"].isin(["warm", "warm_blend"])]
    cold_stats = {
        "n_users": int(len(cold_df)),
        "ctr_mean": round(float(cold_df["ctr"].mean()), 4) if len(cold_df) else 0.0,
        "ndcg10_mean": round(float(cold_df["ndcg10"].mean()), 4) if len(cold_df) else 0.0,
    }

    # Measure warm latency (200 miss vs 200 cache hits)
    warm_uids = warm_df["user"].head(200).tolist()
    cache.clear_local()
    miss_latencies = [fetch_rec(u, k=10, bypass_cache=False)["latency_ms"] for u in warm_uids]
    hit_latencies = [fetch_rec(u, k=10, bypass_cache=False)["latency_ms"] for u in warm_uids]

    latency_summary = {
        "warm_miss_p50_ms": round(float(np.percentile(miss_latencies, 50)), 2),
        "warm_miss_p95_ms": round(float(np.percentile(miss_latencies, 95)), 2),
        "warm_miss_mean_ms": round(float(np.mean(miss_latencies)), 2),
        "cache_hit_p50_ms": round(float(np.percentile(hit_latencies, 50)), 2),
        "cache_hit_p95_ms": round(float(np.percentile(hit_latencies, 95)), 2),
        "cache_hit_mean_ms": round(float(np.mean(hit_latencies)), 2),
    }
    print("\nLatency Benchmark (200 warm requests):", latency_summary)

    summary = {
        "total_visitors": int(len(df)),
        "variant_counts": {"A": n_a, "B": n_b},
        "srm_p_value": round(srm_p, 4),
        "srm_passed": bool(srm_p > 0.05),
        "overall": all_stats,
        "warm_only": warm_stats,
        "cold_start_only": cold_stats,
        "latency": latency_summary,
    }
    with open("artifacts/ab_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    metrics_path = pathlib.Path("artifacts/metrics.json")
    existing = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    existing["ab_test"] = summary
    existing["latency"] = latency_summary
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    run_ab_simulation()
