"""Build exact (Flat) and approximate (HNSW, IVF) FAISS indexes and benchmark ANN speed vs recall."""
import json
import os
import pathlib
import time
import faiss
import numpy as np


def _available_ram_gb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 8.0


def retrieve(user_vecs: np.ndarray, seen_sets: list[set[int]], k: int = 300, index=None) -> list[list[tuple[int, float]]]:
    """user_vecs: (B, 64) float32. seen_sets: list of sets of item_idx to exclude."""
    if index is None:
        index = faiss.read_index("artifacts/items_hnsw.faiss")
    vecs = np.ascontiguousarray(user_vecs, dtype="float32").copy()
    faiss.normalize_L2(vecs)
    max_extra = max((len(s) for s in seen_sets), default=0)
    search_k = min(index.ntotal, k + max(100, min(max_extra + 50, 1000)))
    scores, pos = index.search(vecs, search_k)
    out = []
    for s_row, p_row, seen in zip(scores, pos, seen_sets):
        keep = [
            (int(p) + 1, float(s))
            for p, s in zip(p_row, s_row)
            if p >= 0 and (int(p) + 1) not in seen
        ]
        out.append(keep[:k])
    return out


def build_and_benchmark() -> dict:
    emb = np.load("artifacts/item_embeddings.npy").astype("float32")
    emb = np.ascontiguousarray(emb)
    faiss.normalize_L2(emb)
    d = emb.shape[1]

    # 1) Exact index (ground truth)
    flat = faiss.IndexFlatIP(d)
    flat.add(emb)

    # 2) Approximate index 1: HNSW
    hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(emb)
    hnsw.hnsw.efSearch = 128

    # 3) Approximate index 2: IVF
    nlist = max(16, int(len(emb) ** 0.5))
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(emb)
    ivf.add(emb)
    ivf.nprobe = 16

    n_queries = min(2000, len(emb))
    q = np.ascontiguousarray(emb[np.random.RandomState(0).choice(len(emb), n_queries, replace=False)])
    K = 100
    _, truth = flat.search(q, K)

    catalog_bench = {}
    for name, idx in [("flat", flat), ("hnsw", hnsw), ("ivf", ivf)]:
        # Warmup
        _ = idx.search(q[:32], K)
        t0 = time.perf_counter()
        _, got = idx.search(q, K)
        dt = max(time.perf_counter() - t0, 1e-6)
        rec = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(truth, got)]))
        qps = float(len(q) / dt)
        catalog_bench[name] = {
            "recall_vs_exact@100": round(rec, 4),
            "qps": round(qps, 1),
            "ms_per_query": round(dt * 1000.0 / len(q), 4),
        }
        print(f"{name:5s} recall_vs_exact@{K}={rec:.4f}  qps={qps:,.0f}")

    pathlib.Path("artifacts").mkdir(parents=True, exist_ok=True)
    faiss.write_index(hnsw, "artifacts/items_hnsw.faiss")
    faiss.write_index(flat, "artifacts/items_flat.faiss")

    # Scaling demonstration on realistic clustered embedding manifold
    scale_n = int(os.getenv("FAISS_SCALE_N", "5000000" if _available_ram_gb() >= 6.0 else "300000"))
    rng = np.random.RandomState(42)
    centers = rng.randn(64, d).astype("float32")
    faiss.normalize_L2(centers)
    assign = rng.randint(0, 64, size=scale_n)
    big = (centers[assign] + 0.14 * rng.randn(scale_n, d).astype("float32")).astype("float32")
    faiss.normalize_L2(big)
    q_big = np.ascontiguousarray(big[:500])

    flat_big = faiss.IndexFlatIP(d)
    flat_big.add(big)
    t0 = time.perf_counter()
    _, truth_big = flat_big.search(q_big, K)
    dt_flat = max(time.perf_counter() - t0, 1e-6)
    del flat_big

    hnsw_big = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw_big.hnsw.efConstruction = 100
    hnsw_big.add(big)
    hnsw_big.hnsw.efSearch = 128
    t0 = time.perf_counter()
    _, got_big = hnsw_big.search(q_big, K)
    dt_hnsw = max(time.perf_counter() - t0, 1e-6)
    rec_big = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(truth_big, got_big)]))
    del hnsw_big, big

    scale_bench = {
        "n_vectors": scale_n,
        "n_queries": 500,
        "flat_seconds": round(dt_flat, 4),
        "flat_qps": round(500.0 / dt_flat, 1),
        "hnsw_seconds": round(dt_hnsw, 4),
        "hnsw_qps": round(500.0 / dt_hnsw, 1),
        "speedup_x": round(dt_flat / dt_hnsw, 2),
        "hnsw_recall_vs_exact@100": round(rec_big, 4),
    }
    print(
        f"scale ({scale_n:,d} vecs): flat={dt_flat:.3f}s ({scale_bench['flat_qps']:,.0f} qps) vs "
        f"hnsw={dt_hnsw:.3f}s ({scale_bench['hnsw_qps']:,.0f} qps, speedup={scale_bench['speedup_x']}x, recall={rec_big:.3f})"
    )

    result = {"n_catalog_items": int(len(emb)), "dim": int(d), "catalog": catalog_bench, "scaling": scale_bench}
    with open("artifacts/faiss_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    build_and_benchmark()
