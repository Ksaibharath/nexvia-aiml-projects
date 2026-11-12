"""FastAPI Two-Stage Recommendation Gateway.

Automatically runs in Full Artifacts Mode when `artifacts/items_hnsw.faiss` exists,
or seamlessly falls back to `deploy_bundle/bundle.json` on a fresh clone/unzip so you
can launch the API & Web UI immediately on any machine before running full training.
"""
import hashlib
import json
import pathlib
import time
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.serving.bundle_runtime import BundleEngine
from src.serving.cache import Cache

with open("configs/config.yaml", "r", encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

app = FastAPI(
    title="Two-Stage Personalized Recommendation Engine API",
    description=(
        "Production-style MovieLens 25M recommendation gateway combining Two-Tower retrieval, "
        "FAISS HNSW ANN search, DCN-V2 / Wide & Deep Stage-2 ranking, Cold-Start fallbacks, "
        "versioned Redis caching, and deterministic online A/B routing."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = Cache(cfg["redis_ttl_seconds"])
RANKER = {"A": "ranker_wd", "B": "ranker"}  # A = Wide & Deep, B = DCN-V2

import importlib.util

_HAS_FULL_ARTIFACTS = (
    importlib.util.find_spec("faiss") is not None
    and importlib.util.find_spec("pyarrow") is not None
    and pathlib.Path("artifacts/items_hnsw.faiss").exists()
    and pathlib.Path("data/processed/items").exists()
)

if _HAS_FULL_ARTIFACTS:
    import faiss
    from src.ranking.features import build_dense, load_feature_arrays
    from src.retrieval.data import pad
    from src.serving.cold_start import ColdStart
    from src.serving.triton_client import backend_status, infer

    index = faiss.read_index("artifacts/items_hnsw.faiss")
    emb = np.load("artifacts/item_embeddings.npy").astype("float32")
    faiss.normalize_L2(emb)

    items = pd.read_parquet("data/processed/items")
    items = items[items["item_idx"] > 0].sort_values("item_idx").reset_index(drop=True)

    item_meta = {
        int(row.item_idx): {
            "item_idx": int(row.item_idx),
            "movieId": int(row.movieId),
            "title": str(row.title),
            "year": int(row.year) if pd.notna(row.year) else 1990,
            "genres": str(row.genres),
            "avg_rating": round(float(row.avg_rating), 2),
            "n_pos": int(row.n_pos),
        }
        for row in items.itertuples(index=False)
    }

    UF, IF, NG = load_feature_arrays()
    state_df = pd.read_parquet("data/processed/user_state_val_end")
    state = state_df.set_index("userId")
    cold = ColdStart(items, emb, index)
    ALL_GENRES = sorted(
        {
            g
            for gs in items["genres"].dropna().astype(str)
            for g in gs.split("|")
            if g and g != "(no genres listed)"
        }
    )
    bundle_engine = None
else:
    bundle_engine = BundleEngine("deploy_bundle/bundle.json")
    ALL_GENRES = bundle_engine.all_genres


def variant_for(user_id: int, salt: str = "exp1") -> str:
    """Stable 50/50 user-level A/B split via MD5 hashing."""
    digest = hashlib.md5(f"{salt}:{user_id}".encode("utf-8")).hexdigest()
    return "A" if int(digest, 16) % 2 == 0 else "B"


def _format_item(i: int, score: float | None = None, ret_score: float | None = None) -> dict:
    base = dict(
        item_meta.get(
            int(i),
            {"item_idx": int(i), "title": "Unknown", "genres": "", "year": 0, "avg_rating": 0.0, "n_pos": 0},
        )
    )
    if score is not None:
        base["score"] = round(float(score), 4)
    if ret_score is not None:
        base["retrieval_score"] = round(float(ret_score), 4)
    return base


def warm_pipeline(user_idx: int, history: list[int], seen_items: set[int], variant: str, k: int) -> list[dict]:
    H = pad([history])
    u = np.array([[user_idx]], dtype="int32")
    q = infer("user_tower", {"user_idx": u, "history": H}, "embedding").astype("float32")
    q = np.ascontiguousarray(q)
    faiss.normalize_L2(q)

    ret_k = int(cfg["retrieval_k"])
    search_k = min(index.ntotal, ret_k + len(seen_items) + 60)
    sc, pos = index.search(q, search_k)
    cand = [
        (int(p) + 1, float(s))
        for p, s in zip(pos[0], sc[0])
        if p >= 0 and (int(p) + 1) not in seen_items
    ][:ret_k]

    if not cand:
        return [_format_item(i) for i in cold.popular(k, exclude=seen_items)]

    ids = np.array([c[0] for c in cand], dtype="int32")
    rs = np.array([c[1] for c in cand], dtype="float32")
    dense = build_dense(
        np.full(len(ids), user_idx, dtype="int32"),
        ids,
        rs,
        np.full(len(ids), len(history), dtype="float32"),
        UF,
        IF,
        NG,
    )
    model_name = RANKER.get(variant, "ranker")
    logits = infer(
        model_name,
        {
            "user_idx": np.full((len(ids), 1), user_idx, dtype="int32"),
            "item_idx": ids.reshape(-1, 1),
            "dense": dense,
        },
        "score",
    ).ravel()

    order = np.argsort(-logits)[:k]
    return [_format_item(int(ids[idx]), float(logits[idx]), float(rs[idx])) for idx in order]


@app.get("/health")
@app.get("/api/health")
def health():
    if _HAS_FULL_ARTIFACTS:
        return {
            "status": "ok",
            "mode": "full_faiss_artifacts",
            "backend": backend_status(),
            "cache": cache.stats(),
            "n_catalog_items": int(len(items)),
            "n_warm_users": int(len(state)),
        }
    return {
        "status": "ok",
        "mode": "instant_bundle",
        "backend": {"triton_url": "localhost:8000", "triton_live": False, "active_engine": "deploy_bundle_numpy"},
        "cache": cache.stats(),
        "n_catalog_items": bundle_engine.n_total_catalog,
        "n_warm_users": bundle_engine.n_total_warm_users,
    }


@app.get("/recommend")
@app.get("/api/recommend")
def recommend(
    user_id: int = Query(..., description="MovieLens user ID (use e.g. 99999999 for a brand-new cold-start user)"),
    k: int = Query(10, ge=1, le=50, description="Number of movies to return"),
    liked: str = Query("", description="Comma-separated item_idx values for new-user content-based onboarding"),
    genres: str = Query("", description="Comma-separated genre names for cold-start genre filtering"),
    variant: str = Query("", description="Optional A/B override ('A' = Wide & Deep, 'B' = DCN-V2)"),
    bypass_cache: bool = Query(False, description="Skip Redis/memory cache to measure fresh model inference"),
):
    t0 = time.perf_counter()
    chosen_variant = variant if variant in ("A", "B") else variant_for(user_id)
    extra_sig = f"L={liked}|G={genres}" if (liked or genres) else ""

    if not bypass_cache:
        hit = cache.get(user_id, k, chosen_variant, extra=extra_sig)
        if hit is not None:
            return {
                "user_id": user_id,
                "variant": chosen_variant,
                "ranker_model": RANKER[chosen_variant],
                "source": "cache",
                "items": hit["items"] if isinstance(hit, dict) and "items" in hit else hit,
                "history_preview": hit.get("history_preview", []) if isinstance(hit, dict) else [],
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

    liked_ids = [int(x) for x in liked.split(",") if x.strip().isdigit()]
    genre_list = [g.strip() for g in genres.split(",") if g.strip()]
    history_preview = []

    if _HAS_FULL_ARTIFACTS:
        if user_id in state.index and not liked_ids and not genre_list:
            row = state.loc[user_id]
            hist_list = [int(x) for x in row["history"]]
            seen_raw = row["seen_items"] if "seen_items" in state.columns else hist_list
            seen_set = set(int(x) for x in seen_raw)
            history_preview = [_format_item(i) for i in hist_list[-8:][::-1]]

            if len(hist_list) >= int(cfg["min_history_for_warm"]):
                source = "warm"
                out = warm_pipeline(int(row["user_idx"]), hist_list, seen_set, chosen_variant, k)
            elif len(hist_list) >= 1:
                source = "warm_blend"
                warm_candidates = warm_pipeline(int(row["user_idx"]), hist_list, seen_set, chosen_variant, k)
                blended_ids = cold.blend_lukewarm([x["item_idx"] for x in warm_candidates], hist_list, k=k)
                out = [_format_item(i) for i in blended_ids]
            else:
                source = "popularity"
                out = [_format_item(i) for i in cold.popular(k)]
        elif liked_ids:
            source = "content_based"
            history_preview = [_format_item(i) for i in liked_ids]
            out = [_format_item(i) for i in cold.from_liked_items(liked_ids, k)]
        elif genre_list:
            source = "popularity_genre"
            out = [_format_item(i) for i in cold.popular(k, genres=genre_list)]
        else:
            source = "popularity"
            out = [_format_item(i) for i in cold.popular(k)]
    else:
        u_prof = bundle_engine.users.get(str(user_id))
        if u_prof and not liked_ids and not genre_list:
            hist_list = [int(x) for x in u_prof["history"]]
            history_preview = [dict(bundle_engine.item_by_idx[i]) for i in hist_list[-8:][::-1] if i in bundle_engine.item_by_idx]
            if len(hist_list) >= int(cfg["min_history_for_warm"]):
                source = "warm"
                out = bundle_engine.warm_recommend(u_prof, chosen_variant, k=k)
            elif len(hist_list) >= 1:
                source = "warm_blend"
                w_half = bundle_engine.warm_recommend(u_prof, chosen_variant, k=max(1, k // 2))
                excl = set(hist_list) | {x["item_idx"] for x in w_half}
                p_half = bundle_engine.popular(k=k - len(w_half), exclude=excl)
                out = (w_half + p_half)[:k]
            else:
                source = "popularity"
                out = bundle_engine.popular(k=k)
        elif liked_ids:
            source = "content_based"
            history_preview = [dict(bundle_engine.item_by_idx[i]) for i in liked_ids if i in bundle_engine.item_by_idx]
            out = bundle_engine.content_based(liked_ids, k=k)
        elif genre_list:
            source = "popularity_genre"
            out = bundle_engine.popular(k=k, genres=genre_list)
        else:
            source = "popularity"
            out = bundle_engine.popular(k=k)

    cache.set(user_id, k, chosen_variant, {"items": out, "history_preview": history_preview}, extra=extra_sig)
    return {
        "user_id": user_id,
        "variant": chosen_variant,
        "ranker_model": RANKER[chosen_variant],
        "source": source,
        "items": out,
        "history_preview": history_preview,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


@app.get("/api/compare")
def compare_variants(user_id: int = Query(..., description="Warm user_id to compare Variant A vs Variant B"), k: int = 10):
    res_a = recommend(user_id=user_id, k=k, liked="", genres="", variant="A", bypass_cache=True)
    res_b = recommend(user_id=user_id, k=k, liked="", genres="", variant="B", bypass_cache=True)
    set_a = {x["item_idx"] for x in res_a["items"]}
    set_b = {x["item_idx"] for x in res_b["items"]}
    return {
        "user_id": user_id,
        "source": res_a["source"],
        "history_preview": res_a["history_preview"],
        "overlap_at_k": len(set_a & set_b),
        "variant_A": {"model": "Wide & Deep (ranker_wd)", "items": res_a["items"], "latency_ms": res_a["latency_ms"]},
        "variant_B": {"model": "DCN-V2 (ranker)", "items": res_b["items"], "latency_ms": res_b["latency_ms"]},
    }


@app.get("/api/items")
def list_items(q: str = "", genre: str = "", limit: int = 40):
    if _HAS_FULL_ARTIFACTS:
        df = items.sort_values("n_pos", ascending=False)
        if q.strip():
            df = df[df["title"].str.contains(q.strip(), case=False, na=False)]
        if genre.strip():
            df = df[df["genres"].str.contains(genre.strip(), case=False, na=False)]
        sub = df.head(min(limit, 100))
        return {"genres": ALL_GENRES, "items": [_format_item(int(i)) for i in sub["item_idx"].values]}

    res = sorted(bundle_engine.items, key=lambda x: x["n_pos"], reverse=True)
    if q.strip():
        ql = q.strip().lower()
        res = [it for it in res if ql in it["title"].lower()]
    if genre.strip():
        gl = genre.strip().lower()
        res = [it for it in res if gl in it["genres"].lower()]
    return {"genres": ALL_GENRES, "items": res[: min(limit, 100)]}


@app.get("/api/users")
def sample_users():
    if _HAS_FULL_ARTIFACTS:
        warm_rows = state_df[state_df["history"].apply(len) >= int(cfg["min_history_for_warm"])].head(12)
        lukewarm_rows = state_df[(state_df["history"].apply(len) >= 1) & (state_df["history"].apply(len) < int(cfg["min_history_for_warm"]))].head(4)
        profiles = []
        for row in pd.concat([warm_rows, lukewarm_rows]).itertuples(index=False):
            uid = int(row.userId)
            hist = [int(x) for x in row.history]
            titles = [item_meta.get(i, {}).get("title", "?") for i in hist[-5:][::-1]]
            g_counts: dict[str, int] = {}
            for i in hist:
                for g in item_meta.get(i, {}).get("genres", "").split("|"):
                    if g and g != "(no genres listed)":
                        g_counts[g] = g_counts.get(g, 0) + 1
            top_g = [g for g, _ in sorted(g_counts.items(), key=lambda kv: -kv[1])[:3]]
            profiles.append({
                "user_id": uid,
                "history_count": len(hist),
                "segment": "warm" if len(hist) >= int(cfg["min_history_for_warm"]) else "lukewarm",
                "assigned_variant": variant_for(uid),
                "top_genres": top_g,
                "recent_titles": titles,
            })
        return {"users": profiles}

    profiles = []
    for u in list(bundle_engine.users.values())[:16]:
        uid = int(u["user_id"])
        hist = [int(x) for x in u["history"]]
        g_counts = {}
        for i in hist:
            it = bundle_engine.item_by_idx.get(i)
            if it:
                for g in it["genres"].split("|"):
                    if g and g != "(no genres listed)":
                        g_counts[g] = g_counts.get(g, 0) + 1
        top_g = [g for g, _ in sorted(g_counts.items(), key=lambda kv: -kv[1])[:3]]
        titles = [bundle_engine.item_by_idx.get(i, {}).get("title", "?") for i in hist[-5:][::-1]]
        profiles.append({
            "user_id": uid,
            "history_count": len(hist),
            "segment": "warm" if len(hist) >= int(cfg["min_history_for_warm"]) else "lukewarm",
            "assigned_variant": variant_for(uid),
            "top_genres": top_g,
            "recent_titles": titles,
        })
    return {"users": profiles}


@app.get("/api/metrics")
def get_metrics():
    if _HAS_FULL_ARTIFACTS:
        def _read_json(p: str):
            fp = pathlib.Path(p)
            return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}

        return {
            "metrics": _read_json("artifacts/metrics.json"),
            "faiss_benchmark": _read_json("artifacts/faiss_benchmark.json"),
            "split_summary": _read_json("artifacts/split_summary.json"),
            "retrieval_train": _read_json("artifacts/retrieval_train_info.json"),
            "ranking_train": _read_json("artifacts/ranking_train_info.json"),
            "cache": cache.stats(),
            "backend": backend_status(),
        }
    return {
        **bundle_engine.telemetry,
        "cache": cache.stats(),
        "backend": {"active_engine": "deploy_bundle_numpy", "triton_live": False},
    }


@app.get("/api/ab-results")
def get_ab_results():
    if _HAS_FULL_ARTIFACTS:
        ab_json = pathlib.Path("artifacts/ab_summary.json")
        if ab_json.exists():
            return json.loads(ab_json.read_text(encoding="utf-8"))
    elif bundle_engine and bundle_engine.telemetry.get("ab_summary"):
        return bundle_engine.telemetry["ab_summary"]
    return {"status": "not_run"}


if pathlib.Path("public").exists():
    app.mount("/", StaticFiles(directory="public", html=True), name="public")
