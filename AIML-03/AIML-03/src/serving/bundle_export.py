"""Export a compact (~2.5 MB) cloud deployment bundle of real trained weights, embeddings & metrics for Vercel/Netlify."""
import base64
import json
import pathlib
import faiss
import numpy as np
import pandas as pd
import yaml

from src.ranking.features import load_feature_arrays


def _b64_f16(arr: np.ndarray) -> dict:
    a16 = np.ascontiguousarray(arr, dtype=np.float16)
    return {
        "dtype": "float16",
        "shape": list(a16.shape),
        "b64": base64.b64encode(a16.tobytes()).decode("ascii"),
    }


def _b64_f32(arr: np.ndarray) -> dict:
    a32 = np.ascontiguousarray(arr, dtype=np.float32)
    return {
        "dtype": "float32",
        "shape": list(a32.shape),
        "b64": base64.b64encode(a32.tobytes()).decode("ascii"),
    }


def export_cloud_bundle() -> None:
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    items = pd.read_parquet("data/processed/items")
    items = items[items["item_idx"] > 0].sort_values("item_idx").reset_index(drop=True)
    state_df = pd.read_parquet("data/processed/user_state_val_end")

    # Curate 24 diverse warm & lukewarm real MovieLens user profiles
    warm_pool = state_df[(state_df["user_idx"] > 0) & (state_df["history"].apply(len) >= int(cfg["min_history_for_warm"]))].head(20)
    lukewarm_pool = state_df[(state_df["history"].apply(len) >= 1) & (state_df["history"].apply(len) < int(cfg["min_history_for_warm"]))].head(6)
    sampled_users_df = pd.concat([warm_pool, lukewarm_pool]).drop_duplicates(subset=["userId"]).reset_index(drop=True)

    needed_item_idxs = set()
    for h in sampled_users_df["history"].values:
        needed_item_idxs.update(int(x) for x in h if int(x) > 0)

    # Top 1,800 popular items + all items in curated user histories
    top_pop_idxs = set(items.sort_values(["n_pos", "n_pos_recent"], ascending=False).head(1800)["item_idx"].astype(int).tolist())
    active_item_idxs = sorted(top_pop_idxs | needed_item_idxs)

    sub_items = items[items["item_idx"].isin(active_item_idxs)].sort_values("item_idx").reset_index(drop=True)
    active_ids = sub_items["item_idx"].astype(int).tolist()

    emb_full = np.load("artifacts/item_embeddings.npy").astype("float32")
    faiss.normalize_L2(emb_full)
    sub_emb = emb_full[[i - 1 for i in active_ids]]

    UF, IF, n_g = load_feature_arrays()
    sub_IF = IF[active_ids]

    ut_w = dict(np.load("artifacts/user_tower_weights.npz"))
    wd_w = dict(np.load("artifacts/ranker_wd_weights.npz"))
    dcn_w = dict(np.load("artifacts/ranker_weights.npz"))

    # Compact indexed lookup for active items (0 = padding, 1..M = active items)
    item_records = []
    for row in sub_items.itertuples(index=False):
        item_records.append({
            "item_idx": int(row.item_idx),
            "movieId": int(row.movieId),
            "title": str(row.title),
            "year": int(row.year) if pd.notna(row.year) else 1990,
            "genres": str(row.genres),
            "avg_rating": round(float(row.avg_rating), 2),
            "n_pos": int(row.n_pos),
            "n_pos_recent": int(row.n_pos_recent),
        })

    # Active user profiles with their UF vector and user embeddings
    user_records = {}
    for idx_pos, row in enumerate(sampled_users_df.itertuples(index=False)):
        uid = int(row.userId)
        u_idx = int(row.user_idx)
        hist = [int(x) for x in row.history if int(x) in top_pop_idxs or int(x) in needed_item_idxs]
        seen_raw = row.seen_items if hasattr(row, "seen_items") else hist
        seen = [int(x) for x in seen_raw if int(x) in top_pop_idxs or int(x) in needed_item_idxs]
        user_records[str(uid)] = {
            "user_id": uid,
            "user_idx": u_idx,
            "slot": idx_pos + 1,
            "history": hist,
            "seen_items": seen,
        }

    u_slots_idx = [0] + [int(r.user_idx) for r in sampled_users_df.itertuples(index=False)]
    sub_UF = UF[np.clip(u_slots_idx, 0, len(UF) - 1)]
    sub_ut_user_emb = ut_w["user_emb"][np.clip(u_slots_idx, 0, len(ut_w["user_emb"]) - 1)]
    sub_wd_ue = wd_w["ue"][np.clip(u_slots_idx, 0, len(wd_w["ue"]) - 1)]
    sub_dcn_ue = dcn_w["ue"][np.clip(u_slots_idx, 0, len(dcn_w["ue"]) - 1)]

    i_slots_idx = [0] + active_ids
    sub_ut_hist_emb = ut_w["hist_emb"][i_slots_idx]
    sub_wd_ie = wd_w["ie"][i_slots_idx]
    sub_dcn_ie = dcn_w["ie"][i_slots_idx]

    def _read_json(p: str):
        fp = pathlib.Path(p)
        return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}

    bundle = {
        "config": cfg,
        "n_genres": int(n_g),
        "n_total_catalog": int(len(items)),
        "n_total_warm_users": int(len(state_df)),
        "items": item_records,
        "users": user_records,
        "item_embeddings": _b64_f16(sub_emb),
        "IF": _b64_f16(sub_IF),
        "UF": _b64_f32(sub_UF),
        "user_tower": {
            "user_emb": _b64_f16(sub_ut_user_emb),
            "hist_emb": _b64_f16(sub_ut_hist_emb),
            "mlp_w0": _b64_f16(ut_w["mlp_w0"]),
            "mlp_b0": _b64_f32(ut_w["mlp_b0"]),
            "mlp_w1": _b64_f16(ut_w["mlp_w1"]),
            "mlp_b1": _b64_f32(ut_w["mlp_b1"]),
        },
        "ranker_wd": {
            "ue": _b64_f16(sub_wd_ue),
            "ie": _b64_f16(sub_wd_ie),
            "bn_scale": _b64_f32(wd_w["bn_scale"]),
            "bn_offset": _b64_f32(wd_w["bn_offset"]),
            "wide_w": _b64_f32(wd_w["wide_w"]),
            "wide_b": _b64_f32(wd_w["wide_b"]),
            "n_deep": int(wd_w["n_deep"][0]),
            "deep": [
                {"w": _b64_f16(wd_w[f"deep_w_{i}"]), "b": _b64_f32(wd_w[f"deep_b_{i}"])}
                for i in range(int(wd_w["n_deep"][0]))
            ],
        },
        "ranker_dcn": {
            "ue": _b64_f16(sub_dcn_ue),
            "ie": _b64_f16(sub_dcn_ie),
            "bn_scale": _b64_f32(dcn_w["bn_scale"]),
            "bn_offset": _b64_f32(dcn_w["bn_offset"]),
            "n_cross": int(dcn_w["n_cross"][0]),
            "cross": [
                {"w": _b64_f16(dcn_w[f"cross_w_{i}"]), "b": _b64_f32(dcn_w[f"cross_b_{i}"])}
                for i in range(int(dcn_w["n_cross"][0]))
            ],
            "n_deep": int(dcn_w["n_deep"][0]),
            "deep": [
                {"w": _b64_f16(dcn_w[f"deep_w_{i}"]), "b": _b64_f32(dcn_w[f"deep_b_{i}"])}
                for i in range(int(dcn_w["n_deep"][0]))
            ],
            "out_w": _b64_f32(dcn_w["out_w"]),
            "out_b": _b64_f32(dcn_w["out_b"]),
        },
        "telemetry": {
            "metrics": _read_json("artifacts/metrics.json"),
            "faiss_benchmark": _read_json("artifacts/faiss_benchmark.json"),
            "split_summary": _read_json("artifacts/split_summary.json"),
            "retrieval_train": _read_json("artifacts/retrieval_train_info.json"),
            "ranking_train": _read_json("artifacts/ranking_train_info.json"),
            "ab_summary": _read_json("artifacts/ab_summary.json"),
        },
    }

    out_dir = pathlib.Path("deploy_bundle")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bundle.json"
    out_file.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Exported cloud deployment bundle to {out_file} ({size_mb:.2f} MB, {len(item_records)} movies, {len(user_records)} user profiles)")


if __name__ == "__main__":
    export_cloud_bundle()
