"""Pure-NumPy runtime loader for deploy_bundle/bundle.json so FastAPI runs out-of-the-box before full training."""
import base64
import json
import pathlib
import numpy as np


def _decode_tensor(spec: dict) -> np.ndarray:
    raw = base64.b64decode(spec["b64"])
    dt = np.float16 if spec["dtype"] == "float16" else np.float32
    arr = np.frombuffer(raw, dtype=dt).reshape(spec["shape"])
    return arr.astype("float32")


class BundleEngine:
    """Executes the exact Two-Tower + Cosine ANN + DCN-V2 / Wide&Deep pipeline from deploy_bundle/bundle.json."""

    def __init__(self, bundle_path: str = "deploy_bundle/bundle.json"):
        raw = json.loads(pathlib.Path(bundle_path).read_text(encoding="utf-8"))
        self.cfg = raw["config"]
        self.n_genres = int(raw["n_genres"])
        self.n_total_catalog = int(raw["n_total_catalog"])
        self.n_total_warm_users = int(raw["n_total_warm_users"])
        self.items = raw["items"]
        self.item_by_idx = {int(it["item_idx"]): it for it in self.items}
        self.pos_by_idx = {int(it["item_idx"]): pos for pos, it in enumerate(self.items)}
        self.users = raw["users"]
        self.telemetry = raw["telemetry"]

        self.all_genres = sorted(
            {
                g
                for it in self.items
                for g in it["genres"].split("|")
                if g and g != "(no genres listed)"
            }
        )
        self.sorted_recent = sorted(
            self.items, key=lambda x: (x.get("n_pos_recent", 0), x.get("n_pos", 0)), reverse=True
        )
        self.global_top = [int(x["item_idx"]) for x in self.sorted_recent[:200]]

        self.item_emb = _decode_tensor(raw["item_embeddings"])
        norms = np.linalg.norm(self.item_emb, axis=1, keepdims=True)
        self.item_emb = self.item_emb / np.maximum(norms, 1e-12)

        self.IF = _decode_tensor(raw["IF"])
        self.UF = _decode_tensor(raw["UF"])

        ut = raw["user_tower"]
        self.ut = {
            "user_emb": _decode_tensor(ut["user_emb"]),
            "hist_emb": _decode_tensor(ut["hist_emb"]),
            "mlp_w0": _decode_tensor(ut["mlp_w0"]),
            "mlp_b0": _decode_tensor(ut["mlp_b0"]),
            "mlp_w1": _decode_tensor(ut["mlp_w1"]),
            "mlp_b1": _decode_tensor(ut["mlp_b1"]),
        }

        wd = raw["ranker_wd"]
        self.wd = {
            "ue": _decode_tensor(wd["ue"]),
            "ie": _decode_tensor(wd["ie"]),
            "bn_scale": _decode_tensor(wd["bn_scale"]),
            "bn_offset": _decode_tensor(wd["bn_offset"]),
            "wide_w": _decode_tensor(wd["wide_w"]),
            "wide_b": _decode_tensor(wd["wide_b"]),
            "deep": [(_decode_tensor(l["w"]), _decode_tensor(l["b"])) for l in wd["deep"]],
        }

        dcn = raw["ranker_dcn"]
        self.dcn = {
            "ue": _decode_tensor(dcn["ue"]),
            "ie": _decode_tensor(dcn["ie"]),
            "bn_scale": _decode_tensor(dcn["bn_scale"]),
            "bn_offset": _decode_tensor(dcn["bn_offset"]),
            "cross": [(_decode_tensor(l["w"]), _decode_tensor(l["b"])) for l in dcn["cross"]],
            "deep": [(_decode_tensor(l["w"]), _decode_tensor(l["b"])) for l in dcn["deep"]],
            "out_w": _decode_tensor(dcn["out_w"]),
            "out_b": _decode_tensor(dcn["out_b"]),
        }

    def popular(self, k: int = 10, genres: list[str] | None = None, exclude: set[int] | None = None) -> list[dict]:
        excl = exclude or set()
        if not genres:
            ids = [i for i in self.global_top if i not in excl][:k]
            return [dict(self.item_by_idx[i]) for i in ids]
        lists = [
            [int(it["item_idx"]) for it in self.sorted_recent if g.lower() in it["genres"].lower()][:100]
            for g in genres
        ]
        out, used, idx = [], set(excl), 0
        while len(out) < k and any(idx < len(lst) for lst in lists):
            for lst in lists:
                if idx < len(lst) and lst[idx] not in used:
                    used.add(lst[idx])
                    out.append(dict(self.item_by_idx[lst[idx]]))
                    if len(out) >= k:
                        break
            idx += 1
        return out

    def content_based(self, liked_ids: list[int], k: int = 10) -> list[dict]:
        positions = [self.pos_by_idx[i] for i in liked_ids if i in self.pos_by_idx]
        if not positions:
            return self.popular(k=k)
        q = self.item_emb[positions].mean(axis=0)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self.item_emb @ q
        excl = set(liked_ids)
        order = np.argsort(-sims)
        out = []
        for pos in order:
            it = self.items[int(pos)]
            if int(it["item_idx"]) not in excl:
                rec = dict(it)
                rec["retrieval_score"] = round(float(sims[int(pos)]), 4)
                out.append(rec)
                if len(out) >= k:
                    break
        return out

    def warm_recommend(self, u_profile: dict, variant: str, k: int = 10) -> list[dict]:
        slot = int(u_profile["slot"])
        hist = [int(x) for x in u_profile["history"]][-20:]
        h_slots = [self.pos_by_idx[i] + 1 for i in hist if i in self.pos_by_idx]
        u_vec = self.ut["user_emb"][slot]
        if h_slots:
            h_mean = self.ut["hist_emb"][h_slots].mean(axis=0)
        else:
            h_mean = np.zeros_like(u_vec)
        x = np.concatenate([u_vec, h_mean])
        h1 = np.maximum(0.0, x @ self.ut["mlp_w0"] + self.ut["mlp_b0"])
        q = h1 @ self.ut["mlp_w1"] + self.ut["mlp_b1"]
        q = q / max(float(np.linalg.norm(q)), 1e-12)

        sims = self.item_emb @ q
        seen = set(int(x) for x in u_profile.get("seen_items", hist))
        cand_pos = [int(p) for p in np.argsort(-sims) if int(self.items[int(p)]["item_idx"]) not in seen][: int(self.cfg.get("retrieval_k", 300))]
        if not cand_pos:
            return self.popular(k=k, exclude=seen)

        rs = sims[cand_pos]
        it_f = self.IF[cand_pos]
        u_f = np.repeat(self.UF[slot : slot + 1], len(cand_pos), axis=0)
        cross_g = (u_f[:, -self.n_genres :] * it_f[:, -self.n_genres :]).sum(axis=1, keepdims=True)
        hl_col = np.full((len(cand_pos), 1), np.log1p(len(hist)), dtype="float32")
        dense = np.hstack([u_f, it_f, rs.reshape(-1, 1), hl_col, cross_g]).astype("float32")

        i_slots = np.array([p + 1 for p in cand_pos], dtype="int32")
        if variant == "A":
            m = self.wd
            d_norm = dense * m["bn_scale"] + m["bn_offset"]
            x0 = np.hstack([np.repeat(m["ue"][slot : slot + 1], len(cand_pos), axis=0), m["ie"][i_slots], d_norm])
            wide_l = d_norm @ m["wide_w"] + m["wide_b"]
            h = x0
            for idx, (dw, db) in enumerate(m["deep"]):
                h = h @ dw + db
                if idx < len(m["deep"]) - 1:
                    h = np.maximum(0.0, h)
            logits = (wide_l + h).ravel()
        else:
            m = self.dcn
            d_norm = dense * m["bn_scale"] + m["bn_offset"]
            x0 = np.hstack([np.repeat(m["ue"][slot : slot + 1], len(cand_pos), axis=0), m["ie"][i_slots], d_norm])
            xc = x0
            for cw, cb in m["cross"]:
                xc = x0 * (xc @ cw + cb) + xc
            h = x0
            for dw, db in m["deep"]:
                h = np.maximum(0.0, h @ dw + db)
            logits = (np.hstack([xc, h]) @ m["out_w"] + m["out_b"]).ravel()

        order = np.argsort(-logits)[:k]
        out = []
        for idx in order:
            p = cand_pos[int(idx)]
            rec = dict(self.items[p])
            rec["score"] = round(float(logits[int(idx)]), 4)
            rec["retrieval_score"] = round(float(rs[int(idx)]), 4)
            out.append(rec)
        return out
