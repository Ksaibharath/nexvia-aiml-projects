"""Cold-start fallback strategies: global popularity, genre-diversified popularity, lukewarm blend, and content-based onboarding."""
import faiss
import numpy as np
import pandas as pd


class ColdStart:
    def __init__(self, items_df: pd.DataFrame, item_emb: np.ndarray, index):
        self.items = items_df.sort_values("item_idx").reset_index(drop=True)
        self.emb = np.ascontiguousarray(item_emb, dtype="float32")
        self.index = index
        self.gcols = sorted([c for c in items_df.columns if c.startswith("g_")])

        # Sort by recent popularity first, breaking ties with total positive ratings
        top = items_df.sort_values(["n_pos_recent", "n_pos"], ascending=[False, False])
        self.global_top = top["item_idx"].astype(int).head(300).tolist()
        self.genre_top = {
            g: top[top[g] == 1]["item_idx"].astype(int).head(150).tolist()
            for g in self.gcols
        }

        # Cold-start items (low interaction count) eligible for optional exploration slot
        cold_items = items_df[(items_df["n_ratings"] <= 10) & (items_df["year"] >= 2010)]
        self.exploration_pool = cold_items["item_idx"].astype(int).head(200).tolist()

    def popular(self, k: int = 10, genres: list[str] | None = None, exclude=()) -> list[int]:
        """Global popularity (diversified across top genres) or round-robin inside chosen genres."""
        excl = set(int(x) for x in exclude)
        if not genres:
            out = [i for i in self.global_top if i not in excl]
            return out[:k]

        norm_genres = []
        for g in genres:
            g_clean = g if g.startswith("g_") else "g_" + g.replace("-", "_").replace(" ", "_")
            if g_clean in self.genre_top:
                norm_genres.append(g_clean)

        if not norm_genres:
            out = [i for i in self.global_top if i not in excl]
            return out[:k]

        lists = [self.genre_top[g] for g in norm_genres]
        out, i = [], 0
        while len(out) < k and any(i < len(lst) for lst in lists):
            for lst in lists:
                if i < len(lst):
                    cand = lst[i]
                    if cand not in excl and cand not in out:
                        out.append(cand)
                        if len(out) >= k:
                            break
            i += 1
        return out[:k]

    def from_liked_items(self, liked_item_idx: list[int], k: int = 10, exclude=()) -> list[int]:
        """Content-based onboarding: mean L2-normalised embedding of movies picked by a new user."""
        valid = [int(i) for i in liked_item_idx if 1 <= int(i) <= len(self.emb)]
        if not valid:
            return self.popular(k=k, exclude=exclude)

        v = self.emb[[i - 1 for i in valid]].mean(axis=0, keepdims=True).astype("float32")
        faiss.normalize_L2(v)
        excl = set(valid) | set(int(x) for x in exclude)
        search_k = min(self.index.ntotal, k + len(excl) + 50)
        _, pos = self.index.search(v, search_k)
        out = [int(p) + 1 for p in pos[0] if p >= 0 and (int(p) + 1) not in excl]
        return out[:k]

    def blend_lukewarm(self, warm_ids: list[int], history_items: list[int], k: int = 10) -> list[int]:
        """Known user with 1-4 liked movies: 50% neural pipeline + 50% favourite-genre popularity."""
        excl = set(int(x) for x in history_items)
        sub = self.items[self.items["item_idx"].isin(excl)]
        fav_genres = []
        if len(sub) and self.gcols:
            g_means = sub[self.gcols].mean().sort_values(ascending=False)
            fav_genres = [g for g, val in g_means.head(3).items() if val > 0]

        half = max(1, k // 2)
        out = [int(i) for i in warm_ids if int(i) not in excl][:half]
        pop_fill = self.popular(k=k, genres=fav_genres, exclude=excl | set(out))
        for p in pop_fill:
            if len(out) >= k:
                break
            if p not in out:
                out.append(p)
        return out[:k]
