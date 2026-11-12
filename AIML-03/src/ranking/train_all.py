"""Train both A/B ranker variants: Wide & Deep (ranker_wd = Variant A) and DCN-V2 (ranker = Variant B)."""
import json
import numpy as np

from src.ranking.model import DCNRanker, WideDeepRanker
from src.ranking.train import train_ranker


def main():
    data = np.load("artifacts/ranking_data.npz")
    Xtr = {
        "user_idx": data["tr_user_idx"],
        "item_idx": data["tr_item_idx"],
        "dense": data["tr_dense"],
    }
    ytr = data["tr_y"]
    Xva = {
        "user_idx": data["va_user_idx"],
        "item_idx": data["va_item_idx"],
        "dense": data["va_dense"],
    }
    yva = data["va_y"]
    n_users = int(data["n_users"][0])
    n_items = int(data["n_items"][0])

    _, wd_info = train_ranker(WideDeepRanker, Xtr, ytr, Xva, yva, n_users, n_items, name="ranker_wd")
    _, dcn_info = train_ranker(DCNRanker, Xtr, ytr, Xva, yva, n_users, n_items, name="ranker")

    with open("artifacts/ranking_train_info.json", "w", encoding="utf-8") as f:
        json.dump({"ranker_wd": wd_info, "ranker": dcn_info}, f, indent=2)


if __name__ == "__main__":
    main()
