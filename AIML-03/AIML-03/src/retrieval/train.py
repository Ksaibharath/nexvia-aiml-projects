"""Train the Two-Tower retrieval model and export item embeddings + Triton UserTower."""
import json
import pathlib
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.retrieval.data import HIST, load_items, load_split, to_dataset
from src.retrieval.export import export_user_tower
from src.retrieval.model import HAS_TFRS, ItemTower, TwoTower, UserTower


def main():
    tf.keras.utils.set_random_seed(42)
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dim = int(cfg["embedding_dim"])

    items, G = load_items()
    n_items = int(items["item_idx"].max())
    n_users = int(pd.read_parquet("data/processed/user_features")["user_idx"].max())

    train_df = load_split("train")
    val_df = load_split("val", max_rows=200_000)

    # Empirical item sampling probabilities from train split for LogQ frequency correction
    counts = np.bincount(train_df["item_idx"].values, minlength=n_items + 1).astype("float64")
    item_probs = np.maximum(counts / max(counts.sum(), 1.0), 1e-7).astype("float32")

    batch_size = 1024
    train_ds = to_dataset(train_df, batch=batch_size, shuffle=True)
    val_ds = to_dataset(val_df, batch=batch_size, shuffle=False)

    user_t = UserTower(n_users, n_items, dim)
    item_t = ItemTower(n_items, G, dim)
    # Build layers explicitly with dummy call before training/saving
    _ = user_t({"user_idx": tf.zeros([2], dtype=tf.int32), "history": tf.zeros([2, HIST], dtype=tf.int32)})
    _ = item_t(tf.zeros([2], dtype=tf.int32))

    all_items = np.arange(1, n_items + 1, dtype="int32")
    model = TwoTower(user_t, item_t, all_items, item_probs=item_probs, temperature=0.07)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3))

    t0 = time.time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_total_loss",
                mode="min",
                patience=2,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    train_secs = round(time.time() - t0, 2)

    pathlib.Path("artifacts").mkdir(parents=True, exist_ok=True)
    emb = item_t.predict(all_items, batch_size=4096, verbose=0).astype("float32")
    np.save("artifacts/item_embeddings.npy", emb)
    np.save("artifacts/item_idx.npy", all_items)

    export_user_tower(user_t)

    train_info = {
        "embedding_dim": dim,
        "batch_size": batch_size,
        "learning_rate": 2e-3,
        "temperature": 0.07,
        "epochs_ran": len(hist.history["total_loss"]),
        "final_train_loss": float(hist.history["total_loss"][-1]),
        "best_val_loss": float(min(hist.history["val_total_loss"])),
        "train_seconds": train_secs,
        "n_users": n_users,
        "n_items": n_items,
    }
    with open("artifacts/retrieval_train_info.json", "w", encoding="utf-8") as f:
        json.dump(train_info, f, indent=2)

    print("saved item_embeddings:", emb.shape, "in", train_secs, "s")


if __name__ == "__main__":
    main()
