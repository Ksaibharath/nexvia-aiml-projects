"""Training routine for DCN-V2 and Wide & Deep ranking models."""
import time
import numpy as np
import tensorflow as tf

from src.ranking.export import export_ranker


def make_ds(X: dict, y: np.ndarray, bs: int = 2048, shuffle: bool = False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y.reshape(-1, 1).astype("float32")))
    if shuffle:
        ds = ds.shuffle(min(300_000, max(len(y), 1)), seed=42, reshuffle_each_iteration=True)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)


def train_ranker(cls, Xtr: dict, ytr: np.ndarray, Xva: dict, yva: np.ndarray, n_users: int, n_items: int, name: str):
    tf.keras.utils.set_random_seed(42)
    m = cls(n_users, n_items)
    n_dense = int(Xtr["dense"].shape[1])
    # Warm-build variables
    _ = m(
        {
            "user_idx": tf.zeros([2], dtype=tf.int32),
            "item_idx": tf.zeros([2], dtype=tf.int32),
            "dense": tf.zeros([2, n_dense], dtype=tf.float32),
        },
        training=False,
    )
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.AUC(from_logits=True, name="auc")],
    )
    t0 = time.time()
    hist = m.fit(
        make_ds(Xtr, ytr, shuffle=True),
        validation_data=make_ds(Xva, yva),
        epochs=8,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=2,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    dt = round(time.time() - t0, 2)
    export_ranker(m, n_dense=n_dense, name=name)
    info = {
        "name": name,
        "best_val_auc": round(float(max(hist.history["val_auc"])), 4),
        "epochs_ran": len(hist.history["auc"]),
        "train_seconds": dt,
        "n_dense": n_dense,
    }
    print(f"Exported {name}: best_val_auc={info['best_val_auc']:.4f} in {dt}s")
    return m, info
