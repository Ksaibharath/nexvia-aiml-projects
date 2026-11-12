"""Export the trained UserTower as a Triton TensorFlow SavedModel + compact NumPy weights."""
import pathlib
import numpy as np

HIST = 20


def export_user_tower(
    user_tower,
    path: str = "triton/model_repository/user_tower/1/model.savedmodel",
    npz_path: str = "artifacts/user_tower_weights.npz",
) -> None:
    import tensorflow as tf

    class _UserTowerServeModule(tf.Module):
        def __init__(self, ut):
            super().__init__()
            self.user_tower = ut

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, 1], tf.int32, name="user_idx"),
                tf.TensorSpec([None, HIST], tf.int32, name="history"),
            ]
        )
        def serve(self, user_idx, history):
            emb = self.user_tower(
                {"user_idx": tf.squeeze(user_idx, axis=-1), "history": history},
                training=False,
            )
            return {"embedding": emb}

    out_dir = pathlib.Path(path)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    mod = _UserTowerServeModule(user_tower)
    tf.saved_model.save(mod, str(out_dir), signatures={"serving_default": mod.serve})

    # Also save compact NumPy weights for zero-dependency fast CPU/Serverless inference
    pathlib.Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    d0, d1 = user_tower.mlp.layers[0], user_tower.mlp.layers[1]
    w0, b0 = d0.get_weights()
    w1, b1 = d1.get_weights()
    np.savez_compressed(
        npz_path,
        user_emb=user_tower.user_emb.get_weights()[0].astype("float32"),
        hist_emb=user_tower.hist_emb.get_weights()[0].astype("float32"),
        mlp_w0=w0.astype("float32"),
        mlp_b0=b0.astype("float32"),
        mlp_w1=w1.astype("float32"),
        mlp_b1=b1.astype("float32"),
    )


def infer_user_tower_numpy(weights_npz: dict, user_idx: np.ndarray, history: np.ndarray) -> np.ndarray:
    """Exact NumPy forward pass matching UserTower.call()."""
    u_idx = np.clip(user_idx.reshape(-1), 0, len(weights_npz["user_emb"]) - 1)
    h_idx = np.clip(history, 0, len(weights_npz["hist_emb"]) - 1)
    u = weights_npz["user_emb"][u_idx]                      # (B, D)
    h = weights_npz["hist_emb"][h_idx]                      # (B, L, D)
    mask = (h_idx > 0).astype("float32")[..., None]         # (B, L, 1)
    h_mean = (h * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
    x = np.concatenate([u, h_mean], axis=-1)
    h1 = np.maximum(0.0, x @ weights_npz["mlp_w0"] + weights_npz["mlp_b0"])
    out = h1 @ weights_npz["mlp_w1"] + weights_npz["mlp_b1"]
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return (out / np.maximum(norms, 1e-12)).astype("float32")
