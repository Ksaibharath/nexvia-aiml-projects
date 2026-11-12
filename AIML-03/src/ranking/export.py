"""Export trained rankers to Triton SavedModel + config.pbtxt + compact NumPy weights."""
import pathlib
import numpy as np


def _bn_params(bn_layer):
    gamma, beta, mean, var = bn_layer.get_weights()
    eps = bn_layer.epsilon
    scale = gamma / np.sqrt(var + eps)
    offset = beta - mean * scale
    return scale.astype("float32"), offset.astype("float32")


def export_ranker(model, n_dense: int, name: str) -> None:
    import tensorflow as tf

    class _RankerServeModule(tf.Module):
        def __init__(self, m, nd: int):
            super().__init__()
            self.model = m
            self.n_dense = nd

            @tf.function(
                input_signature=[
                    tf.TensorSpec([None, 1], tf.int32, name="user_idx"),
                    tf.TensorSpec([None, 1], tf.int32, name="item_idx"),
                    tf.TensorSpec([None, nd], tf.float32, name="dense"),
                ]
            )
            def _serve(user_idx, item_idx, dense):
                logit = self.model(
                    {
                        "user_idx": tf.squeeze(user_idx, axis=-1),
                        "item_idx": tf.squeeze(item_idx, axis=-1),
                        "dense": dense,
                    },
                    training=False,
                )
                return {"score": logit}

            self.serve = _serve

    sm_path = pathlib.Path(f"triton/model_repository/{name}/1/model.savedmodel")
    sm_path.parent.mkdir(parents=True, exist_ok=True)
    mod = _RankerServeModule(model, n_dense)
    tf.saved_model.save(mod, str(sm_path), signatures={"serving_default": mod.serve})

    pbtxt_path = pathlib.Path(f"triton/model_repository/{name}/config.pbtxt")
    pbtxt_content = f'''name: "{name}"
platform: "tensorflow_savedmodel"
max_batch_size: 512
input [
  {{ name: "user_idx" data_type: TYPE_INT32 dims: [ 1 ] }},
  {{ name: "item_idx" data_type: TYPE_INT32 dims: [ 1 ] }},
  {{ name: "dense"    data_type: TYPE_FP32  dims: [ {n_dense} ] }}
]
output [ {{ name: "score" data_type: TYPE_FP32 dims: [ 1 ] }} ]
dynamic_batching {{ max_queue_delay_microseconds: 2000 }}
'''
    pbtxt_path.write_text(pbtxt_content, encoding="utf-8")

    # Export exact NumPy inference weights
    bn_scale, bn_offset = _bn_params(model.norm)
    payload = {
        "arch": np.array([ "dcn" if hasattr(model, "cross") else "wd" ]),
        "ue": model.ue.get_weights()[0].astype("float32"),
        "ie": model.ie.get_weights()[0].astype("float32"),
        "bn_scale": bn_scale,
        "bn_offset": bn_offset,
    }
    if hasattr(model, "cross"):
        for idx, c_layer in enumerate(model.cross):
            cw, cb = c_layer.get_weights()
            payload[f"cross_w_{idx}"] = cw.astype("float32")
            payload[f"cross_b_{idx}"] = cb.astype("float32")
        for idx, d_layer in enumerate(model.deep.layers):
            dw, db = d_layer.get_weights()
            payload[f"deep_w_{idx}"] = dw.astype("float32")
            payload[f"deep_b_{idx}"] = db.astype("float32")
        ow, ob = model.out.get_weights()
        payload["out_w"] = ow.astype("float32")
        payload["out_b"] = ob.astype("float32")
        payload["n_cross"] = np.array([len(model.cross)], dtype="int32")
        payload["n_deep"] = np.array([len(model.deep.layers)], dtype="int32")
    else:
        ww, wb = model.wide.get_weights()
        payload["wide_w"] = ww.astype("float32")
        payload["wide_b"] = wb.astype("float32")
        for idx, d_layer in enumerate(model.deep.layers):
            dw, db = d_layer.get_weights()
            payload[f"deep_w_{idx}"] = dw.astype("float32")
            payload[f"deep_b_{idx}"] = db.astype("float32")
        payload["n_deep"] = np.array([len(model.deep.layers)], dtype="int32")

    pathlib.Path("artifacts").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"artifacts/{name}_weights.npz", **payload)


def infer_ranker_numpy(weights_npz: dict, user_idx: np.ndarray, item_idx: np.ndarray, dense: np.ndarray) -> np.ndarray:
    """Exact NumPy forward pass matching DCNRanker / WideDeepRanker."""
    u = weights_npz["ue"][np.clip(user_idx.reshape(-1), 0, len(weights_npz["ue"]) - 1)]
    it = weights_npz["ie"][np.clip(item_idx.reshape(-1), 0, len(weights_npz["ie"]) - 1)]
    d = dense * weights_npz["bn_scale"] + weights_npz["bn_offset"]
    x0 = np.concatenate([u, it, d], axis=-1)

    arch = str(weights_npz["arch"][0])
    if arch == "dcn":
        x = x0
        n_cross = int(weights_npz["n_cross"][0])
        for idx in range(n_cross):
            proj = x @ weights_npz[f"cross_w_{idx}"] + weights_npz[f"cross_b_{idx}"]
            x = x0 * proj + x
        h = x0
        n_deep = int(weights_npz["n_deep"][0])
        for idx in range(n_deep):
            h = np.maximum(0.0, h @ weights_npz[f"deep_w_{idx}"] + weights_npz[f"deep_b_{idx}"])
        cat = np.concatenate([x, h], axis=-1)
        return (cat @ weights_npz["out_w"] + weights_npz["out_b"]).astype("float32")
    else:
        wide_logit = d @ weights_npz["wide_w"] + weights_npz["wide_b"]
        h = x0
        n_deep = int(weights_npz["n_deep"][0])
        for idx in range(n_deep - 1):
            h = np.maximum(0.0, h @ weights_npz[f"deep_w_{idx}"] + weights_npz[f"deep_b_{idx}"])
        deep_logit = h @ weights_npz[f"deep_w_{n_deep - 1}"] + weights_npz[f"deep_b_{n_deep - 1}"]
        return (wide_logit + deep_logit).astype("float32")
