"""NVIDIA Triton HTTP inference client with automatic local weight fallback when Triton is offline."""
import os
import time
import numpy as np

from src.ranking.export import infer_ranker_numpy
from src.retrieval.export import infer_user_tower_numpy

try:
    import tritonclient.http as httpclient
    HAS_TRITONCLIENT = True
except ImportError:
    httpclient = None
    HAS_TRITONCLIENT = False

DT = {np.dtype("int32"): "INT32", np.dtype("float32"): "FP32"}

_TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
_triton_client = None
_triton_ready = False
_last_check_ts = 0.0
_local_weights: dict[str, dict] = {}


def _load_local_weights(model: str) -> dict:
    if model not in _local_weights:
        path_map = {
            "user_tower": "artifacts/user_tower_weights.npz",
            "ranker": "artifacts/ranker_weights.npz",
            "ranker_wd": "artifacts/ranker_wd_weights.npz",
        }
        _local_weights[model] = dict(np.load(path_map[model]))
    return _local_weights[model]


def _is_triton_live() -> bool:
    global _triton_client, _triton_ready, _last_check_ts
    if os.getenv("DISABLE_TRITON", "0") == "1" or not HAS_TRITONCLIENT:
        return False
    now = time.time()
    if now - _last_check_ts < 15.0:
        return _triton_ready
    _last_check_ts = now
    try:
        _triton_client = httpclient.InferenceServerClient(_TRITON_URL, connection_timeout=0.3, network_timeout=0.5)
        _triton_ready = bool(_triton_client.is_server_ready())
    except Exception:
        _triton_ready = False
    return _triton_ready


def infer(model: str, inputs: dict[str, np.ndarray], output: str) -> np.ndarray:
    if _is_triton_live():
        try:
            ins = []
            for name, arr in inputs.items():
                t = httpclient.InferInput(name, list(arr.shape), DT[arr.dtype])
                t.set_data_from_numpy(arr)
                ins.append(t)
            res = _triton_client.infer(model, ins, outputs=[httpclient.InferRequestedOutput(output)])
            return res.as_numpy(output)
        except Exception:
            pass

    # Local high-speed NumPy forward pass using exported SavedModel weights
    w = _load_local_weights(model)
    if model == "user_tower":
        return infer_user_tower_numpy(w, inputs["user_idx"], inputs["history"])
    return infer_ranker_numpy(w, inputs["user_idx"], inputs["item_idx"], inputs["dense"])


def backend_status() -> dict:
    return {
        "triton_url": _TRITON_URL,
        "triton_live": _is_triton_live(),
        "active_engine": "triton_http" if _is_triton_live() else "local_exported_weights",
    }
