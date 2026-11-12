"""Shared pytest fixtures."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cold_start():
    faiss = pytest.importorskip("faiss")
    from src.serving.cold_start import ColdStart
    rng = np.random.RandomState(0)
    n = 30
    items_df = pd.DataFrame({
        "item_idx": np.arange(1, n + 1, dtype="int32"),
        "movieId": np.arange(101, 101 + n, dtype="int32"),
        "title": [f"Movie {i} (2015)" for i in range(1, n + 1)],
        "year": [2015] * n,
        "genres": ["Action|Sci-Fi" if i % 2 == 0 else "Comedy|Drama" for i in range(1, n + 1)],
        "g_Action": [1.0 if i % 2 == 0 else 0.0 for i in range(1, n + 1)],
        "g_Sci_Fi": [1.0 if i % 2 == 0 else 0.0 for i in range(1, n + 1)],
        "g_Comedy": [0.0 if i % 2 == 0 else 1.0 for i in range(1, n + 1)],
        "g_Drama": [0.0 if i % 2 == 0 else 1.0 for i in range(1, n + 1)],
        "n_ratings": np.arange(50, 50 + n),
        "avg_rating": [4.1] * n,
        "n_pos": np.arange(20, 20 + n),
        "n_pos_recent": np.arange(1, 1 + n),
    })
    emb = rng.randn(n, 64).astype("float32")
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(64)
    idx.add(emb)
    return ColdStart(items_df, emb, idx)
