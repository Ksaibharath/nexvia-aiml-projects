"""Core offline & online recommendation evaluation metrics."""
import numpy as np


def recall_at_k(recs, relevant, k: int) -> float:
    rel = set(relevant)
    return len(set(recs[:k]) & rel) / len(rel) if rel else np.nan


def ndcg_at_k(recs, relevant, k: int) -> float:
    rel = set(relevant)
    if not rel:
        return np.nan
    dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(recs[:k]) if x in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg > 0 else np.nan


def hit_rate_at_k(recs, relevant, k: int) -> float:
    rel = set(relevant)
    return float(len(set(recs[:k]) & rel) > 0) if rel else np.nan


def mean_metric(fn, recs_list, rel_list, k: int) -> float:
    return float(np.nanmean([fn(r, g, k) for r, g in zip(recs_list, rel_list)]))
