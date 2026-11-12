"""Unit tests for offline and online evaluation metrics (Part 11.1)."""
import numpy as np
import pytest

from src.evaluation.metrics import hit_rate_at_k, mean_metric, ndcg_at_k, recall_at_k


def test_recall():
    assert recall_at_k([1, 2, 3], [2, 9], 3) == 0.5


def test_ndcg_perfect():
    assert ndcg_at_k([5, 6], [5, 6], 2) == pytest.approx(1.0)


def test_ndcg_order_matters():
    assert ndcg_at_k([9, 5], [5], 2) < ndcg_at_k([5, 9], [5], 2)


def test_hit_rate_and_empty():
    assert hit_rate_at_k([10, 20, 30], [20], 3) == 1.0
    assert hit_rate_at_k([10, 20, 30], [99], 3) == 0.0
    assert np.isnan(recall_at_k([1, 2], [], 2))


def test_mean_metric():
    val = mean_metric(recall_at_k, [[1, 2], [3, 4]], [[1], [9]], 2)
    assert val == pytest.approx(0.5)
