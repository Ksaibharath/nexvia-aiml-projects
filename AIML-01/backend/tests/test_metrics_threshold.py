import numpy as np
from fraud.models.threshold import best_threshold
def test_threshold_respects_alert_rate():
    rng = np.random.default_rng(0); y = (rng.random(20000) < 0.03).astype(int)
    p = np.clip(y * 0.4 + rng.random(20000) * 0.6, 0, 1); amt = rng.uniform(10, 500, 20000)
    thr, s = best_threshold(y, p, amt, review_cost=5.0, max_alert_rate=0.02)
    assert s["alert_rate"] <= 0.02 and s["precision"] > y.mean()
