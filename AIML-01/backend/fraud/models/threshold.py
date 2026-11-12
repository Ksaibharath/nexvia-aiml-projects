import numpy as np
def best_threshold(y, p, amount, review_cost=5.0, max_alert_rate=0.02):
    best = None
    for t in np.quantile(p, np.linspace(0.90, 0.9995, 300)):
        flag = p >= t
        if flag.mean() > max_alert_rate: continue
        caught = float(amount[flag & (y == 1)].sum())
        net = caught - review_cost * int(flag.sum())
        if best is None or net > best[0]:
            tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
            best = (net, float(t), {"alert_rate": float(flag.mean()), "precision": tp / max(tp + fp, 1),
                    "recall": tp / max(int(y.sum()), 1), "caught": caught, "net_savings": net, "alerts": int(flag.sum())})
    _, thr, stats = best
    return thr, stats
