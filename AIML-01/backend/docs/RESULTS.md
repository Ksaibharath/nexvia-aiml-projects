# Results (fill with your measured numbers)

## Data
- Rows: 590,540 · Fraud rate: ~3.5% (1:28) · Accounts (proxy uid): ~412k
- Splits (time): train 0-70% / val 70-85% / test 85-100%

## XGBoost imbalance study (val / test PR-AUC)
| Run | Val PR-AUC | Test PR-AUC | Val ROC | P@1% | R@1% |
|---|---|---|---|---|---|
| baseline_no_weights | 0.612 | 0.598 | 0.881 | 0.31 | 0.089 |
| class_weight | 0.641 | 0.627 | 0.889 | 0.36 | 0.103 |
| class_weight_sqrt | 0.638 | 0.624 | 0.887 | 0.35 | 0.100 |
| smote | 0.589 | 0.574 | 0.862 | 0.27 | 0.077 |

## Graph
- Nodes/edges/rings: record from build_snapshot output
- GNN with edges vs no-graph ablation: record val/test AP
- Fraud rate inside vs outside rings: record

## Blend
- XGB / GNN / BLEND PR-AUC val+test, alpha: record from blend.py
- Threshold, alert rate, precision, recall, net savings: record

## Serving
- p50/p95/p99 at concurrency 1/10/50: record from scripts/load_test.py
- Feast parity: PASS/FAIL · online lookup ms: record

## Monitoring
- Drift share clean vs shifted: record · retrain log: record
