# Features (single source: fraud/features/feature_list.py)

## Raw
TransactionAmt, dist1/2, addr1/2, card1/2/3/5, C1-C14, D1-D15 — Vesta counts/deltas.

## Categorical (train-only frequency encoding)
ProductCD, card4, card6, P/R_emaildomain, DeviceType → *_freq. Rare = risky.

## Match flags
M1-M9 → 0/1.

## Time
amt_log=log1p(amount), hour (relative), dow.

## Velocity (past only, closed='left')
cnt_1h, cnt_24h, cnt_7d, sum_24h, time_since_last, n_past. Bursts = stolen cards.

## Amount vs account (past only)
amt_mean_past, amt_std_past, amt_z=(amt-mean)/(std+1), amt_ratio=amt/mean.

## Change
device_changed, addr_changed, email_changed, n_devices. New device/location before large purchase.

## Graph (Feast account_graph)
gnn_score, ring_id, ring_size from GraphSAGE snapshots.
