C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]
RAW_NUM = ["TransactionAmt", "dist1", "dist2", "addr1", "addr2", "card1", "card2", "card3", "card5"] + C_COLS + D_COLS
CAT = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"]
ENGINEERED = ["amt_log", "hour", "dow", "n_past", "time_since_last", "amt_mean_past", "amt_std_past",
              "amt_z", "amt_ratio", "cnt_1h", "cnt_24h", "cnt_7d", "sum_24h",
              "device_changed", "addr_changed", "email_changed", "n_devices"]
FEATURES = RAW_NUM + [c + "_freq" for c in CAT] + M_COLS + ENGINEERED
TXN_REFS = [f"txn_features:{f}" for f in FEATURES]
GRAPH_FEATURES = ["gnn_score", "ring_id", "ring_size"]
GRAPH_REFS = [f"account_graph:{f}" for f in GRAPH_FEATURES]
