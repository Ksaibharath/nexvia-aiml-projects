# Data Dictionary

Every Parquet table below is generated reproducibly by `src/data/clean_split.py` and `src/spark/build_features.py`. **Index `0` is strictly reserved for `unknown / sequence padding`** across all user and item integer IDs; all learned features and ID maps are computed **exclusively on the `train` split** to prevent temporal leakage.

---

## 1. `data/interim/ratings` (Partitioned by `split={train,val,test}`)
Cleaned, deduplicated, 2-pass 5-core filtered MovieLens ratings split chronologically (`80% / 10% / 10%`).

| Column | Type | Computed On | Meaning |
| :--- | :--- | :--- | :--- |
| `userId` | `int32` | Raw MovieLens | Original MovieLens user identifier |
| `movieId` | `int32` | Raw MovieLens | Original MovieLens movie identifier |
| `rating` | `float32` | Raw MovieLens | Explicit star rating (`0.5` to `5.0` in `0.5` increments) |
| `timestamp` | `int64` | Raw MovieLens | Unix epoch timestamp (UTC seconds) |
| `split` | `string` | Quantiles `0.80, 0.90` | Partition key: `train` (oldest 80%), `val` (next 10%), `test` (newest 10%) |

---

## 2. `data/processed/items`
One row per movie present in the `train` split (`item_idx >= 1`).

| Column | Type | Computed On | Meaning |
| :--- | :--- | :--- | :--- |
| `item_idx` | `int32` | `train` only | Dense integer ID (`1 .. N_items`); `FAISS position = item_idx - 1` |
| `movieId` | `int32` | `movies.csv` | Original MovieLens movie ID |
| `title` | `string` | `movies.csv` | Movie title including release year |
| `year` | `int32` | `movies.csv` | Release year parsed via regex `\((\d{4})\)\s*$` (default `1990`) |
| `genres` | `string` | `movies.csv` | Pipe-separated genre string (e.g., `Action|Adventure|Sci-Fi`) |
| `g_*` (20 cols) | `float32` | `movies.csv` | Multi-hot genre indicator flags (`0.0` or `1.0` per genre) |
| `n_ratings` | `int64` | `train` only | Total number of ratings received during the `train` period |
| `avg_rating` | `float32` | `train` only | Mean star rating during the `train` period |
| `n_pos` | `int64` | `train` only | Number of positive ratings (`rating >= 4.0`) in `train` |
| `n_pos_recent` | `int64` | `train` last 90d | Positive ratings in the final 90 days of the `train` window |

---

## 3. `data/processed/user_features`
One row per user present in the `train` split (`user_idx >= 1`).

| Column | Type | Computed On | Meaning |
| :--- | :--- | :--- | :--- |
| `userId` | `int32` | `train` only | Original MovieLens user ID |
| `user_idx` | `int32` | `train` only | Dense integer ID (`1 .. N_users`); `0` = unseen cold-start user |
| `u_n` | `int64` | `train` only | Total ratings submitted by the user in `train` |
| `u_avg` | `float32` | `train` only | Mean star rating given by the user in `train` |
| `ua_g_*` (20 cols) | `float32` | `train` positives | User's genre affinity = fraction of liked `train` movies in each genre |

---

## 4. `data/processed/interactions` (Partitioned by `split={train,val,test}`)
One row per positive interaction (`rating >= 4.0`) with strictly past-only sliding history.

| Column | Type | Computed On | Meaning |
| :--- | :--- | :--- | :--- |
| `userId` | `int32` | All splits | Original MovieLens user ID |
| `movieId` | `int32` | All splits | Original MovieLens movie ID |
| `user_idx` | `int32` | `train` map | Mapped user ID (`0` if user first appeared after `train`) |
| `item_idx` | `int32` | `train` map | Mapped item ID (`0` if movie first appeared after `train`) |
| `timestamp` | `int64` | All splits | Unix epoch seconds of the positive rating |
| `history` | `list<int32>` | Past window `-20..-1` | Up to 20 most recent `item_idx` liked **strictly before** this row |
| `split` | `string` | Partition key | `train`, `val`, or `test` |

---

## 5. `data/processed/user_state_train_end` & `data/processed/user_state_val_end`
Point-in-time user history snapshots used for validation (`train_end`) and test evaluation / live online serving (`val_end`).

| Column | Type | Computed On | Meaning |
| :--- | :--- | :--- | :--- |
| `userId` | `int32` | `train` / `train+val` | Original MovieLens user ID |
| `user_idx` | `int32` | `train` map | Dense user ID (`0` if user joined during `val`) |
| `history` | `list<int32>` | Up to snapshot cutoff | Chronological list of up to 20 most recent liked `item_idx` values |
| `seen_items` | `list<int32>` | Up to snapshot cutoff | Complete set of known liked `item_idx` values to exclude from recommendations |

---

## 6. `data/processed/popularity`
Top-1,000 popular movies computed from `train` positives for cold-start baselines.

| Column | Type | Computed On | Meaning |
| :--- | :--- | :--- | :--- |
| `movieId` | `int32` | `train` positives | Original MovieLens movie ID |
| `count` | `int64` | `train` positives | Positive rating count in `train` |
| `item_idx` | `int32` | `train` map | Mapped dense integer ID |
