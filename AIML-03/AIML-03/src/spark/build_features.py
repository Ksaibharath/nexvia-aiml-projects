"""Build leak-free user/item features, interaction histories, and state snapshots."""
import os
import pathlib
import re
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def _available_ram_gb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 8.0


def _use_spark() -> bool:
    if os.getenv("FORCE_PANDAS", "0") == "1":
        return False
    if os.getenv("FORCE_SPARK", "0") == "1":
        return True
    if _available_ram_gb() < 4.0:
        return False
    try:
        import pyspark  # noqa: F401
        return True
    except ImportError:
        return False


def run_spark_features(cfg: dict, out_dir: str = "data/processed") -> None:
    import pyspark.sql.functions as F
    from pyspark.sql import SparkSession, Window

    hist_len = int(cfg["history_len"])
    mem_gb = max(2, min(8, int(_available_ram_gb() * 0.6)))
    spark = (
        SparkSession.builder.appName("features")
        .config("spark.driver.memory", f"{mem_gb}g")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )

    ratings = spark.read.parquet("data/interim/ratings")
    movies = spark.read.csv("data/raw/ml-25m/movies.csv", header=True, inferSchema=True)
    train = ratings.filter("split = 'train'")
    pos = ratings.filter("rating >= 4")
    tr_pos = pos.filter("split = 'train'")

    user_map = (
        train.select("userId")
        .distinct()
        .withColumn("user_idx", F.dense_rank().over(Window.orderBy("userId")))
    )
    item_map = (
        train.select("movieId")
        .distinct()
        .withColumn("item_idx", F.dense_rank().over(Window.orderBy("movieId")))
    )

    movies = movies.withColumn("year", F.regexp_extract("title", r"\((\d{4})\)\s*$", 1).cast("int"))
    glist = sorted(
        r[0] for r in movies.select(F.explode(F.split("genres", r"\|")).alias("g")).distinct().collect() if r[0]
    )
    gcols = []
    for g in glist:
        c = "g_" + re.sub(r"\W", "_", g)
        gcols.append(c)
        movies = movies.withColumn(c, F.array_contains(F.split("genres", r"\|"), g).cast("float"))

    max_ts = train.agg(F.max("timestamp")).first()[0]
    item_stats = (
        train.groupBy("movieId")
        .agg(F.count("*").alias("n_ratings"), F.avg("rating").alias("avg_rating"))
        .join(
            tr_pos.groupBy("movieId").agg(
                F.count("*").alias("n_pos"),
                F.sum(F.when(F.col("timestamp") > max_ts - 90 * 86400, 1).otherwise(0)).alias("n_pos_recent"),
            ),
            "movieId",
            "left",
        )
        .fillna(0)
    )
    items = (
        movies.join(item_map, "movieId")
        .join(item_stats, "movieId", "left")
        .fillna(0)
        .select("item_idx", "movieId", "title", "year", "genres", *gcols, "n_ratings", "avg_rating", "n_pos", "n_pos_recent")
    )
    items.write.mode("overwrite").parquet(f"{out_dir}/items")

    u_stats = train.groupBy("userId").agg(F.count("*").alias("u_n"), F.avg("rating").alias("u_avg"))
    u_aff = (
        tr_pos.join(movies.select("movieId", *gcols), "movieId")
        .groupBy("userId")
        .agg(*[F.avg(c).alias("ua_" + c) for c in gcols])
    )
    (
        u_stats.join(u_aff, "userId", "left")
        .fillna(0)
        .join(user_map, "userId")
        .write.mode("overwrite")
        .parquet(f"{out_dir}/user_features")
    )

    w = Window.partitionBy("userId").orderBy("timestamp", "movieId").rowsBetween(-hist_len, -1)
    inter = (
        pos.join(item_map, "movieId", "left")
        .fillna({"item_idx": 0})
        .withColumn("history", F.collect_list("item_idx").over(w))
        .join(user_map, "userId", "left")
        .fillna({"user_idx": 0})
        .select("userId", "movieId", "user_idx", "item_idx", "timestamp", "split", "history")
    )
    inter.write.mode("overwrite").partitionBy("split").parquet(f"{out_dir}/interactions")

    def snapshot(splits, name):
        p = (
            pos.join(item_map, "movieId", "left")
            .fillna({"item_idx": 0})
            .filter(F.col("split").isin(splits) & (F.col("item_idx") > 0))
        )
        w2 = Window.partitionBy("userId").orderBy(F.col("timestamp").desc(), "movieId")
        seen_all = p.groupBy("userId").agg(F.collect_set("item_idx").alias("seen_items"))
        s = (
            p.withColumn("rn", F.row_number().over(w2))
            .filter(F.col("rn") <= hist_len)
            .groupBy("userId")
            .agg(F.sort_array(F.collect_list(F.struct("timestamp", "item_idx"))).alias("h"))
            .withColumn("history", F.col("h.item_idx"))
            .drop("h")
            .join(seen_all, "userId", "left")
            .join(user_map, "userId", "left")
            .fillna({"user_idx": 0})
        )
        s.write.mode("overwrite").parquet(f"{out_dir}/{name}")

    snapshot(["train"], "user_state_train_end")
    snapshot(["train", "val"], "user_state_val_end")

    (
        tr_pos.groupBy("movieId")
        .count()
        .join(item_map, "movieId")
        .orderBy(F.desc("count"))
        .limit(1000)
        .write.mode("overwrite")
        .parquet(f"{out_dir}/popularity")
    )
    spark.stop()
    print("features written to", out_dir)


def run_pandas_features(cfg: dict, out_dir: str = "data/processed") -> None:
    hist_len = int(cfg["history_len"])
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_parquet("data/interim/ratings")
    movies = pd.read_csv("data/raw/ml-25m/movies.csv")

    train = ratings[ratings["split"] == "train"].copy()
    pos = ratings[ratings["rating"] >= 4.0].copy()
    tr_pos = pos[pos["split"] == "train"].copy()

    # 1) ID maps from TRAIN only (dense_rank starting at 1; 0 = unknown/padding)
    train_users = np.sort(train["userId"].unique())
    train_items = np.sort(train["movieId"].unique())
    user_to_idx = pd.Series(np.arange(1, len(train_users) + 1, dtype=np.int32), index=train_users, name="user_idx")
    item_to_idx = pd.Series(np.arange(1, len(train_items) + 1, dtype=np.int32), index=train_items, name="item_idx")

    # 2) Item content: year and multi-hot genres
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    years = movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
    movies["year"] = pd.to_numeric(years, errors="coerce").fillna(1990).astype("int32")

    genre_lists = movies["genres"].str.split("|")
    all_genres = sorted({g for gl in genre_lists for g in gl if g})
    gcols = []
    for g in all_genres:
        c = "g_" + re.sub(r"\W", "_", g)
        gcols.append(c)
        movies[c] = genre_lists.apply(lambda gl, target=g: 1.0 if target in gl else 0.0).astype("float32")

    # 3) Item popularity from TRAIN only
    max_ts = int(train["timestamp"].max())
    recent_cutoff = max_ts - 90 * 86400

    i_base = train.groupby("movieId").agg(
        n_ratings=("rating", "count"),
        avg_rating=("rating", "mean"),
    )
    tr_pos_recent = tr_pos.assign(is_recent=(tr_pos["timestamp"] > recent_cutoff).astype("int32"))
    i_pos = tr_pos_recent.groupby("movieId").agg(
        n_pos=("rating", "count"),
        n_pos_recent=("is_recent", "sum"),
    )

    items = (
        movies.merge(item_to_idx.rename_axis("movieId").reset_index(), on="movieId", how="inner")
        .merge(i_base.reset_index(), on="movieId", how="left")
        .merge(i_pos.reset_index(), on="movieId", how="left")
    )
    items["n_ratings"] = items["n_ratings"].fillna(0).astype("int64")
    items["avg_rating"] = items["avg_rating"].fillna(0.0).astype("float32")
    items["n_pos"] = items["n_pos"].fillna(0).astype("int64")
    items["n_pos_recent"] = items["n_pos_recent"].fillna(0).astype("int64")
    items = items[
        ["item_idx", "movieId", "title", "year", "genres", *gcols, "n_ratings", "avg_rating", "n_pos", "n_pos_recent"]
    ].sort_values("item_idx").reset_index(drop=True)

    items_dir = out_path / "items"
    if items_dir.exists():
        shutil.rmtree(items_dir)
    items_dir.mkdir(parents=True)
    items.to_parquet(items_dir / "part-00000.parquet", index=False)

    # 4) User features from TRAIN only
    u_stats = train.groupby("userId").agg(
        u_n=("rating", "count"),
        u_avg=("rating", "mean"),
    ).reset_index()

    tr_pos_g = tr_pos[["userId", "movieId"]].merge(movies[["movieId", *gcols]], on="movieId", how="left")
    u_aff = tr_pos_g.groupby("userId")[gcols].mean().reset_index()
    u_aff = u_aff.rename(columns={c: f"ua_{c}" for c in gcols})
    uac = [f"ua_{c}" for c in gcols]

    user_features = (
        u_stats.merge(u_aff, on="userId", how="left")
        .merge(user_to_idx.rename_axis("userId").reset_index(), on="userId", how="inner")
    )
    for col in uac:
        user_features[col] = user_features[col].fillna(0.0).astype("float32")
    user_features["u_n"] = user_features["u_n"].astype("int64")
    user_features["u_avg"] = user_features["u_avg"].astype("float32")
    user_features = user_features.sort_values("user_idx").reset_index(drop=True)

    uf_dir = out_path / "user_features"
    if uf_dir.exists():
        shutil.rmtree(uf_dir)
    uf_dir.mkdir(parents=True)
    user_features.to_parquet(uf_dir / "part-00000.parquet", index=False)

    # 5) Interactions with strictly past-only history (-HIST .. -1)
    pos["item_idx"] = pos["movieId"].map(item_to_idx).fillna(0).astype("int32")
    pos["user_idx"] = pos["userId"].map(user_to_idx).fillna(0).astype("int32")
    pos["split"] = pos["split"].astype(str)
    pos = pos.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)

    uids = pos["userId"].values
    iidxs = pos["item_idx"].values
    histories = []
    cur_uid = None
    buf: list[int] = []
    for uid, iidx in zip(uids, iidxs):
        if uid != cur_uid:
            cur_uid = uid
            buf = []
        histories.append(buf[-hist_len:].copy())
        if iidx > 0:
            buf.append(int(iidx))
    pos["history"] = histories

    inter_cols = ["userId", "movieId", "user_idx", "item_idx", "timestamp", "history"]
    inter_dir = out_path / "interactions"
    if inter_dir.exists():
        shutil.rmtree(inter_dir)
    inter_dir.mkdir(parents=True)
    for sp in ("train", "val", "test"):
        sp_dir = inter_dir / f"split={sp}"
        sp_dir.mkdir(parents=True)
        sub = pos[pos["split"] == sp][inter_cols].reset_index(drop=True)
        table = pa.Table.from_pandas(sub, preserve_index=False)
        pq.write_table(table, sp_dir / "part-00000.parquet", compression="snappy")

    # 6) User state snapshots at end of train and end of val
    def write_snapshot(splits: list[str], name: str):
        sub = pos[pos["split"].isin(splits) & (pos["item_idx"] > 0)].sort_values(["userId", "timestamp", "movieId"])
        grouped_hist = sub.groupby("userId")["item_idx"].apply(lambda s: s.values[-hist_len:].astype(np.int32).tolist())
        grouped_seen = sub.groupby("userId")["item_idx"].apply(lambda s: np.unique(s.values).astype(np.int32).tolist())
        snap = pd.DataFrame({
            "userId": grouped_hist.index.astype(np.int32),
            "history": grouped_hist.values,
            "seen_items": grouped_seen.values,
        })
        snap["user_idx"] = snap["userId"].map(user_to_idx).fillna(0).astype("int32")
        s_dir = out_path / name
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        snap.to_parquet(s_dir / "part-00000.parquet", index=False)

    write_snapshot(["train"], "user_state_train_end")
    write_snapshot(["train", "val"], "user_state_val_end")

    # 7) Popularity table (top 1000 from TRAIN positives only)
    pop = (
        tr_pos.groupby("movieId")
        .size()
        .reset_index(name="count")
        .merge(item_to_idx.rename_axis("movieId").reset_index(), on="movieId", how="inner")
        .sort_values("count", ascending=False)
        .head(1000)
        .reset_index(drop=True)
    )
    pop_dir = out_path / "popularity"
    if pop_dir.exists():
        shutil.rmtree(pop_dir)
    pop_dir.mkdir(parents=True)
    pop.to_parquet(pop_dir / "part-00000.parquet", index=False)

    print("features written to", out_dir)


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if _use_spark():
        run_spark_features(cfg)
    else:
        run_pandas_features(cfg)


if __name__ == "__main__":
    main()
