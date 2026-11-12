"""Clean MovieLens 25M ratings, apply 2-pass 5-core filter, and split 80/10/10 by time."""
import datetime
import json
import os
import pathlib
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


def run_spark_clean_split(cfg: dict, raw_ml_dir: pathlib.Path, out_dir: pathlib.Path) -> dict:
    from pyspark.sql import SparkSession, functions as F

    mem_gb = max(2, min(8, int(_available_ram_gb() * 0.6)))
    spark = (
        SparkSession.builder.appName("clean_split")
        .config("spark.driver.memory", f"{mem_gb}g")
        .getOrCreate()
    )
    r = spark.read.csv(str(raw_ml_dir / "ratings.csv"), header=True, inferSchema=True)
    r = r.dropna().dropDuplicates(["userId", "movieId"])

    if cfg.get("sample_users"):
        users = (
            r.select("userId")
            .distinct()
            .sample(False, 0.5, seed=1)
            .limit(int(cfg["sample_users"]))
        )
        r = r.join(users, "userId")

    # 2-pass 5-core filter
    for _ in range(2):
        keep_u = r.groupBy("userId").count().filter("count >= 5").select("userId")
        keep_i = r.groupBy("movieId").count().filter("count >= 5").select("movieId")
        r = r.join(keep_u, "userId").join(keep_i, "movieId")

    t_train, t_val = r.approxQuantile("timestamp", [0.8, 0.9], 0.001)
    r = r.withColumn(
        "split",
        F.when(F.col("timestamp") <= t_train, "train")
        .when(F.col("timestamp") <= t_val, "val")
        .otherwise("test"),
    )
    r.write.mode("overwrite").partitionBy("split").parquet(str(out_dir))

    summary_df = (
        spark.read.parquet(str(out_dir))
        .groupBy("split")
        .agg(
            F.count("*").alias("rows"),
            F.countDistinct("userId").alias("users"),
            F.countDistinct("movieId").alias("items"),
            F.min("timestamp").alias("min_ts"),
            F.max("timestamp").alias("max_ts"),
            F.from_unixtime(F.min("timestamp")).alias("from"),
            F.from_unixtime(F.max("timestamp")).alias("to"),
        )
    )
    summary_df.show(truncate=False)
    rows = [row.asDict() for row in summary_df.collect()]
    spark.stop()
    return {"engine": "pyspark", "t_train": t_train, "t_val": t_val, "splits": rows}


def run_chunked_clean_split(cfg: dict, raw_ml_dir: pathlib.Path, out_dir: pathlib.Path) -> dict:
    ratings_csv = raw_ml_dir / "ratings.csv"
    sample_n = cfg.get("sample_users")
    rng = np.random.RandomState(1)

    if sample_n:
        # Pass 1: collect distinct userIds cleanly with low memory
        distinct_users = set()
        for chunk in pd.read_csv(ratings_csv, usecols=["userId"], dtype={"userId": "int32"}, chunksize=2_000_000):
            distinct_users.update(chunk["userId"].dropna().unique().tolist())
        sorted_users = np.array(sorted(distinct_users), dtype=np.int32)
        chosen_users = set(rng.choice(sorted_users, size=min(int(sample_n), len(sorted_users)), replace=False).tolist())

        # Pass 2: load rows only for sampled users
        parts = []
        for chunk in pd.read_csv(
            ratings_csv,
            dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
            chunksize=2_000_000,
        ):
            sub = chunk[chunk["userId"].isin(chosen_users)]
            if len(sub):
                parts.append(sub)
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(
            ratings_csv,
            dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
        )

    df = df.dropna().drop_duplicates(subset=["userId", "movieId"], keep="last").reset_index(drop=True)

    # 2-pass 5-core filter (identical logic to Spark pipeline)
    for _ in range(2):
        u_counts = df["userId"].value_counts()
        keep_u = u_counts[u_counts >= 5].index
        df = df[df["userId"].isin(keep_u)]
        i_counts = df["movieId"].value_counts()
        keep_i = i_counts[i_counts >= 5].index
        df = df[df["movieId"].isin(keep_i)]

    df = df.reset_index(drop=True)

    # Time split: oldest 80% = train, next 10% = val, newest 10% = test
    t_train = int(np.quantile(df["timestamp"].values, 0.80))
    t_val = int(np.quantile(df["timestamp"].values, 0.90))

    split_arr = np.where(
        df["timestamp"].values <= t_train,
        "train",
        np.where(df["timestamp"].values <= t_val, "val", "test"),
    )
    df["split"] = split_arr

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_meta = []
    print(f"{'split':<8} {'rows':>10} {'users':>8} {'items':>8}  {'from':<20} {'to':<20}")
    print("-" * 80)
    for sp in ("train", "val", "test"):
        sub = df[df["split"] == sp].drop(columns=["split"]).reset_index(drop=True)
        part_dir = out_dir / f"split={sp}"
        part_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(sub, preserve_index=False)
        pq.write_table(table, part_dir / "part-00000.parquet", compression="snappy")

        mn_ts = int(sub["timestamp"].min())
        mx_ts = int(sub["timestamp"].max())
        dt_from = datetime.datetime.fromtimestamp(mn_ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        dt_to = datetime.datetime.fromtimestamp(mx_ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        row_info = {
            "split": sp,
            "rows": int(len(sub)),
            "users": int(sub["userId"].nunique()),
            "items": int(sub["movieId"].nunique()),
            "min_ts": mn_ts,
            "max_ts": mx_ts,
            "from": dt_from,
            "to": dt_to,
        }
        splits_meta.append(row_info)
        print(f"{sp:<8} {row_info['rows']:>10,d} {row_info['users']:>8,d} {row_info['items']:>8,d}  {dt_from:<20} {dt_to:<20}")

    return {"engine": "pyarrow_chunked", "t_train": t_train, "t_val": t_val, "splits": splits_meta}


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_base = pathlib.Path(os.getenv("ML25M_RAW_DIR", "data/raw"))
    raw_ml_dir = raw_base / "ml-25m"
    if not (raw_ml_dir / "ratings.csv").exists():
        # Check fallback data/raw/ml-25m
        raw_ml_dir = pathlib.Path("data/raw/ml-25m")

    # Keep movies.csv accessible in data/raw/ml-25m/movies.csv for downstream scripts
    local_ml_dir = pathlib.Path("data/raw/ml-25m")
    local_ml_dir.mkdir(parents=True, exist_ok=True)
    if (raw_ml_dir / "movies.csv").exists() and not (local_ml_dir / "movies.csv").exists():
        shutil.copy2(raw_ml_dir / "movies.csv", local_ml_dir / "movies.csv")

    out_dir = pathlib.Path("data/interim/ratings")
    if _use_spark():
        summary = run_spark_clean_split(cfg, raw_ml_dir, out_dir)
    else:
        summary = run_chunked_clean_split(cfg, raw_ml_dir, out_dir)

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
