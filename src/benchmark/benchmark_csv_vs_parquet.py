# -*- coding: utf-8 -*-
"""
Script đo đạc và so sánh hiệu năng giữa CSV (Hàng) và Parquet (Cột, nén Snappy)
- Dung lượng lưu trữ (Storage Footprint)
- Thời gian quét toàn bộ (Full Table Scan & Count)
- Thời gian tính toán tổng hợp (Column Pruning: avg rating per movie)
- Thời gian truy vấn có điều kiện lọc (Predicate Pushdown: rating_year = 2018)
"""

import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyspark.sql.functions import avg, col, count

from src.common.schemas import RATINGS_SCHEMA
from src.common.spark_session import get_spark_session


def get_dir_size(path):
    """Tính tổng dung lượng (MB) của 1 file hoặc thư mục"""
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)


def run_benchmark():
    spark = get_spark_session(app_name="Benchmark_CSV_vs_Parquet")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    csv_path = os.path.join(base_dir, "data", "bronze", "ratings.csv")
    parquet_path = os.path.join(base_dir, "data", "silver", "ratings_cleaned.parquet")

    if not os.path.exists(parquet_path):
        print(f"[-] Không tìm thấy dữ liệu Parquet tại: {parquet_path}")
        print("[!] Vui lòng chạy 2_bronze_to_silver.py trước!")
        spark.stop()
        return

    results = {}

    print("\n" + "=" * 65)
    print("      THỰC NGHIỆM SO SÁNH HIỆU NĂNG: CSV vs PARQUET             ")
    print("      (Cấu hình: Intel Core Ultra 7 | 32 GB RAM)                ")
    print("=" * 65)

    # 1. So sánh dung lượng lưu trữ
    print("\n[1] Đang đo đạc dung lượng lưu trữ...")
    csv_size_mb = get_dir_size(csv_path)
    parquet_size_mb = get_dir_size(parquet_path)
    saved_pct = ((csv_size_mb - parquet_size_mb) / csv_size_mb * 100) if csv_size_mb > 0 else 0

    results["storage"] = {
        "csv_size_mb": round(csv_size_mb, 2),
        "parquet_size_mb": round(parquet_size_mb, 2),
        "saved_percentage": round(saved_pct, 2),
    }
    print(f"  + Dung lượng CSV gốc:       {csv_size_mb:.2f} MB")
    print(f"  + Dung lượng Parquet nén:   {parquet_size_mb:.2f} MB")
    print(f"  + Tiết kiệm dung lượng:     {saved_pct:.2f}%")

    # 2. Benchmark Query 1: Full Scan & Count
    print("\n[2] Benchmark Query 1: Đọc toàn bộ bảng & Đếm số dòng (Full Scan)...")
    t0 = time.time()
    df_csv = spark.read.schema(RATINGS_SCHEMA).option("header", "true").csv(csv_path)
    cnt_csv = df_csv.count()
    time_csv_count = time.time() - t0

    t0 = time.time()
    df_parquet = spark.read.parquet(parquet_path)
    cnt_parquet = df_parquet.count()
    time_parquet_count = time.time() - t0

    results["query_count"] = {
        "csv_time_sec": round(time_csv_count, 3),
        "parquet_time_sec": round(time_parquet_count, 3),
        "speedup": round(time_csv_count / time_parquet_count, 2) if time_parquet_count > 0 else 0,
        "csv_rows": cnt_csv,
        "parquet_rows": cnt_parquet,
    }
    print(f"  + CSV Scan & Count:         {time_csv_count:.3f} s ({cnt_csv:,} dòng)")
    print(f"  + Parquet Scan & Count:     {time_parquet_count:.3f} s ({cnt_parquet:,} dòng)")
    print(f"  => Parquet nhanh gấp:       {results['query_count']['speedup']}x")

    # 3. Benchmark Query 2: Column Pruning (Chỉ quét cột rating & movieId để tính trung bình)
    print("\n[3] Benchmark Query 2: Column Pruning (Tính điểm trung bình theo Movie)...")
    t0 = time.time()
    df_csv.groupBy("movieId").agg(avg("rating").alias("avg_rating"), count("rating").alias("cnt")).write.format("noop").mode("overwrite").save()
    time_csv_agg = time.time() - t0

    t0 = time.time()
    df_parquet.groupBy("movieId").agg(avg("rating").alias("avg_rating"), count("rating").alias("cnt")).write.format("noop").mode("overwrite").save()
    time_parquet_agg = time.time() - t0

    results["query_aggregation"] = {
        "csv_time_sec": round(time_csv_agg, 3),
        "parquet_time_sec": round(time_parquet_agg, 3),
        "speedup": round(time_csv_agg / time_parquet_agg, 2) if time_parquet_agg > 0 else 0,
    }
    print(f"  + CSV GroupBy Aggregation:     {time_csv_agg:.3f} s")
    print(f"  + Parquet GroupBy Aggregation: {time_parquet_agg:.3f} s")
    print(f"  => Parquet nhanh gấp:          {results['query_aggregation']['speedup']}x")

    # 4. Benchmark Query 3: Predicate Pushdown & Partition Pruning
    print("\n[4] Benchmark Query 3: Predicate Pushdown (Lọc rating >= 4.5)...")
    t0 = time.time()
    df_csv.filter(col("rating") >= 4.5).count()
    time_csv_filter = time.time() - t0

    t0 = time.time()
    df_parquet.filter(col("rating") >= 4.5).count()
    time_parquet_filter = time.time() - t0

    results["query_filter"] = {
        "csv_time_sec": round(time_csv_filter, 3),
        "parquet_time_sec": round(time_parquet_filter, 3),
        "speedup": round(time_csv_filter / time_parquet_filter, 2) if time_parquet_filter > 0 else 0,
    }
    print(f"  + CSV Filter rating >= 4.5:     {time_csv_filter:.3f} s")
    print(f"  + Parquet Filter rating >= 4.5: {time_parquet_filter:.3f} s")
    print(f"  => Parquet nhanh gấp:           {results['query_filter']['speedup']}x")

    # Lưu kết quả ra file JSON
    out_json = os.path.join(base_dir, "data", "benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Đã lưu toàn bộ số liệu đo đạc vào: {out_json}")

    spark.stop()


if __name__ == "__main__":
    run_benchmark()

