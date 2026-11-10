# -*- coding: utf-8 -*-
"""
Pipeline chuyển đổi Bronze (Raw CSV) -> Silver (Cleaned Parquet)
Hỗ trợ:
- Chế độ --sample: Lấy mẫu 1 triệu dòng để test nhanh trong vài giây
- Chế độ --full: Xử lý toàn bộ 25 triệu dòng chính thức
"""

import argparse
import os
import sys
import time

# Thêm thư mục gốc vào path để import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyspark.sql import Window
from pyspark.sql.functions import (
    col,
    count,
    from_unixtime,
    lower,
    length,
    regexp_extract,
    regexp_replace,
    row_number,
    split,
    to_timestamp,
    trim,
    when,
    year,
)
from pyspark.sql.types import IntegerType

from src.common.schemas import MOVIES_SCHEMA, RATINGS_SCHEMA, TAGS_SCHEMA
from src.common.spark_session import get_spark_session


def clean_and_transform_movies(spark, bronze_dir, silver_dir):
    print("\n" + "=" * 65)
    print("[*] BƯỚC 1: LÀM SẠCH VÀ BÓC TÁCH ĐẶC TRƯNG MOVIES")
    print("=" * 65)

    movies_path = os.path.join(bronze_dir, "movies.csv")
    if not os.path.exists(movies_path):
        print(f"[-] Không tìm thấy {movies_path}. Vui lòng chạy 1_download_data.py trước!")
        return None

    # 1. Đọc với Schema định sẵn
    df_raw = spark.read.schema(MOVIES_SCHEMA).option("header", "true").csv(movies_path)
    total_raw = df_raw.count()
    print(f"[+] Số lượng phim gốc: {total_raw:,} bản ghi")

    # 2. Làm sạch: Loại bỏ movieId null hoặc trùng lặp
    df_clean = df_raw.filter(col("movieId").isNotNull()).dropDuplicates(["movieId"])

    # 3. Bóc tách năm phát hành từ tiêu đề bằng Regex: ví dụ "Toy Story (1995)" -> year = 1995
    extracted_year = regexp_extract(col("title"), r"\((\d{4})\)$", 1)
    df_transformed = (
        df_clean.withColumn(
            "year",
            when(length(extracted_year) == 4, extracted_year.cast(IntegerType())).otherwise(None),
        )
        .withColumn("clean_title", trim(regexp_replace(col("title"), r"\s*\(\d{4}\)$", "")))
        .withColumn("genre_array", split(col("genres"), r"\|"))
    )

    # 4. Ghi ra Silver Parquet
    out_path = os.path.join(silver_dir, "movies_cleaned.parquet")
    print(f"[*] Đang ghi movies đã làm sạch vào: {out_path}")
    df_transformed.write.mode("overwrite").parquet(out_path)
    print(f"[+] Hoàn tất xử lý movies! (Ghi nhận: {df_transformed.count():,} bản ghi)")

    return df_transformed


def clean_and_transform_ratings(spark, bronze_dir, silver_dir, movies_df=None, sample_mode=False, min_ratings=None):
    print("\n" + "=" * 65)
    mode_text = "SAMPLE (1,000,000 DÒNG TEST NHANH)" if sample_mode else "FULL (25,000,000 DÒNG CHÍNH THỨC)"
    print(f"[*] BƯỚC 2: LÀM SẠCH VÀ PHÂN VÙNG RATINGS - CHẾ ĐỘ {mode_text}")
    print("=" * 65)

    ratings_path = os.path.join(bronze_dir, "ratings.csv")
    if not os.path.exists(ratings_path):
        print(f"[-] Không tìm thấy {ratings_path}.")
        return None

    start_time = time.time()

    # 1. Đọc ratings với Schema tường minh
    df_raw = spark.read.schema(RATINGS_SCHEMA).option("header", "true").csv(ratings_path)
    
    if sample_mode:
        print("[*] Đang trích xuất mẫu 1,000,000 dòng để xử lý nhanh...")
        df_raw = df_raw.limit(1000000)

    total_raw = df_raw.count()
    print(f"[+] Số lượng ratings nạp vào: {total_raw:,} bản ghi")

    # 2. Kiểm tra tính hợp lệ & Loại bỏ giá trị thiếu (Drop Null)
    df_valid = df_raw.filter(
        (col("userId").isNotNull())
        & (col("movieId").isNotNull())
        & (col("rating").isNotNull())
        & (col("userId") > 0)
        & (col("movieId") > 0)
        & (col("rating") >= 0.5)
        & (col("rating") <= 5.0)
    )

    # 3. Chuyển đổi timestamp sang TimestampType và trích xuất năm đánh giá (rating_year)
    df_typed = df_valid.withColumn(
        "rating_time", to_timestamp(from_unixtime(col("timestamp")))
    ).withColumn("rating_year", year(col("rating_time")))

    # 4. Khử trùng lặp: Nếu 1 user đánh giá cùng 1 movie nhiều lần -> Lấy bản ghi có timestamp mới nhất
    window_spec = Window.partitionBy("userId", "movieId").orderBy(col("timestamp").desc())
    df_dedup = df_typed.withColumn("row_num", row_number().over(window_spec)).filter(col("row_num") == 1).drop("row_num")

    # 5. Đảm bảo toàn vẹn tham chiếu (Referential Integrity): movieId phải tồn tại trong movies_cleaned
    if movies_df is not None:
        df_dedup = df_dedup.join(movies_df.select("movieId"), on="movieId", how="left_semi")

    # 6. Lọc thưa & Loại bỏ phim rác (Popularity & Sparsity Filtering)
    # Ngưỡng chất lượng: >= 30 lượt đánh giá (chế độ mẫu 1M), >= 150 lượt đánh giá (chế độ toàn phần 25M)
    threshold = min_ratings if min_ratings is not None else (30 if sample_mode else 150)
    print(f"[*] Áp dụng bộ lọc chất lượng: Chỉ giữ các phim có >= {threshold} lượt đánh giá...")
    movie_counts = df_dedup.groupBy("movieId").agg(count("rating").alias("review_count"))
    active_movies = movie_counts.filter(col("review_count") >= threshold)
    df_filtered = df_dedup.join(active_movies.select("movieId"), on="movieId", how="left_semi")

    total_clean = df_filtered.count()
    distinct_movies = df_filtered.select("movieId").distinct().count()
    distinct_users = df_filtered.select("userId").distinct().count()
    print(f"[+] Số lượng ratings sau làm sạch & lọc chất lượng: {total_clean:,} bản ghi")
    print(f"[+] Giữ lại: {distinct_users:,} users và {distinct_movies:,} bộ phim chất lượng cao.")
    print(f"[+] Đã loại bỏ: {total_raw - total_clean:,} bản ghi rác/trùng/nhiễu thưa.")

    # 7. Ghi ra Silver Parquet (Phân vùng theo rating_year và nén Snappy)
    out_path = os.path.join(silver_dir, "ratings_cleaned.parquet")
    print(f"[*] Đang ghi ratings ra định dạng Parquet nén Snappy (partitionBy: rating_year)...")
    (
        df_filtered.write.mode("overwrite")
        .partitionBy("rating_year")
        .parquet(out_path)
    )

    elapsed = time.time() - start_time
    print(f"[+] Hoàn tất xử lý ratings trong: {elapsed:.2f} giây! Lưu tại: {out_path}")
    return df_filtered


def clean_and_transform_tags(spark, bronze_dir, silver_dir):
    print("\n" + "=" * 65)
    print("[*] BƯỚC 3: LÀM SẠCH VÀ CHUẨN HÓA DỮ LIỆU THẺ (TAGS)")
    print("=" * 65)

    tags_path = os.path.join(bronze_dir, "tags.csv")
    if not os.path.exists(tags_path):
        print(f"[!] Không tìm thấy {tags_path}. Bỏ qua bước làm sạch tags.")
        return None

    # Đọc với Schema định sẵn
    df_raw = spark.read.schema(TAGS_SCHEMA).option("header", "true").csv(tags_path)
    total_raw = df_raw.count()
    print(f"[+] Số lượng tags gốc: {total_raw:,} bản ghi")

    # Làm sạch: loại null, lowercase, trim khoảng trắng
    df_clean = df_raw.filter(
        (col("userId").isNotNull())
        & (col("movieId").isNotNull())
        & (col("tag").isNotNull())
        & (trim(col("tag")) != "")
    ).withColumn("tag_cleaned", lower(trim(col("tag"))))

    out_path = os.path.join(silver_dir, "tags_cleaned.parquet")
    print(f"[*] Đang ghi tags ra Parquet: {out_path}")
    df_clean.write.mode("overwrite").parquet(out_path)
    print(f"[+] Hoàn tất xử lý tags! ({df_clean.count():,} bản ghi)")
    return df_clean


def main():
    parser = argparse.ArgumentParser(description="Pipeline Bronze -> Silver (ETL)")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Chạy ở chế độ mẫu (1M dòng ratings) để test nhanh",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=None,
        help="Ngưỡng đánh giá tối thiểu (Mặc định: 30 cho --sample, 150 cho full 25M)",
    )
    args = parser.parse_args()

    spark = get_spark_session(app_name="Bronze_to_Silver_ETL")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    bronze_dir = os.path.join(base_dir, "data", "bronze")
    silver_dir = os.path.join(base_dir, "data", "silver")

    os.makedirs(silver_dir, exist_ok=True)

    mode_label = "SAMPLE (1M)" if args.sample else "FULL (25M)"
    print("================================================================")
    print(f"    KHỞI CHẠY PIPELINE BRONZE -> SILVER [{mode_label}]          ")
    print("================================================================")

    movies_df = clean_and_transform_movies(spark, bronze_dir, silver_dir)
    clean_and_transform_ratings(spark, bronze_dir, silver_dir, movies_df, sample_mode=args.sample, min_ratings=args.min_ratings)
    clean_and_transform_tags(spark, bronze_dir, silver_dir)

    print("\n[SUCCESS] Pipeline Bronze -> Silver đã hoàn tất xuất sắc!")
    spark.stop()


if __name__ == "__main__":
    main()
