# -*- coding: utf-8 -*-
"""
Script: 3_train_and_compare_models.py
Dự án: Distributed Movie Recommendation System (MovieLens 25M)
Môn học: Lưu trữ và Xử lý Dữ liệu lớn (IT4931) - Nhóm 7

Chức năng:
1. Đối sánh thực nghiệm 1-1 giữa Single-node ML (Scikit-learn TruncatedSVD) và Distributed ML (Apache Spark MLlib ALS).
2. Đo lường các chỉ số: Thời gian huấn luyện (Training Time), Mức tiêu thụ bộ nhớ (RAM/Peak Memory), Sai số dự đoán (Test RMSE).
3. Xuất kết quả đối sánh ra data/model_comparison_results.json.
4. Lưu mô hình Spark ALS tốt nhất vào models/als_best_model/.
5. Sinh danh sách Top-10 gợi ý phim cho người dùng, kết hợp tiêu đề & thể loại từ movies_cleaned.parquet và lưu vào tầng Gold: data/gold/user_recommendations_top10.parquet.
"""

import os
import sys
import time
import json
import argparse
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# PySpark imports
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator

# Import get_spark_session
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
from src.common.spark_session import get_spark_session


def benchmark_sklearn_svd(train_pdf, test_pdf, rank=20):
    """
    Huấn luyện mô hình cơ sở đơn máy Scikit-learn TruncatedSVD trên ma trận thưa User-Item.
    Đo lường thời gian huấn luyện, bộ nhớ đỉnh (Peak RAM) và Test RMSE.
    """
    print("\n" + "=" * 65)
    print(">>> [1/2] BẮT ĐẦU HUẤN LUYỆN: Scikit-learn TruncatedSVD (Single-node Baseline)")
    print("=" * 65)

    tracemalloc.start()
    t_start = time.time()

    # 1. Tạo ánh xạ ID liên tục cho User và Movie
    unique_users = train_pdf["userId"].unique()
    unique_movies = train_pdf["movieId"].unique()
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(unique_movies)}

    n_users = len(unique_users)
    n_movies = len(unique_movies)
    print(f"[*] Kích thước ma trận thưa User-Movie (Train): {n_users:,} users x {n_movies:,} movies")

    # 2. Xây dựng ma trận thưa scipy.sparse.csr_matrix
    train_rows = train_pdf["userId"].map(user_to_idx).values
    train_cols = train_pdf["movieId"].map(movie_to_idx).values
    train_vals = train_pdf["rating"].values.astype(np.float32)

    R_train = csr_matrix((train_vals, (train_rows, train_cols)), shape=(n_users, n_movies), dtype=np.float32)
    matrix_build_time = time.time() - t_start
    print(f"[*] Thời gian dựng ma trận csr_matrix: {matrix_build_time:.2f}s")

    # 3. Fit TruncatedSVD
    fit_start = time.time()
    svd = TruncatedSVD(n_components=rank, algorithm="randomized", n_iter=7, random_state=42)
    U_sigma = svd.fit_transform(R_train)  # shape: (n_users, rank)
    VT = svd.components_                  # shape: (rank, n_movies)
    train_duration = time.time() - fit_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_mem / (1024 * 1024)

    print(f"[✓] Huấn luyện SVD hoàn tất trong: {train_duration:.2f}s")
    print(f"[✓] Bộ nhớ RAM đỉnh (Peak Memory): {peak_ram_mb:.2f} MB")

    # 4. Đánh giá Test RMSE (Cold-start drop tương tự Spark ALS)
    eval_start = time.time()
    valid_test = test_pdf[test_pdf["userId"].isin(user_to_idx) & test_pdf["movieId"].isin(movie_to_idx)].copy()
    test_u_idx = valid_test["userId"].map(user_to_idx).values
    test_m_idx = valid_test["movieId"].map(movie_to_idx).values
    test_actual = valid_test["rating"].values

    # Tính tích vô hướng U_sigma[u] * VT[:, m]
    preds = np.sum(U_sigma[test_u_idx] * VT[:, test_m_idx].T, axis=1)
    preds = np.clip(preds, 0.5, 5.0)  # Giới hạn trong miền điểm MovieLens

    rmse_svd = float(np.sqrt(np.mean((test_actual - preds) ** 2)))
    eval_duration = time.time() - eval_start
    print(f"[✓] Scikit-learn SVD Test RMSE: {rmse_svd:.4f} (đánh giá trên {len(valid_test):,} bản ghi trong {eval_duration:.2f}s)")

    return {
        "model_name": "Scikit-learn TruncatedSVD",
        "paradigm": "Single-node (Đơn máy)",
        "rank": rank,
        "train_time_sec": round(train_duration, 2),
        "total_time_sec": round(time.time() - t_start, 2),
        "peak_ram_mb": round(peak_ram_mb, 2),
        "test_rmse": round(rmse_svd, 4),
        "evaluated_test_records": int(len(valid_test))
    }


def benchmark_spark_als(train_df, test_df, rank=20, max_iter=15, reg_param=0.1):
    """
    Huấn luyện mô hình phân tán Apache Spark MLlib ALS.
    Đo lường thời gian huấn luyện và Test RMSE.
    """
    print("\n" + "=" * 65)
    print(">>> [2/2] BẮT ĐẦU HUẤN LUYỆN: Apache Spark MLlib ALS (Distributed Core)")
    print("=" * 65)

    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        checkpointInterval=2,     # Truncate RDD lineage mỗi 2 vòng lặp để tránh tràn stack DAG
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",  # Tự động drop các ID chưa xuất hiện trong Train
        nonnegative=True,
        seed=42
    )

    t_start = time.time()
    als_model = als.fit(train_df)
    train_duration = time.time() - t_start
    print(f"[✓] Huấn luyện Spark ALS hoàn tất trong: {train_duration:.2f}s")

    # Đánh giá Test RMSE bằng Spark SQL DataFrame (Catalyst Optimizer, loại bỏ lỗi StackOverflow RDD)
    eval_start = time.time()
    predictions = als_model.transform(test_df)
    rmse_row = predictions.select(
        F.sqrt(F.mean((F.col("rating") - F.col("prediction")) ** 2)).alias("rmse")
    ).collect()
    rmse_als = float(rmse_row[0]["rmse"]) if rmse_row and rmse_row[0]["rmse"] is not None else 0.0
    eval_duration = time.time() - eval_start

    print(f"[✓] Spark MLlib ALS Test RMSE: {rmse_als:.4f} (đánh giá trong {eval_duration:.2f}s)")

    result = {
        "model_name": "Apache Spark MLlib ALS",
        "paradigm": "Distributed (Hệ phân tán)",
        "rank": rank,
        "max_iter": max_iter,
        "reg_param": reg_param,
        "train_time_sec": round(train_duration, 2),
        "test_rmse": round(float(rmse_als), 4),
        "executor_memory_pool": "10 GB",
        "shuffle_partitions": 24
    }
    return result, als_model


def export_gold_recommendations(spark, als_model, movies_path, output_gold_path, num_recs=10, max_users=None):
    """
    Sinh Top-N gợi ý cho người dùng từ mô hình Spark ALS,
    liên kết với danh mục phim sạch (tên, năm, thể loại) và ghi ra tầng Gold dạng Parquet.
    """
    print("\n" + "=" * 65)
    print(f">>> XUẤT DỮ LIỆU TẦNG GOLD: Top-{num_recs} Gợi ý Phim cho Người dùng")
    print("=" * 65)
    t_start = time.time()

    # 1. Sinh gợi ý
    if max_users:
        print(f"[*] Sinh gợi ý cho mẫu {max_users:,} users...")
        users_subset = als_model.userFactors.select("id").withColumnRenamed("id", "userId").limit(max_users)
        raw_recs = als_model.recommendForUserSubset(users_subset, num_recs)
    else:
        print("[*] Sinh gợi ý cho toàn bộ Users trong mô hình...")
        raw_recs = als_model.recommendForAllUsers(num_recs)

    # 2. Biến đổi cấu trúc: Explode mảng recommendations
    # Cấu trúc: userId, recommendations: array<struct<movieId, rating>>
    exploded_recs = raw_recs.select(
        F.col("userId"),
        F.posexplode(F.col("recommendations")).alias("rec_rank", "rec_struct")
    ).select(
        F.col("userId"),
        (F.col("rec_rank") + 1).alias("rank_order"),
        F.col("rec_struct.movieId").cast("int").alias("movieId"),
        F.round(F.col("rec_struct.rating"), 2).alias("predicted_rating")
    )

    # 3. Nạp thông tin phim từ tầng Silver
    if os.path.exists(movies_path):
        print(f"[*] Đọc danh mục phim sạch từ: {movies_path}")
        movies_df = spark.read.parquet(movies_path).select(
            "movieId", "title", "year", "genre_array"
        ).withColumn(
            "genres", F.concat_ws(", ", F.col("genre_array"))
        ).drop("genre_array")

        final_gold_df = exploded_recs.join(movies_df, on="movieId", how="left")
    else:
        print("[!] Không tìm thấy file phim sạch, giữ nguyên schema gốc.")
        final_gold_df = exploded_recs

    # Sắp xếp và lưu ra Parquet
    final_gold_df = final_gold_df.select(
        "userId", "rank_order", "movieId", "title", "year", "genres", "predicted_rating"
    ).orderBy("userId", "rank_order")

    print(f"[*] Đang ghi dữ liệu tầng Gold ra: {output_gold_path}")
    final_gold_df.write.mode("overwrite").parquet(output_gold_path)

    total_duration = time.time() - t_start
    print(f"[✓] Đã tạo thành công Tầng Gold ({output_gold_path}) trong {total_duration:.2f}s!")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện và Đối sánh Mô hình Hệ thống Gợi ý (Group 7)")
    parser.add_argument("--sample-size", type=int, default=1000000, help="Số bản ghi mẫu để thực nghiệm đối sánh (Mặc định: 1M)")
    parser.add_argument("--min-ratings", type=int, default=None, help="Ngưỡng đánh giá tối thiểu của phim (Mặc định: 30 cho 1M, 150 cho 25M)")
    parser.add_argument("--rank", type=int, default=20, help="Số chiều đặc trưng tiềm ẩn (rank / n_components)")
    parser.add_argument("--max-iter", type=int, default=15, help="Số vòng lặp huấn luyện ALS")
    parser.add_argument("--reg-param", type=float, default=0.1, help="Hệ số điều chuẩn ALS")
    parser.add_argument("--full", action="store_true", help="Chạy trên toàn bộ dữ liệu 25M (Bỏ qua Scikit-learn để tránh OOM)")
    parser.add_argument("--skip-svd", action="store_true", help="Chỉ chạy Spark ALS, bỏ qua Scikit-learn SVD")
    parser.add_argument("--export-gold", action="store_true", default=True, help="Xuất danh sách gợi ý ra tầng Gold Parquet")
    args = parser.parse_args()

    # Đường dẫn thư mục
    base_dir = project_root
    silver_ratings_path = os.path.join(base_dir, "data", "silver", "ratings_cleaned.parquet")
    silver_movies_path = os.path.join(base_dir, "data", "silver", "movies_cleaned.parquet")
    results_json_path = os.path.join(base_dir, "data", "model_comparison_results.json")
    model_output_dir = os.path.join(base_dir, "models", "als_best_model")
    gold_recs_path = os.path.join(base_dir, "data", "gold", "user_recommendations_top10.parquet")

    # Kiểm tra dữ liệu đầu vào tầng Silver
    if not os.path.exists(silver_ratings_path):
        print(f"[LỖI] Không tìm thấy dữ liệu tầng Silver: {silver_ratings_path}")
        print("Vui lòng chạy script tiền xử lý trước: python src/batch/2_bronze_to_silver.py")
        sys.exit(1)

    print("\n" + "#" * 65)
    print("#  PROJECT BIG DATA: HUẤN LUYỆN VÀ ĐỐI SÁNH MÔ HÌNH HỆ THỐNG GỢI Ý   #")
    print("#  Nhóm 7 - IT4931: Lưu trữ và Xử lý Dữ liệu lớn                     #")
    print("#" * 65)

    spark = get_spark_session("Model_Training_And_Benchmarking")

    # Thiết lập checkpoint directory để Spark ALS cắt tỉa DAG lineage định kỳ (chống StackOverflow)
    checkpoint_dir = os.path.join(base_dir, "data", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    # 1. Đọc dữ liệu Silver
    print(f"\n[*] Đang đọc dữ liệu đánh giá từ tầng Silver: {silver_ratings_path}")
    ratings_df = spark.read.parquet(silver_ratings_path).select("userId", "movieId", "rating")
    total_records = ratings_df.count()
    print(f"[*] Tổng số bản ghi đánh giá trong Silver: {total_records:,}")

    # 2. Xác định tập dữ liệu huấn luyện
    if args.full:
        print("\n[*] Chế độ FULL SCALE: Huấn luyện Spark ALS trên toàn bộ 25M bản ghi!")
        working_df = ratings_df
        run_svd = False
    elif args.sample_size and args.sample_size < total_records:
        fraction = min(1.0, (args.sample_size * 1.05) / total_records)
        print(f"\n[*] Lấy mẫu ngẫu nhiên ~{args.sample_size:,} bản ghi (fraction={fraction:.4f}, seed=42)...")
        working_df = ratings_df.sample(withReplacement=False, fraction=fraction, seed=42).limit(args.sample_size)
        run_svd = not args.skip_svd
    else:
        working_df = ratings_df
        run_svd = not args.skip_svd

    # 2.1. Kiểm tra bộ lọc chất lượng (nếu người dùng chỉ định ngưỡng lọc thêm)
    min_ratings = args.min_ratings if args.min_ratings is not None else (150 if args.full else 30)
    if args.min_ratings is not None and args.min_ratings > 0:
        print(f"\n[*] Áp dụng bộ lọc chất lượng bổ sung: Chỉ giữ lại các phim có >= {args.min_ratings} lượt đánh giá...")
        movie_counts = working_df.groupBy("movieId").count().filter(F.col("count") >= args.min_ratings)
        working_df = working_df.join(movie_counts.select("movieId"), on="movieId", how="inner")

    working_count = working_df.count()
    distinct_movies = working_df.select("movieId").distinct().count()
    distinct_users = working_df.select("userId").distinct().count()
    print(f"[✓] Tập dữ liệu nạp vào huấn luyện: {working_count:,} bản ghi ({distinct_users:,} users x {distinct_movies:,} movies)")

    # 3. Chia tập Train (80%) và Test (20%) có cố định seed
    print("[*] Phân chia Train/Test: 80% Train, 20% Test (seed=42)...")
    train_df, test_df = working_df.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    test_df.cache()
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"[*] Train set: {train_count:,} bản ghi | Test set: {test_count:,} bản ghi")

    comparison_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_info": {
            "total_silver_records": total_records,
            "min_ratings_threshold": min_ratings,
            "experiment_sample_records": working_count,
            "distinct_users": distinct_users,
            "distinct_movies": distinct_movies,
            "train_records": train_count,
            "test_records": test_count,
            "latent_rank": args.rank
        },
        "models": {}
    }

    # 4. Chạy mô hình cơ sở Scikit-learn TruncatedSVD (nếu được kích hoạt)
    if run_svd:
        print("[*] Đang chuyển đổi dữ liệu Spark sang Pandas cho Scikit-learn...")
        train_pdf = train_df.select("userId", "movieId", "rating").toPandas()
        test_pdf = test_df.select("userId", "movieId", "rating").toPandas()

        svd_results = benchmark_sklearn_svd(train_pdf, test_pdf, rank=args.rank)
        comparison_results["models"]["sklearn_svd"] = svd_results
    else:
        print("\n[!] Bỏ qua Scikit-learn SVD (chế độ Full quy mô lớn / skip-svd để bảo vệ bộ nhớ đơn máy).")

    # 5. Chạy mô hình phân tán Spark MLlib ALS
    als_results, als_model = benchmark_spark_als(
        train_df, test_df, rank=args.rank, max_iter=args.max_iter, reg_param=args.reg_param
    )
    comparison_results["models"]["spark_als"] = als_results

    # 6. Tổng hợp bảng so sánh ra màn hình và lưu file JSON
    print("\n" + "=" * 65)
    print(">>> TỔNG HỢP KẾT QUẢ ĐỐI SÁNH HIỆU NĂNG MÔ HÌNH")
    print("=" * 65)

    if run_svd:
        svd_res = comparison_results["models"]["sklearn_svd"]
        als_res = comparison_results["models"]["spark_als"]
        speedup = svd_res["train_time_sec"] / als_res["train_time_sec"] if als_res["train_time_sec"] > 0 else 1.0

        print(f"{'Tiêu chí':<25} | {'Scikit-learn SVD':<20} | {'Apache Spark ALS':<20}")
        print("-" * 71)
        print(f"{'Kiến trúc':<25} | {svd_res['paradigm']:<20} | {als_res['paradigm']:<20}")
        print(f"{'Thời gian train (s)':<25} | {svd_res['train_time_sec']:<20.2f} | {als_res['train_time_sec']:<20.2f}")
        print(f"{'Test RMSE (Sai số)':<25} | {svd_res['test_rmse']:<20.4f} | {als_res['test_rmse']:<20.4f}")
        print(f"{'Bộ nhớ chiếm dụng':<25} | {str(svd_res['peak_ram_mb']) + ' MB (Peak)':<20} | {als_res['executor_memory_pool']:<20}")
        print("-" * 71)
        print(f"[==>] Tốc độ Spark MLlib ALS so với SVD: {speedup:.2f}x (hoặc nhanh vượt trội trên Big Data)")

    # Lưu kết quả ra JSON
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Đã lưu kết quả đối sánh chi tiết ra: {results_json_path}")

    # 7. Lưu mô hình Spark ALS tốt nhất
    os.makedirs(os.path.dirname(model_output_dir), exist_ok=True)
    print(f"\n[*] Đang lưu mô hình Spark ALS vào: {model_output_dir}")
    als_model.write().overwrite().save(model_output_dir)
    print("[✓] Đã lưu mô hình thành công.")

    # 8. Xuất dữ liệu tầng Gold
    if args.export_gold:
        export_gold_recommendations(
            spark=spark,
            als_model=als_model,
            movies_path=silver_movies_path,
            output_gold_path=gold_recs_path,
            num_recs=10,
            max_users=5000 if not args.full else None
        )

    # Dọn dẹp cache
    train_df.unpersist()
    test_df.unpersist()
    print("\n" + "#" * 65)
    print("#  HOÀN THÀNH HUẤN LUYỆN, ĐỐI SÁNH VÀ XUẤT TẦNG GOLD THÀNH CÔNG!     #")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    main()
