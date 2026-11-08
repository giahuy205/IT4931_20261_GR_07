# -*- coding: utf-8 -*-

import os
from pyspark.sql import SparkSession


def get_spark_session(app_name="MovieLens_Group7_BigData"):
    """
    Tạo hoặc lấy SparkSession với cấu hình tối ưu 10 GB RAM (để dành 22 GB cho hệ điều hành & ứng dụng khác):
    - spark.driver.memory: 10g
    - spark.executor.memory: 10g
    - spark.driver.maxResultSize: 2g
    - spark.sql.shuffle.partitions: 24
    - spark.sql.adaptive.enabled: True (AQE tự động tối ưu hóa query plan khi thực thi)
    - spark.serializer: Java Kryo Serializer tăng tốc shuffle mạng/bộ nhớ
    - spark.sql.execution.arrow.pyspark.enabled: True (tăng tốc chuyển đổi Spark <-> Pandas)
    """
    # Thiết lập JAVA_HOME nếu chưa có
    if "JAVA_HOME" not in os.environ:
        default_java = r"C:\Program Files\Java\jdk-21.0.12.1"
        if os.path.exists(default_java):
            os.environ["JAVA_HOME"] = default_java
            os.environ["PATH"] = os.path.join(default_java, "bin") + os.pathsep + os.environ.get("PATH", "")

    # Thiết lập HADOOP_HOME cho Windows (winutils.exe & hadoop.dll)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    hadoop_dir = os.path.join(project_root, "hadoop")
    if os.path.exists(os.path.join(hadoop_dir, "bin", "winutils.exe")):
        os.environ["HADOOP_HOME"] = hadoop_dir
        os.environ["hadoop.home.dir"] = hadoop_dir
        os.environ["PATH"] = os.path.join(hadoop_dir, "bin") + os.pathsep + os.environ.get("PATH", "")

    # Đảm bảo timezone nhất quán
    os.environ["TZ"] = "UTC"

    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")  # Khai thác toàn bộ số luồng của Intel Core Ultra 7
        .config("spark.driver.memory", "10g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.executor.memory", "10g")
        .config("spark.memory.fraction", "0.7")
        .config("spark.memory.storageFraction", "0.3")
        .config("spark.sql.shuffle.partitions", "24")
        .config("spark.default.parallelism", "24")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")  # Tự động xử lý lệch dữ liệu (skew join)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark
