# Hệ thống Gợi ý Phim Cá nhân hóa Quy mô lớn - Apache Spark
> **Học phần:** IT4931 - Lưu trữ và xử lý dữ liệu lớn (HUST)  
> **Nhóm thực hiện:** Nhóm 7  
> **Thành viên:**  
> - Dương Gia Huy - 20236035  
> - Nguyễn Nhật Linh - 20236040  
> - Nguyễn Chí Đức - 20235042  
> - Nguyễn Khoa Bằng - 20230016  

Chi tiết đề xuất đề tài xem tại: [ĐỀ XUẤT ĐỀ TÀI PROJECT MÔN BIG DATA.md](./%5BNH%C3%93M%207%5D%20%C4%90%E1%BB%80%20XU%E1%BA%A4T%20%C4%90%E1%BB%80%20T%C3%80I%20PROJECT%20M%C3%94N%20BIG%20DATA.md)

---

## 1. Cài đặt Môi trường

* **Yêu cầu:** Python 3.10+, Java JDK 21 (đặt tại `C:\Program Files\Java\jdk-21.0.12.1`).
* **Cài đặt thư viện:**
  ```powershell
  pip install -r requirements.txt
  ```

---

## 2. Hướng dẫn Chạy Nhanh (Quickstart)

### Bước 1: Tải và Giải nén Dữ liệu MovieLens 25M
Tải tự động `ml-25m.zip` (250 MB) và giải nén toàn bộ 25 triệu dòng vào thư mục `data/bronze/`:
```powershell
python src/batch/1_download_data.py
```

### Bước 2: Chạy Pipeline Làm sạch dữ liệu (Bronze $\rightarrow$ Silver)
Bóc tách năm, chuẩn hóa mảng thể loại, khử trùng lặp và lưu ra Parquet nén Snappy (phân vùng theo năm):
```powershell
# Chế độ test nhanh với 1 triệu dòng:
python src/batch/2_bronze_to_silver.py --sample

# Chế độ chạy toàn bộ 25 triệu dòng chính thức:
python src/batch/2_bronze_to_silver.py
```

### Bước 3: Đo đạc Hiệu năng Big Data (Benchmark CSV vs Parquet)
Tự động so sánh dung lượng lưu trữ, tốc độ Scan, Column Pruning, Predicate Pushdown và lưu kết quả vào `data/benchmark_results.json`:
```powershell
python src/benchmark/benchmark_csv_vs_parquet.py
```

### Bước 4: Mở Dashboard xem kết quả Tuần 4 trên Localhost
Trực quan hóa dữ liệu EDA (25M dòng) và biểu đồ Benchmark hiệu năng:
```powershell
python -m streamlit run src/dashboard/app_week4.py
```
> Truy cập giao diện tại: **`http://localhost:8501`**

---

## 3. Cấu trúc Thư mục Dự án

```text
.
├── data/
│   ├── bronze/                   # Dữ liệu thô gốc (CSV) từ MovieLens 25M
│   ├── silver/                   # Dữ liệu đã làm sạch (Parquet nén Snappy)
│   ├── gold/                     # Dữ liệu phục vụ ALS Model & Top-N Serving
│   └── benchmark_results.json    # Kết quả đo đạc hiệu năng thực tế
├── hadoop/bin/                   # winutils.exe & hadoop.dll hỗ trợ Windows
├── src/
│   ├── common/
│   │   ├── spark_session.py      # Cấu hình SparkSession (10GB RAM, 24 luồng, AQE)
│   │   └── schemas.py            # StructType tường minh cho toàn bộ tệp CSV
│   ├── batch/
│   │   ├── 1_download_data.py    # Tải và giải nén dữ liệu MovieLens 25M
│   │   └── 2_bronze_to_silver.py # Pipeline ETL làm sạch & lưu Parquet
│   ├── benchmark/
│   │   └── benchmark_csv_vs_parquet.py # Benchmark đo hiệu năng CSV vs Parquet
│   └── dashboard/
│       └── app_week4.py          # Dashboard Streamlit báo cáo tiến độ Tuần 4
├── requirements.txt
└── README.md
```

