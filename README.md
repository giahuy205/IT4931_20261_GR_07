# Hệ thống Gợi ý Phim Cá nhân hóa Quy mô lớn - Apache Spark

> **Học phần:** IT4931 - Lưu trữ và Xử lý Dữ liệu Lớn (Đại học Bách Khoa Hà Nội - HUST)  
> **Nhóm thực hiện:** Nhóm 7  
> **Thành viên:**  
> - **Dương Gia Huy** - 20236035 (Nhóm trưởng)  
> - **Nguyễn Nhật Linh** - 20236040  
> - **Nguyễn Chí Đức** - 20235042  
> - **Nguyễn Khoa Bằng** - 20230016  

📄 **Tài liệu Báo cáo:**
* [BÁO CÁO TIẾN ĐỘ DỰ ÁN BIG DATA.md](./%5BNH%C3%93M%207%5D%20B%C3%81O%20C%C3%81O%20TI%E1%BA%BEN%20%C4%90%E1%BB%98%20D%E1%BB%B0%20%C3%81N%20BIG%20DATA.md)
* [ĐỀ XUẤT ĐỀ TÀI PROJECT MÔN BIG DATA.md](./%5BNH%C3%93M%207%5D%20%C4%90%E1%BB%80%20XU%E1%BA%A4T%20%C4%90%E1%BB%80%20T%C3%80I%20PROJECT%20M%C3%94N%20BIG%20DATA.md)

---

## 1. Kết quả Thực nghiệm Định lượng Nổi bật

### 1.1. Hiệu năng Lưu trữ: CSV vs Parquet Snappy (Tầng Bronze vs Silver)
* **Tiết kiệm Dung lượng:** Giảm **97.83%** (từ **646.84 MB** xuống **14.03 MB**).
* **Tốc độ Quét (Scan & Count):** Nhanh hơn **7.10x** (1.05s so với 7.43s).
* **Tốc độ Tổng hợp (GroupBy Aggregation):** Nhanh hơn **4.72x** (nhờ cơ chế *Column Pruning*).
* **Tốc độ Lọc có điều kiện (Filter Pushdown):** Nhanh hơn **5.79x** (nhờ cơ chế *Predicate Pushdown*).

### 1.2. Đối sánh Mô hình: Apache Spark MLlib ALS vs Scikit-learn TruncatedSVD
* **Sai số Dự đoán (Test RMSE):** Spark ALS đạt **0.8091**, vượt trội so với SVD đơn máy **2.6898** (**Giảm 69.9% sai số**, chính xác hơn 3.3 lần).
* **Khả năng Mở rộng lên 25M dòng:**
  * *Scikit-learn SVD:* **Crash do tràn bộ nhớ (Out Of Memory - OOM)** trên máy đơn.
  * *Spark MLlib ALS:* **Hoàn thành ổn định trong ~45 – 60 phút** nhờ cơ chế chia nhỏ BlockMatrix, hỗ trợ spill-to-disk và tính toán phân tán.
* **Ổn định JVM:** Cấu hình `-Xss8m` và Checkpoint định kỳ (`checkpointInterval = 2`) loại bỏ 100% lỗi `StackOverflowError`.

### 1.3. Giải pháp Người dùng mới (User Cold-Start)
* Kỹ thuật **ALS Folding-in** giải nghiệm hồi quy Ridge trực tiếp $u_{new} = (V_{sub}^T V_{sub} + \lambda I)^{-1} V_{sub}^T r_{new}$ dưới **0.001 giây** trên CPU, loại bỏ việc phải huấn luyện lại toàn bộ mô hình cho người dùng mới.

---

## 2. Cài đặt Môi trường

* **Yêu cầu hệ thống:** Python 3.10+, Java OpenJDK 21 (đặt tại `C:\Program Files\Java\jdk-21.0.12.1`), Windows Hadoop native binaries (`hadoop/bin/winutils.exe`).
* **Cài đặt các gói phụ thuộc:**
  ```powershell
  pip install -r requirements.txt
  ```

---

## 3. Hướng dẫn Tái lập Kết quả (Quickstart)

### Bước 1: Tải và Giải nén Dữ liệu MovieLens 25M
Tải tự động `ml-25m.zip` (250 MB) và giải nén 25 triệu dòng đánh giá vào thư mục `data/bronze/`:
```powershell
python src/batch/1_download_data.py
```

### Bước 2: Chạy Pipeline Làm sạch & Tinh lọc (Bronze $\rightarrow$ Silver)
Chuẩn hóa Schema, bóc tách năm sản xuất bằng Regex, chuẩn hóa mảng thể loại, áp dụng ngưỡng lọc phổ biến và lưu Parquet nén Snappy phân vùng theo `rating_year`:
```powershell
# Chế độ mẫu 1 triệu dòng (dùng cho thực nghiệm đối sánh giữa kỳ):
python src/batch/2_bronze_to_silver.py --sample --min-ratings 30

# Chế độ chạy toàn bộ 25 triệu dòng:
python src/batch/2_bronze_to_silver.py --min-ratings 150
```

### Bước 3: Đo đạc Hiệu năng Lưu trữ (Benchmark CSV vs Parquet)
Tự động so sánh dung lượng lưu trữ, tốc độ Scan, Column Pruning, Predicate Pushdown và lưu kết quả vào `data/benchmark_results.json`:
```powershell
python src/benchmark/benchmark_csv_vs_parquet.py
```

### Bước 4: Huấn luyện & Đối sánh Mô hình (Spark ALS vs Scikit-learn SVD)
Thực nghiệm đối chứng 1-1 giữa Single-node ML và Distributed ML, lưu mô hình vào `models/als_best_model/` và xuất danh sách gợi ý ra tầng Gold (`data/gold/user_recommendations_top10.parquet`):
```powershell
# Chạy đối sánh trên tập mẫu 1 triệu dòng và xuất tầng Gold:
python src/batch/3_train_and_compare_models.py --sample-size 1000000

# Chế độ chạy toàn bộ 25 triệu dòng trên Spark ALS:
python src/batch/3_train_and_compare_models.py --full
```

### Bước 5: Khởi chạy Dashboard Báo cáo Trực quan hóa
Báo cáo toàn diện với 6 tab tương tác chuẩn học thuật:
```powershell
streamlit run src/dashboard/app_week4.py
```
> Truy cập giao diện ứng dụng tại: **`http://localhost:8501`**

---

## 4. Cấu trúc Giao diện Dashboard (6 Tabs)

1. **Tab 1: Bối cảnh, Mục tiêu & Nguồn Dữ liệu:** Bối cảnh thực tế, mục tiêu kỹ thuật, phân tích nguồn dữ liệu MovieLens 25M và kiến trúc 3 tầng Medallion Lakehouse (Bronze, Silver, Gold).
2. **Tab 2: Tiền xử lý & Chọn lọc Dữ liệu:** Quy trình bóc tách đặc trưng, bộ lọc ngưỡng phổ biến ($\ge 150$ trên 25M và $\ge 30$ trên 1M), phân bố số lượng phim theo các khoảng đánh giá.
3. **Tab 3: Lưu trữ & Benchmark Dữ liệu:** 4 thẻ KPI dung lượng & truy vấn, biểu đồ so sánh chi tiết và giải thích 3 cơ chế kỹ thuật của Parquet Snappy.
4. **Tab 4: Huấn luyện & Đối sánh Mô hình:** Công thức toán học (SVD vs Spark ALS), biểu đồ đối sánh Test RMSE & thời gian chạy, phân tích chi phí khởi tạo JVM và giới hạn OOM trên 25M dòng.
5. **Tab 5: Demo Gợi ý Cá nhân hóa (Explainable UI):** Đối chiếu song song giữa **Lịch sử 10 phim đánh giá cao nhất** và **Top 10 phim gợi ý từ Spark ALS** (kèm điểm số dự đoán cá nhân hóa `X.XX ⭐`), cơ chế ALS Folding-in cho người dùng mới.
6. **Tab 6: Tiến độ & Kế hoạch:** Tổng kết ngắn gọn các hạng mục đã hoàn thành giai đoạn 1 và kế hoạch 4 trọng tâm giai đoạn tiếp theo (Tối ưu siêu tham số, ALS Folding-in, Huấn luyện full 25M, Giao diện tương tác Onboarding).

---

## 5. Cấu trúc Thư mục Dự án

```text
.
├── data/
│   ├── bronze/                       # Dữ liệu thô gốc (CSV) từ MovieLens 25M
│   ├── silver/                       # Dữ liệu sạch (Parquet Snappy phân vùng theo rating_year)
│   ├── gold/                         # Tầng Serving: user_recommendations_top10.parquet
│   ├── benchmark_results.json        # Số liệu đo đạc hiệu năng CSV vs Parquet Snappy
│   └── model_comparison_results.json # Kết quả đối sánh thực nghiệm Spark ALS vs SVD
├── models/
│   └── als_best_model/               # Trọng số mô hình Spark ALS tốt nhất đã lưu
├── hadoop/bin/                       # winutils.exe & hadoop.dll hỗ trợ môi trường Windows
├── src/
│   ├── common/
│   │   ├── spark_session.py          # Cấu hình Spark Session tối ưu (10GB RAM, 24 luồng, AQE)
│   │   └── schemas.py                # Định nghĩa StructType tường minh cho các bảng
│   ├── batch/
│   │   ├── 1_download_data.py        # Tự động tải và giải nén MovieLens 25M
│   │   ├── 2_bronze_to_silver.py     # Pipeline ETL làm sạch, tinh lọc và lưu Silver Parquet
│   │   └── 3_train_and_compare_models.py # Huấn luyện Spark ALS, đối sánh SVD và xuất tầng Gold
│   ├── benchmark/
│   │   └── benchmark_csv_vs_parquet.py # Script đo lường định lượng CSV vs Parquet Snappy
│   └── dashboard/
│       ├── app_week4.py              # Dashboard Streamlit chính thức 6 tabs
│       └── app.py                    # Entrypoint cho dashboard
├── [NHÓM 7] BÁO CÁO TIẾN ĐỘ DỰ ÁN BIG DATA.md # Báo cáo tiến độ chi tiết
├── [NHÓM 7] ĐỀ XUẤT ĐỀ TÀI PROJECT MÔN BIG DATA.md # Đề xuất đề tài ban đầu
├── requirements.txt
└── README.md
```
