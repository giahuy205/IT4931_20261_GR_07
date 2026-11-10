# BÁO CÁO TIẾN ĐỘ ĐỒ ÁN MÔN HỌC: LƯU TRỮ VÀ XỬ LÝ DỮ LIỆU LỚN (IT4931)

**ĐỀ TÀI:** XÂY DỰNG HỆ THỐNG GỢI Ý PHIM CÁ NHÂN HÓA QUY MÔ LỚN TRÊN NỀN TẢNG APACHE SPARK (MOVIELENS 25M)  
**NHÓM THỰC HIỆN:** NHÓM 7  

### THÀNH VIÊN NHÓM
* **Dương Gia Huy** - 20236035
* **Nguyễn Nhật Linh** - 20236040
* **Nguyễn Chí Đức** - 20235042
* **Nguyễn Khoa Bằng** - 20230016

---

## 1. BỐI CẢNH, MỤC TIÊU & NGUỒN DỮ LIỆU MOVIELENS 25M

### 1.1. Bối cảnh Thực tế & Bài toán Kỹ thuật
* **Bối cảnh thực tế:** Sự phát triển của các nền tảng số và dịch vụ streaming tạo ra lượng lớn dữ liệu tương tác người dùng. Trước hàng chục nghìn nội dung, người dùng cần hệ thống gợi ý cá nhân hóa để giảm tình trạng quá tải lựa chọn (Choice Overload).
* **Bài toán kỹ thuật:** Xây dựng hệ thống gợi ý phim có khả năng xử lý ma trận User-Item với hàng chục triệu bản ghi, độ thưa cao và đảm bảo tốc độ phản hồi nhanh.

### 1.2. Mục tiêu Kỹ thuật của Đề tài
* **Kiến trúc dữ liệu:** Xây dựng Medallion Lakehouse nhằm đảm bảo dữ liệu sạch, toàn vẹn và có thể tái lập.
* **Tối ưu lưu trữ:** Sử dụng Parquet nén Snappy, phân vùng theo thời gian để tăng hiệu quả lưu trữ và truy vấn.
* **Học máy phân tán:** Huấn luyện mô hình Spark MLlib ALS và so sánh hiệu năng với mô hình đơn máy Scikit-learn SVD.
* **Phục vụ nghiệp vụ:** Phát triển giao diện trực quan, hỗ trợ đối chiếu và giải thích kết quả gợi ý phim (Explainable UI).
* **Xử lý thời gian thực:** Thiết kế Speed Layer sử dụng Spark Structured Streaming để cập nhật xu hướng phim thịnh hành.

### 1.3. Phân tích Nguồn Dữ liệu MovieLens 25M
* **Lý do lựa chọn MovieLens 25M:**
  1. **Chuẩn nghiên cứu:** Được phát hành bởi GroupLens (Đại học Minnesota), sử dụng rộng rãi trong nghiên cứu hệ thống gợi ý.
  2. **Quy mô Big Data:** Hơn 25 triệu lượt đánh giá từ 162.000 người dùng trên 62.000 bộ phim trong 24 năm, với độ thưa 99.75%.
  3. **Dữ liệu đa dạng:** Bao gồm ratings, tags, genres và mã liên kết ngoài, hỗ trợ làm sạch, trích xuất đặc trưng và phân tích dữ liệu.
* **Các Tệp Dữ liệu Chính:**
  * `ratings.csv` (~646.8 MB): 25 triệu lượt đánh giá từ 0.5 đến 5.0 sao.
  * `movies.csv` (~2.9 MB): 62.423 bộ phim kèm tiêu đề và thể loại.
  * `tags.csv` (~37.7 MB): 1.093.360 thẻ do người dùng gán.
  * `links.csv` (~1.3 MB): mã liên kết IMDB và TMDB.

### 1.4. Kiến trúc Hồ Dữ liệu Phân tầng (Medallion Lakehouse)
* **TẦNG BRONZE (Raw Lake):**
  * *Định dạng:* CSV gốc (`ratings.csv`, `movies.csv`, `tags.csv`...).
  * *Nguyên tắc:* Lưu trữ dữ liệu bất biến, làm nguồn chuẩn (Single Source of Truth) để tái lập pipeline.
* **TẦNG SILVER (Curated Lake):**
  * *Định dạng:* Parquet nén Snappy, phân vùng theo năm (`rating_year`).
  * *Xử lý:* Chuẩn hóa dữ liệu, tách năm sản xuất và thể loại, loại bỏ dữ liệu nhiễu và phim không hợp lệ.
* **TẦNG GOLD (Serving Layer):**
  * *Định dạng:* Parquet tối ưu truy vấn.
  * *Nghiệp vụ:* Lưu sẵn Top-10 phim gợi ý kèm thông tin phim đầy đủ, đảm bảo độ trễ phản hồi dưới 50ms.

---

## 2. TIỀN XỬ LÝ & CHỌN LỌC DỮ LIỆU

### 2.1. Yêu cầu Làm sạch & Đặc trưng Dữ liệu MovieLens 25M
* **Các đặc trưng cần xử lý:**
  * **Chuẩn hóa Schema:** Ép kiểu PySpark (`userId`, `movieId`, `rating`, `timestamp`), loại bỏ dữ liệu null và trùng lặp.
  * **Bóc tách Năm sản xuất:** Dùng Regex trích xuất năm từ tên phim thành cột `year`.
  * **Chuẩn hóa Thể loại:** Tách chuỗi `genres` theo dấu `|` thành `array<string>`.
  * **Phân vùng Thời gian:** Trích xuất `rating_year` từ timestamp để tối ưu lưu trữ và truy vấn.
* **Áp dụng bộ lọc tối thiểu ($\ge 150$ lượt đánh giá/phim trên tập 25M):**
  * **Giảm độ thưa:** Ma trận User-Item có độ thưa trên 99.75%, tồn tại nhiều phim chỉ có 1–2 lượt đánh giá.
  * **Loại bỏ nhiễu:** Phim ít tương tác không đại diện cho thị hiếu cộng đồng, dễ tạo liên kết ngẫu nhiên.
  * **Hạn chế Overfitting:** Tránh ALS học các Latent Factors thiếu tin cậy, dự đoán sai lệch (có thể vượt 5.0★) và lãng phí tài nguyên tính toán cho phim ít giá trị gợi ý.

### 2.2. Thống kê Quy mô Toàn phần 25M (Bronze vs Silver)
| Chỉ số Đo lường | Tập Thô (Bronze 25M) | Sau Lọc Ngưỡng ($\ge 150$ lượt) (Silver) | Mức độ Thay đổi |
| :--- | :---: | :---: | :---: |
| **Lượt đánh giá (Ratings)** | 25 Triệu bản ghi | **23.8 Triệu bản ghi** | **Chỉ mất ~4.7%** (Còn 95.3% cốt lõi) |
| **Số lượng đầu phim** | 59.000 phim | **14.500 phim** | **Giảm 75.4%** (Loại bỏ ~44.500 phim) |
| **Số người dùng tương tác** | 162.541 users | 162.541 users | Bảo toàn 100% |

### 2.3. Điều chỉnh Thông số Ngưỡng & Mẫu Huấn luyện (1M Dòng)
* **Điều chỉnh thông số lọc:** Thu nhỏ tỷ lệ tương ứng từ **$\ge 150$ đánh giá** (trên tập 25M) xuống **$\ge 30$ đánh giá** (trên tập 1M) nhằm bảo đảm cân bằng giữa việc loại bỏ phim rác và giữ độ phủ đa dạng các thể loại phim.
* **Đối chiếu Mẫu 1M:**
  * *Mẫu Gốc (Bronze 1M):* 1.000.000 dòng / 20.430 phim (6.746 người dùng).
  * *Sau Lọc Ngưỡng ($\ge 30$ lượt):* 911.285 dòng / 4.448 phim.
  * *Số lượng phim giảm:* Giảm 78.2% (Loại bỏ 15.982 phim ít tương tác).
  * *Lượt đánh giá mất đi:* Chỉ mất 8.87% (Giữ lại 91.13% dữ liệu có giá trị).

---

## 3. LƯU TRỮ & BENCHMARK DỮ LIỆU (CSV VS PARQUET SNAPPY)

### 3.1. Đặc trưng Dữ liệu & Yêu cầu Lưu trữ MovieLens 25M
* **Quy mô dữ liệu:** `ratings.csv` gần 650 MB, hơn 25 triệu bản ghi, đòi hỏi lưu trữ tối ưu và hỗ trợ Spark đọc/ghi phân tán, tránh nghẽn I/O.
* **Tối ưu truy vấn ALS:** Spark ALS chủ yếu sử dụng 3 cột `userId`, `movieId`, `rating` qua nhiều vòng lặp. Đọc toàn bộ CSV gây lãng phí I/O và bộ nhớ.
* **Bảo toàn Schema:** Parquet Silver lưu giữ kiểu dữ liệu gốc (Integer, Float, Array), giảm chi phí CPU do parse text khi nạp dữ liệu huấn luyện.

### 3.2. Kết quả Đo đạc Định lượng (CSV vs Parquet Snappy)
| Tiêu chí Đánh giá | Định dạng CSV (Gốc) | Định dạng Parquet Snappy | Mức độ Tối ưu |
| :--- | :---: | :---: | :---: |
| **Dung lượng Lưu trữ** | 646.84 MB | **14.03 MB** | **Giảm 97.83%** (Tiết kiệm đĩa) |
| **Tốc độ Scan & Count** | 7.43 giây | **1.05 giây** | **Nhanh hơn 7.1x** |
| **Tốc độ GroupBy Aggregation** | 4.41 giây | **0.93 giây** | **Nhanh hơn 4.72x** (Column Pruning) |
| **Tốc độ Filter Pushdown** | 2.64 giây | **0.46 giây** | **Nhanh hơn 5.79x** (Predicate Pushdown) |

### 3.3. Cơ chế Kỹ thuật tạo nên sự vượt trội của Parquet Snappy
1. **Lưu trữ dạng cột (Columnar Storage) & Tỉa cột (Column Pruning):** Parquet lưu dữ liệu theo từng cột thay vì từng dòng như CSV. Khi Spark chỉ cần cột `rating`, hệ thống chỉ đọc dữ liệu của cột này, bỏ qua các cột không cần thiết, giúp giảm I/O và tiết kiệm RAM.
2. **Tỉa phân vùng & Đẩy bộ lọc (Partition Pruning & Predicate Pushdown):** Dữ liệu được phân vùng theo `rating_year=XXXX`. Spark sử dụng metadata Min/Max trong Footer của mỗi Row Group để loại bỏ sớm các vùng dữ liệu không thỏa điều kiện lọc, tránh đọc và giải nén dữ liệu không cần thiết.
3. **Nén Snappy tốc độ cao (Snappy Block Compression):** Parquet gom dữ liệu cùng kiểu vào từng cột, giúp Snappy nén hiệu quả. Dung lượng giảm từ 646.8 MB xuống 14.03 MB (tiết kiệm 97.83%), đồng thời giải nén nhanh với chi phí CPU thấp.

---

## 4. HUẤN LUYỆN MÔ HÌNH & ĐỐI SÁNH HIỆU NĂNG

### 4.1. Mẫu Dữ liệu Huấn luyện & Lý do Lựa chọn 2 Mô hình
* **Lý do lựa chọn tập mẫu 1M dòng cho giai đoạn này:**
  1. *Tối ưu tốc độ huấn luyện:* Giảm thời gian chạy Spark ALS từ ~45 phút xuống 1–2 phút, đảm bảo demo mượt mà và tránh quá tải bộ nhớ.
  2. *Bảo toàn đặc trưng dữ liệu:* Duy trì cấu trúc ma trận thưa và phân phối người dùng, phục vụ kiểm chứng thuật toán.
  3. *Đảm bảo đối chứng công bằng:* Giới hạn dữ liệu phù hợp với khả năng xử lý của Scikit-learn SVD (~26.12 MB RAM cho ma trận thưa), tránh lỗi tràn bộ nhớ (OOM) khi xử lý toàn bộ 25M dòng.
* **Phân chia dữ liệu:**
  * Dữ liệu Sạch nạp vào Model (Silver): **911.285 dòng** (6.746 users $\times$ 4.448 phim).
  * Tập Huấn luyện (Train 80%): **728.892 dòng** (`seed=42`).
  * Tập Kiểm thử (Test 20%): **182.393 dòng** (Đo lường RMSE độc lập).
* **Lý do lựa chọn 2 mô hình:**
  * **Scikit-learn TruncatedSVD (Baseline Đơn máy):** Đại diện cho phương pháp phân rã ma trận truyền thống trên máy đơn, đóng vai trò mô hình cơ sở để đo lường định lượng sự khác biệt về sai số và giới hạn tài nguyên.
  * **Apache Spark MLlib ALS (Trọng tâm Phân tán):** Mô hình cốt lõi của đề tài được thiết kế chuyên biệt cho hệ phân tán, có khả năng tối ưu hóa song song trên ma trận thưa quy mô lớn và mở rộng tuyến tính trên Big Data.

### 4.2. Bản chất Thuật toán, Cơ chế Hoạt động & Công thức Toán học
* **Scikit-learn TruncatedSVD (Đơn máy / Centralized):**
  * *Bản chất:* Phân rã ma trận xấp xỉ bậc thấp (Low-Rank Approximation) dựa trên đại số tuyến tính:
    $$R \approx U \cdot \Sigma \cdot V^T$$
  * *Cơ chế trên ma trận thưa (Unobserved = 0):* Coi các ô chưa đánh giá là 0. Với độ thưa > 99.75%, hàng triệu ô trống làm lệch kết quả dự đoán về 0, khiến Test RMSE cao (**2.6898**).
  * *Xử lý & giới hạn phần cứng:* Chạy trên CPU một máy, nạp ma trận CSR vào RAM cục bộ. Xử lý được 1M dòng (~26 MB RAM) nhưng sẽ lập tức bị lỗi tràn bộ nhớ (**Out Of Memory - OOM**) khi mở rộng lên 25M dòng.

* **Apache Spark MLlib ALS (Phân tán / Distributed):**
  * *Bản chất:* Phân rã ma trận thành 2 ma trận vector ẩn người dùng $u_u \in \mathbb{R}^k$ và phim $v_i \in \mathbb{R}^k$ ($k=20$), điểm dự đoán là tích vô hướng:
    $$\hat{r}_{u,i} = u_u^T \cdot v_i$$
  * *Hàm mục tiêu tối ưu (Loss Function):* Chỉ tối ưu trên tập các ô **thực sự có đánh giá** ($R_{obs}$), không coi ô trống là 0, kết hợp điều chuẩn $L_2$ chống Overfitting:
    $$\min_{U, V} \sum_{(u,i) \in R_{obs}} (r_{u,i} - u_u^T v_i)^2 + \lambda \left( \sum_u \|u_u\|_2^2 + \sum_i \|v_i\|_2^2 \right)$$
  * *Cơ chế luân phiên (Alternating):* Lần lượt cố định $V$ để giải song song cho từng $u_u$; sau đó cố định $U$ để giải song song cho từng $v_i$. Phân tán trên 24 Spark Partitions, giải quyết triệt để độ thưa và đạt Test RMSE thấp hơn (**0.8091**).

### 4.3. Kết quả Thực nghiệm Đối sánh Định lượng
| Tiêu chí Đối sánh | Scikit-learn SVD (Đơn máy) | Apache Spark MLlib ALS (Hệ phân tán) | Đánh giá Chuyên môn |
| :--- | :---: | :---: | :--- |
| **Sai số Dự đoán (Test RMSE)** | 2.6898 | **0.8091** | **Giảm 69.9% sai số** (Chính xác hơn 3.3x) |
| **Thời gian Huấn luyện (1M Dòng)** | **0.27 giây** | 34.48 giây | Spark chịu chi phí khởi tạo phân tán |
| **Khả năng Mở rộng lên 25M Dòng** | **Crash (OOM)** | **~45 – 60 phút** | Spark xử lý ổn định trên Cluster phân tán |

* **Phân tích Hiệu năng & Giới hạn Quy mô (1M vs 25M Dòng):**
  * **Cấu hình & Tham số Spark ALS được thiết lập:** `rank = 20`, `maxIter = 15`, `regParam = 0.1`, `partitions = 24`, `Driver/Executor RAM = 10 GB`, `checkpointInterval = 2`.
  * **Thời gian huấn luyện trên 1M dòng:** Spark ALS mất 34.48s, lâu hơn SVD (0.27s) do chi phí khởi tạo JVM, chia 24 Partitions, shuffle dữ liệu và checkpoint qua 10 vòng lặp. Trong khi đó, SVD chạy cục bộ trên một CPU, xử lý ma trận thưa chỉ chiếm 26.12 MB RAM, nên có tốc độ tính toán nhanh hơn.
  * **Khả năng mở rộng lên 25M dòng:**
    * *Scikit-learn SVD:* Không thể xử lý ổn định trên máy đơn; ma trận kích thước 162.000 users × 62.000 movies có nguy cơ gây Out Of Memory (OOM).
    * *Spark MLlib ALS:* Hoàn thành trong khoảng 45–60 phút nhờ chia nhỏ dữ liệu thành các Block, hỗ trợ spill-to-disk khi thiếu RAM và tính toán song song trên cluster.
  * **Chống lỗi JVM:** Cấu hình `-Xss8m` và Checkpoint mỗi 2 vòng lặp (`checkpointInterval = 2`) giúp rút ngắn RDD DAG Lineage, hạn chế lỗi `StackOverflowError`.

---

## 5. DEMO GỢI Ý CÁ NHÂN HÓA (TẦNG GOLD SERVING)

### 5.1. Bố cục Đối chiếu Song song Giữa Lịch sử và Gợi ý Mô hình (Explainable UI)
* **Ô chọn User ID duy nhất:** Hỗ trợ chọn trực tiếp từ danh sách hoặc nhập số ID bất kỳ để phân tích.
* **Hai bảng đối chiếu trực quan:**
  * **Bảng bên trái (Lịch sử 10 Phim Đánh giá Cao nhất):** Hiển thị Top 10 phim người dùng từng chấm $4.0 - 5.0\text{ ⭐}$ trong quá khứ kèm cột số thứ tự `#` (1–10), Năm sản xuất, Thể loại và Điểm đã chấm.
  * **Bảng bên phải (Top 10 Phim Mô hình Gợi ý Spark ALS):** Hiển thị 10 phim do Spark ALS đề xuất (**đã loại trừ toàn bộ phim người dùng đã xem**), kèm Năm sản xuất, Thể loại và **Điểm Dự đoán (ALS)** dạng `X.XX ⭐`.
* **Minh chứng tính giải thích được (Explainability):** Đối chiếu trực quan giữa 2 cột cho thấy danh sách gợi ý của Spark ALS phản ánh chuẩn xác gu thưởng thức (thể loại, phong cách) mà người dùng ưa thích trong quá khứ, đồng thời bảo đảm tính mới mẻ và loại bỏ hoàn toàn các phim rác vô danh.

### 5.2. Giải pháp cho Bài toán Người dùng mới (Cold-Start Onboarding & ALS Folding-in)
* **Vấn đề:** Người dùng mới đăng ký vào hệ thống chưa có vector đặc trưng trong ma trận người dùng $U$. Việc huấn luyện lại toàn bộ mô hình Spark ALS cho mỗi người dùng mới là bất khả thi vì tốn thời gian tính toán và độ trễ cao.
* **Kỹ thuật ALS Folding-in:**
  * Giữ cố định ma trận đặc trưng các bộ phim $V \in \mathbb{R}^{M \times k}$ đã được huấn luyện sẵn từ tầng Silver.
  * Sau khi người dùng mới đánh giá nhanh $m$ bộ phim ($m \ll M$, vector điểm $r_{new}$), vector sở thích $u_{new} \in \mathbb{R}^k$ được giải nghiệm tối ưu tức thì bằng hồi quy Ridge:
    $$u_{new} = \left(V_{sub}^T V_{sub} + \lambda I\right)^{-1} V_{sub}^T r_{new}$$
  * Phép tính nghịch đảo ma trận $20 \times 20$ chỉ mất chưa đầy **0.001 giây** trên CPU, sau đó tính ngay điểm dự đoán $\hat{r} = u_{new} \cdot V^T$ để xuất Top-10 phim theo thời gian thực.

---

## 6. TIẾN ĐỘ ĐỒ ÁN & KẾ HOẠCH TRIỂN KHAI GIAI ĐOẠN TỚI

### 6.1. Tiến độ Đã Hoàn thành (Giai đoạn 1)
* **Kiến trúc Medallion:** Luồng 3 tầng tự động Bronze $\to$ Silver $\to$ Gold.
* **Tiền xử lý dữ liệu:** Chuẩn hóa Schema và lọc ngưỡng giảm độ thưa.
* **Tối ưu lưu trữ:** Parquet Snappy giảm 97.83% dung lượng, tăng tốc 7.1x.
* **Huấn luyện mô hình:** Spark ALS đạt RMSE 0.8091, vượt trội SVD.
* **Serving & Dashboard:** Xuất tầng Gold, hoàn thiện đối chiếu trực quan Streamlit.

### 6.2. Kế hoạch Triển khai Giai đoạn Tới (Tuần 8 – Tuần 10)
* **Tối ưu siêu tham số Spark:** Dò tìm cấu hình ALS tốt nhất bằng `ParamGridBuilder`.
* **Xử lý người dùng mới:** Tích hợp ALS Folding-in phản hồi tức thì dưới 0.001s.
* **Mở rộng toàn phần 25M:** Huấn luyện full 25M dòng và đo đường cong Scalability.
* **Giao diện FE tương tác:** Cho phép người dùng chọn thể loại, nhận gợi ý realtime.
