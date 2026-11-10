# **IT4931 \- Lưu trữ và xử lý dữ liệu lớn** **Đề xuất đề tài**

## **1\. Thông tin nhóm thực hiện \- Nhóm 7**

* **Thành viên 1:** Dương Gia Huy \- 20236035  
* **Thành viên 2:** Nguyễn Nhật Linh \- 20236040  
* **Thành viên 3:** Nguyễn Chí Đức \- 20235042   
* **Thành viên 4:** Nguyễn Khoa Bằng \- 20230016

## **2\. Tên đề tài đề xuất**

***Hệ thống gợi ý phim cá nhân hóa quy mô lớn sử dụng Apache Spark (Distributed Movie Recommendation System)***

## **3\. Mô tả bài toán và bối cảnh thực tế**

* **Bối cảnh:** Trong các nền tảng giải trí trực tuyến quy mô lớn hiện nay như Netflix hay VieON lượng dữ liệu về lịch sử xem, tương tác và đánh giá của hàng triệu người dùng được sinh ra liên tục. Việc cá nhân hóa trải nghiệm thông qua hệ thống gợi ý sản phẩm/phim chính xác đóng vai trò cốt lõi giữ chân người dùng.  
* **Bài toán:** Xây dựng một hệ thống phân tích dữ liệu lớn để xử lý tập hợp hàng triệu lượt đánh giá phim (ratings) từ người dùng, từ đó huấn luyện mô hình học máy phân tán nhằm dự đoán sở thích và tự động đề xuất danh sách các bộ phim phù hợp nhất cho từng cá nhân.

## **4\. Mục tiêu đề tài**

Đề tài hướng đến xây dựng hệ thống gợi ý phim cá nhân hóa sử dụng Apache Spark, áp dụng kỹ thuật xử lý dữ liệu phân tán và học máy để phân tích dữ liệu đánh giá phim quy mô lớn.

Các mục tiêu cụ thể:

* **Xử lý dữ liệu:** Sử dụng PySpark để nạp, làm sạch và phân tích dữ liệu MovieLens.  
* **Xây dựng mô hình:** Áp dụng thuật toán ALS trong Spark MLlib để dự đoán sở thích người dùng.  
* **Đánh giá mô hình:** Sử dụng chỉ số RMSE để đánh giá độ chính xác dự đoán.  
* **Gợi ý phim:** Tạo danh sách Top-N phim phù hợp với từng người dùng.  
* **Đánh giá hiệu năng:** Thực nghiệm và so sánh thời gian xử lý với các cấu hình phân vùng dữ liệu khác nhau.  
* **Trực quan hóa dữ liệu:** Xây dựng dashboard bằng Streamlit để hiển thị kết quả phân tích và gợi ý phim.  
* **Xử lý dữ liệu theo luồng:** Mô phỏng các lượt đánh giá mới và sử dụng Spark Structured Streaming để cập nhật thống kê.

## **5\. Nguồn dữ liệu của đề tài**

Nhóm sử dụng bộ dữ liệu MovieLens 25M do GroupLens Research cung cấp.

* **Trang dữ liệu chính:** [https\://grouplens.org/datasets/movielens/25m/](https://grouplens.org/datasets/movielens/25m/)  
* **Link tải dữ liệu:** [https\://files.grouplens.org/datasets/movielens/ml-25m.zip](https://files.grouplens.org/datasets/movielens/ml-25m.zip)  
* **Tài liệu mô tả dữ liệu:** [https\://files.grouplens.org/datasets/movielens/ml-25m-README.html](https://files.grouplens.org/datasets/movielens/ml-25m-README.html)

Bộ dữ liệu có khoảng 25 triệu lượt đánh giá, hơn 62.000 bộ phim và dữ liệu gắn nhãn do người dùng tạo. Dữ liệu được cung cấp dưới dạng CSV và có thể sử dụng cho mục đích nghiên cứu theo điều kiện của GroupLens.

Các tệp dữ liệu chính:

* `ratings.csv`: Lịch sử đánh giá phim của người dùng.  
* `movies.csv`: Tên phim và thể loại.  
* `tags.csv`: Các nhãn mô tả phim do người dùng cung cấp.  
* `genome-scores.csv`: Điểm liên quan giữa phim và các nhãn.  
* `genome-tags.csv`: Nội dung của các nhãn tương ứng.

Nhóm sẽ sử dụng các tệp này để kết hợp dữ liệu, phân tích hành vi đánh giá và xây dựng mô hình gợi ý phim.

### **Làm rõ bản chất đặc trưng người dùng (User Profile vs. Interaction Profile)**
Trong bộ dữ liệu **MovieLens 25M**, GroupLens **hoàn toàn không thu thập thông tin nhân khẩu học (Demographic Profile)** như tuổi, giới tính, nghề nghiệp, khu vực địa lý (khác với các bộ dữ liệu nhỏ trước đây như MovieLens 100K hay 1M). Do đó, đề tài xác định rõ hệ thống khai thác và xây dựng **Hồ sơ Tương tác và Hành vi Người dùng (Interaction-based & Derived Behavioral Profile)**, bao gồm:
1. **Lịch sử tương tác rõ ràng (Explicit Interaction Matrix):** Bộ ba thuộc tính cốt lõi (userId, movieId, rating) cùng dấu thời gian timestamp.
2. **Hồ sơ thiên hướng thể loại (User Genre Preference Profile):** Vector đặc trưng dẫn xuất phản ánh phân bố lượt xem và điểm đánh giá trung bình của từng người dùng đối với 19 thể loại phim khác nhau (Action, Romance, Sci-Fi,...).
3. **Độ lệch và hành vi đánh giá cá nhân (User Rating Behavior & Bias):** Tần suất đánh giá phim, độ khó tính cá nhân (điểm trung bình của user so với điểm trung bình toàn hệ thống).
4. **Hành vi gán nhãn cộng đồng (User Tagging Activity):** Tập hợp các từ khóa/thẻ do người dùng chủ động gán cho phim trong `tags.csv`.

### **Phân tích đặc trưng dữ liệu lớn (3V)**

Tập dữ liệu MovieLens có các đặc trưng của dữ liệu lớn như sau:

| Đặc trưng | Mô tả |
| ----- | ----- |
| **Volume (Khối lượng)** | MovieLens 25M chứa khoảng 25 triệu lượt đánh giá phim và dữ liệu phụ trợ (~1.1 GB file thô CSV, hơn 62.000 phim, 162.000 người dùng, 14 triệu điểm genome), tạo điều kiện thực nghiệm xử lý dữ liệu quy mô lớn và tối ưu hóa bộ nhớ RAM/Shuffle bằng Apache Spark. |
| **Velocity (Tốc độ)** | Mô phỏng dữ liệu tương tác người dùng phát sinh liên tục theo thời gian thực bằng cách chia nhỏ dữ liệu đánh giá và đưa liên tục vào thư mục nguồn, sử dụng Spark Structured Streaming với cơ chế micro-batch (trigger 3–5 giây) để xử lý cửa sổ trượt (windowing) và phát hiện phim thịnh hành (Trending). |
| **Variety (Đa dạng)** | Dữ liệu gồm nhiều cấu trúc và dạng thức khác nhau: điểm đánh giá dạng số nguyên/thực (`ratings`), nhãn văn bản tự do (`tags`), thể loại dạng danh sách chuỗi (`genres`), ma trận mức độ liên quan nhiều chiều (`genome-scores`) và chuỗi thời gian Unix (`timestamp`). |

## **6. Kiến trúc hệ thống và Công nghệ dự kiến**

### **6.1. Kiến trúc hệ thống**

Hệ thống được thiết kế theo mô hình xử lý kép (Hybrid Batch & Streaming Architecture) kết hợp **Kiến trúc phân tầng Medallion (Bronze – Silver – Gold)**:

1. **Tầng Thu thập & Nạp dữ liệu (Ingestion Layer):**
   * *Luồng Batch:* Nạp toàn bộ dữ liệu MovieLens 25M thô (CSV) vào khu vực lưu trữ Bronze.
   * *Luồng Streaming:* Script mô phỏng người dùng phát sinh lượt đánh giá mới theo thời gian thực vào thư mục giám sát.
2. **Tầng Lưu trữ phân tầng (Medallion Storage Layer):**
   * *Bronze Zone:* Dữ liệu thô ban đầu (CSV), lưu trữ nguyên bản, bất biến.
   * *Silver Zone:* Dữ liệu đã làm sạch, chuẩn hóa kiểu dữ liệu, bóc tách đặc trưng, lưu trữ định dạng Parquet nén Snappy.
   * *Gold Zone:* Bảng dữ liệu ma trận phục vụ huấn luyện ALS, bảng tổng hợp phân tích (Aggregated tables) và bảng kết quả gợi ý Top-N.
3. **Tầng Xử lý & Phân tích (Processing & Analytics Layer):**
   * *Batch Layer (Apache Spark & Spark MLlib):* Thực hiện tiền xử lý, kiểm tra hợp lệ, phân vùng song song và huấn luyện mô hình phân tán ALS để tìm vector ẩn người dùng/phim, sinh danh sách Top-10 phim gợi ý.
   * *Speed Layer (Spark Structured Streaming):* Xử lý luồng micro-batch, áp dụng Watermarking và Windowing để tính toán chỉ số thống kê tức thời (các bộ phim đang được đánh giá nhiều nhất trong 10 phút gần nhất).
4. **Tầng Trực quan hóa & Phục vụ (Serving & Visualization Layer):**
   * Xây dựng Dashboard tương tác bằng Streamlit để tra cứu gợi ý cá nhân hóa cho từng `userId`, hiển thị báo cáo phân tích xu hướng phim và theo dõi luồng xử lý thời gian thực.

**Luồng xử lý tổng thể:** MovieLens 25M & Luồng mô phỏng → Apache Spark Ingestion → Lưu trữ phân tầng Medallion (Parquet) → Phân tích Spark SQL & Huấn luyện ALS MLlib / Xử lý luồng Structured Streaming → Phục vụ trên Dashboard Streamlit.

![][image1]

### **6.2. Công nghệ sử dụng**

| Công nghệ | Mục đích |
| ----- | ----- |
| Python (PySpark) | Lập trình xử lý dữ liệu và xây dựng pipeline phân tán |
| Apache Spark, Spark SQL | Đọc, biến đổi, kiểm tra hợp lệ, tổng hợp và tối ưu truy vấn dữ liệu lớn |
| Spark Structured Streaming | Xử lý dữ liệu luồng theo micro-batch với Windowing và Watermarking |
| Spark MLlib (ALS) | Mô hình lõi phân tán: Huấn luyện mô hình lọc cộng tác phân tán quy mô lớn (Matrix Factorization) |
| Scikit-learn (TruncatedSVD) | Mô hình cơ sở đơn máy (Single-node Baseline): Đối chứng hiệu năng, mức tiêu thụ tài nguyên và giới hạn mở rộng với Spark |
| Parquet (Snappy Compression) | Lưu trữ dữ liệu dạng cột, hỗ trợ Column Pruning, nén dung lượng và tối ưu đọc |
| Streamlit, Matplotlib | Xây dựng giao diện Dashboard trực quan hóa và báo cáo hiệu năng |
| Docker (tùy chọn) | Đóng gói và chuẩn hóa môi trường thực thi cụm Spark |

## **7. Các chức năng chính** 

1. **Tiền xử lý và Nạp dữ liệu (Data Ingestion & Cleaning):**  
   * Khởi tạo Spark Session với cấu hình phân bổ RAM/Executor tối ưu.
   * Đọc dữ liệu lớn với Schema định nghĩa tường minh (`StructType`), kiểm tra tính hợp lệ và khử trùng lặp.
   * Xử lý giá trị thiếu (missing values) theo nguyên tắc Drop có chọn lọc, trích xuất đặc trưng (năm phát hành, mảng thể loại).
   * Phân vùng lại dữ liệu (Repartitioning / Hash Partitioning) để cân bằng tải và hạn chế hiện tượng Data Skew.
2. **Huấn luyện mô hình và Đối sánh (Model Training & Benchmarking):**  
   * Phân chia dữ liệu huấn luyện (80%) và kiểm thử (20%) ngẫu nhiên có cố định seed.
   * Huấn luyện mô hình phân tán **ALS (Alternating Least Squares)** trong Spark MLlib trên tập dữ liệu sạch tầng Silver (`ratings_cleaned.parquet`) để trích xuất các vector ẩn (Latent factors) có số chiều $rank$ giữa người dùng và phim.
   * Xây dựng mô hình cơ sở đơn máy (Single-node Baseline) bằng **Scikit-learn TruncatedSVD** trên ma trận thưa (Sparse Matrix) để thực nghiệm đối chứng trực diện.
   * Áp dụng kỹ thuật điều chuẩn (Regularization) để tránh Overfitting.
3. **Đánh giá và Đối sánh Hiệu năng (Model Evaluation & Scalability Comparison):**  
   * Đo lường sai số dự đoán bằng chỉ số **RMSE (Root Mean Square Error)** trên tập kiểm thử cho cả hai mô hình.  
   * Đánh giá chất lượng danh sách đề xuất bằng các chỉ số xếp hạng **Precision@10** và **Recall@10**.
   * So sánh chi tiết về thời gian huấn luyện (Training Time), mức tiêu thụ bộ nhớ (RAM Footprint / Peak Memory) và khả năng mở rộng quy mô khi tăng dữ liệu từ mẫu thử nghiệm (1M dòng) lên toàn phần (25M dòng).
4. **Truy vấn kết quả và Lưu trữ tầng Gold (Serving & Recommendation Pipeline):**  
   * Lưu mô hình phân tán đã huấn luyện vào `models/als_best_model/` phục vụ suy luận linh hoạt.
   * Xây dựng pipeline tự động sinh danh sách *Top-10 bộ phim gợi ý* có điểm dự đoán cao nhất cho từng `userId`, loại trừ các bộ phim người dùng đã từng xem/đánh giá trong lịch sử.  
   * Ghi nhận kết quả gợi ý hoàn chỉnh (kèm tiêu đề, thể loại phim) vào tầng **Gold Zone** (`data/gold/user_recommendations_top10.parquet`) nhằm tối ưu tốc độ phản hồi cho Dashboard.
5. **Xử lý bài toán Người dùng mới (Interactive Cold-Start & Onboarding Demo):**
   * Giải quyết triệt để bài toán Cold-Start khi người dùng mới truy cập hệ thống chưa hề có lịch sử đánh giá trong tập dữ liệu.
   * Cung cấp giao diện Onboarding tương tác: Người dùng chọn các thể loại phim yêu thích và thể loại muốn loại trừ.
   * Hệ thống hiển thị 5–8 bộ phim tiêu biểu để người dùng đánh giá nhanh (Đã xem & Thích / Không thích / Chưa xem).
   * Ứng dụng kỹ thuật **ALS Folding-in** (chiếu nhanh vector tương tác của người dùng mới lên không gian tiềm ẩn của các bộ phim đã huấn luyện) để lập tức sinh danh sách Top-10 phim chuẩn gu trong vòng vài mili-giây mà không cần kích hoạt lại toàn bộ cụm Spark.
6. **Xử lý luồng thời gian thực (Real-time Stream Processing):**  
   * Mô phỏng luồng đánh giá phim liên tục và nạp vào Spark Structured Streaming.
   * Tính toán chỉ số thống kê tức thời: Danh sách phim thịnh hành (Trending Movies) và biến động điểm trung bình trong cửa sổ trượt.
7. **Trực quan hóa và đánh giá hiệu năng (Visualization & Performance Evaluation):**  
   * Xây dựng Dashboard Streamlit đa chức năng: Khám phá dữ liệu, tra cứu gợi ý theo `userId`, demo gợi ý tương tác cho người dùng mới, và biểu đồ luồng thời gian thực.
   * Thực nghiệm so sánh hiệu năng: Đánh giá dung lượng lưu trữ và thời gian truy vấn giữa CSV và Parquet; đo lường thời gian thực thi theo các cấu hình phân vùng (4, 8, 16 partitions).

## **8. Tiền xử lý, Quản lý Lưu trữ và Xử lý Luồng dữ liệu**

### **8.1. Thu thập và nạp dữ liệu**

Sử dụng bộ dữ liệu MovieLens 25M gồm: `ratings.csv` (25M bản ghi), `movies.csv` (62K bản ghi), `tags.csv` (1M bản ghi), `genome-scores.csv` (14M bản ghi) và `genome-tags.csv` (1.1K bản ghi).

* **Nạp dữ liệu với Schema tường minh:** Sử dụng PySpark đọc các tệp CSV với `schema` được định nghĩa bằng `StructType` thay vì dùng `inferSchema=True` để loại bỏ chi phí quét dữ liệu hai lần, đảm bảo kiểu dữ liệu chính xác tuyệt đối.
* **Kiểm tra sơ bộ:** Thống kê tổng số bản ghi, kiểm tra phân bố dữ liệu, tỷ lệ khuyết thiếu trên từng trường.

### **8.2. Yêu cầu kiểm tra tính hợp lệ và làm sạch dữ liệu**

Quy trình làm sạch dữ liệu đặt ra các yêu cầu kỹ thuật chi tiết đối với từng tệp:

* **Tệp `ratings.csv`:**
  * *Kiểm tra miền giá trị:* $userId > 0$, $movieId > 0$, $0.5 \le rating \le 5.0$ (bước nhảy 0.5), $timestamp$ phải thuộc khoảng thời gian hợp lệ (từ năm 1995 đến nay).
  * *Xử lý giá trị thiếu (Missing Values):* Trong bài toán hệ thống gợi ý, **tuyệt đối không áp dụng phương pháp điền khuyết (Imputation bằng trung bình/trung vị)** cho các trường khóa và điểm đánh giá vì sẽ làm sai lệch bản chất tương tác thực tế. Áp dụng quy tắc loại bỏ triệt để (Drop) các bản ghi có $userId$, $movieId$ hoặc $rating$ bị Null.
  * *Xử lý trùng lặp (Deduplication):* Với trường hợp một người dùng đánh giá cùng một bộ phim nhiều lần tại các mốc thời gian khác nhau, sử dụng Window Function (`row_number() over (partition by userId, movieId order by timestamp desc)`) để chỉ giữ lại lượt đánh giá mới nhất.
  * *Giảm độ thưa dữ liệu (Sparsity & Noise Filtering):* Lọc bỏ các bộ phim có ít hơn 5 lượt đánh giá trong toàn bộ tập dữ liệu nhằm loại bỏ nhiễu ngoại lai và giảm bớt bài toán Cold-Start.
* **Tệp `movies.csv`:**
  * *Khóa chính:* $movieId$ không được Null và không bị trùng lặp.
  * *Bóc tách năm phát hành (Feature Extraction):* Tên phim trong dữ liệu thường có định dạng `"Toy Story (1995)"`. Sử dụng PySpark Regex trích xuất năm sản xuất sang cột riêng `year` (kiểu Integer) và làm sạch cột `title`.
  * *Chuẩn hóa thể loại (`genres`):* Chuỗi phân tách bởi dấu gạch đứng (ví dụ: `"Action|Adventure|Sci-Fi"`) được chuyển đổi thành kiểu mảng `ArrayType(StringType())` bằng hàm `split()`. Loại bỏ các bản ghi mang nhãn `"(no genres listed)"`.
* **Tệp `tags.csv` & `genome-scores.csv`:**
  * Loại bỏ khoảng trắng thừa, chuyển văn bản thẻ về chữ thường (lowercase), loại bỏ thẻ rỗng.
  * Với `genome-scores.csv`, chỉ lọc giữ lại các bản ghi có độ liên quan $relevance \ge 0.5$ để giảm bớt hàng triệu bản ghi không mang ý nghĩa phân loại cao.

### **8.3. Chuẩn hóa và liên kết dữ liệu**

* **Kiểm tra tính nhất quán toàn vẹn:** Thực hiện Left Semi Join để đối chiếu mã phim giữa `ratings.csv` và `movies.csv`, loại bỏ các lượt đánh giá trỏ tới $movieId$ không tồn tại trong danh mục phim.
* **Liên kết dữ liệu phục vụ huấn luyện và phân tích:**
  * Bảng nạp vào ALS: Tinh gọn chỉ gồm 3 trường (userId, movieId, rating).
  * Bảng phục vụ phân tích: Kết hợp thông tin `movies.csv` (tiêu đề, năm, mảng thể loại) với thống kê đánh giá trung bình.

### **8.4. Yêu cầu và Đặc trưng của Dữ liệu lưu trữ (Medallion Architecture)**

Để đáp ứng yêu cầu xử lý dữ liệu lớn, nhóm tổ chức dữ liệu theo **Kiến trúc phân tầng Medallion (Lakehouse Pattern)**:

1. **Phân tầng dữ liệu:**
   * **Bronze Zone (Raw Data - `/data/bronze/`):** Chứa dữ liệu gốc dạng CSV tải về từ MovieLens. Đây là dữ liệu bất biến (immutable), phục vụ mục đích kiểm toán và tái tạo luồng xử lý khi cần.
   * **Silver Zone (Cleaned Data - `/data/silver/`):** Chứa các bảng dữ liệu sau khi làm sạch, ép kiểu chuẩn xác, bóc tách đặc trưng (năm, thể loại), lưu trữ dưới định dạng Parquet.
   * **Gold Zone (Curated Data - `/data/gold/`):** Chứa các bảng dữ liệu chuyên biệt phục vụ phân tích và học máy: ma trận nạp vào ALS, bảng thống kê phân bố thể loại, và bảng kết quả Top-10 gợi ý phim.

2. **Đặc trưng và lý do lựa chọn định dạng lưu trữ Parquet:**
   * **Lưu trữ dạng cột (Columnar Format):** Cho phép kỹ thuật **Column Pruning** – khi mô hình ALS cần đọc dữ liệu, Spark chỉ quét đúng 3 cột $userId, movieId, rating$ mà không cần đọc các cột khác, giảm thiểu I/O đĩa và băng thông mạng.
   * **Nén dữ liệu hiệu quả:** Sử dụng thuật toán nén **Snappy** mặc định của Parquet. Dung lượng 25M dòng đánh giá giảm từ ~1.1 GB (CSV) xuống còn ~250–300 MB (giảm hơn 70% dung lượng lưu trữ).
   * **Lưu trữ Metadata và Schema:** Định dạng Parquet lưu sẵn cấu trúc dữ liệu và thống kê (min, max) trong footer của file, hỗ trợ kỹ thuật **Predicate Pushdown** (lọc dữ liệu trực tiếp ở tầng lưu trữ trước khi nạp vào RAM Spark).

3. **Yêu cầu về Chiến lược Phân vùng dữ liệu (Partitioning Strategy):**
   * **Phân vùng lưu trữ (Storage Partitioning):** Khi ghi dữ liệu Silver phục vụ truy vấn thời gian, áp dụng `partitionBy("rating_year")` để tối ưu hóa các truy vấn báo cáo theo từng năm mà không cần quét toàn bộ tập dữ liệu.
   * **Phân vùng tính toán phân tán (Compute Partitioning):** Khi nạp dữ liệu vào huấn luyện ALS, dữ liệu lớn dễ gặp hiện tượng **Data Skew** (lệch dữ liệu do các bộ phim nổi tiếng tập trung quá nhiều rating, dẫn đến một số Task chạy chậm gây tắc nghẽn cụm - stragglers). Nhóm áp dụng phương pháp `repartition(N)` hoặc phân vùng băm (Hash Partitioning) theo $userId$ với số phân vùng được tính toán tối ưu: $N_{partitions} = 2 	ext{ đến } 4 	imes 	ext{tổng số cores}$ của hệ thống.

### **8.5. Thiết kế và Xác định Bài toán Xử lý Luồng (Spark Structured Streaming)**

* **Làm rõ ranh giới kỹ thuật:** Thuật toán ALS trong Spark MLlib là mô hình **huấn luyện theo lô (Batch Processing)**, không hỗ trợ cập nhật trọng số trực tuyến (Online Learning) theo từng bản ghi. Do đó, bài toán xử lý luồng được xác định tập trung vào **Speed Layer** phục vụ cập nhật chỉ số tương tác và thống kê thời gian thực:
* **Mô phỏng nguồn dữ liệu (Stream Source):**
  * Xây dựng script Python độc lập (`data_producer.py`) trích xuất các lượt đánh giá mới nhất và định kỳ mỗi 3–5 giây ghi một file CSV nhỏ chứa vài trăm bản ghi vào thư mục nguồn `streaming_inputs/`.
* **Cơ chế xử lý vi phân lô (Micro-batch Processing):**
  * Spark Structured Streaming đọc dữ liệu với `spark.readStream` kèm Schema định sẵn từ thư mục giám sát.
  * Chu kỳ kích hoạt: `trigger(processingTime='3 seconds')`.
* **Nghiệp vụ xử lý luồng:**
  * *Kiểm tra hợp lệ luồng (Stream Validation):* Lọc nhanh các bản ghi lỗi format trước khi đưa vào luồng tính toán.
  * *Xử lý trễ thời gian (Watermarking):* Áp dụng `withWatermark("timestamp", "10 minutes")` để xử lý các sự kiện đến muộn trong vòng 10 phút.
  * *Tính toán thống kê cửa sổ trượt (Sliding Window Aggregation):* Thống kê số lượt đánh giá và điểm trung bình trong cửa sổ 10 phút, trượt mỗi 2 phút (`window(col("timestamp"), "10 minutes", "2 minutes")`) để xác định **Top phim thịnh hành thời gian thực (Real-time Trending Movies)**.
* **Đích dữ liệu (Sink) & Hiển thị:**
  * Dữ liệu luồng được cập nhật liên tục ra `Memory Sink` hoặc append vào khu vực Gold streaming table.
  * Ứng dụng Streamlit tự động làm mới giao diện để hiển thị biểu đồ xu hướng phim đang được cộng đồng quan tâm nhất theo thời gian thực.

## **9\. Huấn luyện và đánh giá mô hình**

### **9.1. Nguyên lý xây dựng mô hình**

Sử dụng thuật toán **ALS (Alternating Least Squares)** trong Apache Spark MLlib làm mô hình học máy phân tán cốt lõi để xây dựng hệ thống gợi ý phim cá nhân hóa.

ALS dựa trên lịch sử đánh giá của người dùng để tìm ra các đặc trưng tiềm ẩn (Latent factors) của người dùng ($U$) và phim ($V$). Từ đó, mô hình tái tạo ma trận tương tác $R \approx U \cdot V^T$ và dự đoán mức độ yêu thích của người dùng đối với những bộ phim chưa được đánh giá.

Dữ liệu đầu vào lấy trực tiếp từ tầng **Silver** (`ratings_cleaned.parquet`), gồm ba trường chính: `userId`, `movieId` và `rating`.

### **9.2. Chuẩn bị dữ liệu huấn luyện**

* **Nguồn dữ liệu:** Dữ liệu đánh giá sạch sau tiền xử lý tại `data/silver/ratings_cleaned.parquet`.  
* **Chia dữ liệu:** Phân chia ngẫu nhiên thành 80% tập huấn luyện (training set) và 20% tập kiểm thử (test set), sử dụng cùng một `seed = 42` cố định để bảo đảm tính khách quan và khả năng tái lập kết quả.  
* **Quy mô dữ liệu:** Trên toàn bộ 25 triệu lượt đánh giá, khoảng 20 triệu lượt dùng để huấn luyện và 5 triệu lượt dùng để kiểm thử.  
* **Nguyên tắc:** Tập kiểm thử hoàn toàn độc lập, không được dùng để huấn luyện hoặc điều chỉnh siêu tham số.

### **9.3. Huấn luyện mô hình phân tán Apache Spark ALS**

Sử dụng Spark MLlib để huấn luyện mô hình với các siêu tham số chính:

* `rank`: Số chiều của vector đặc trưng tiềm ẩn; thử nghiệm các giá trị 10, 20 và 50.  
* `maxIter`: Số vòng lặp tối ưu hóa xen kẽ; thiết lập 10–15 vòng.  
* `regParam`: Hệ số điều chuẩn ($L_2$ regularization) nhằm hạn chế Overfitting; thử nghiệm 0.01, 0.1 và 1.0.  
* `coldStartStrategy`: Thiết lập `drop` trong giai đoạn đánh giá để loại bỏ các trường hợp người dùng hoặc bộ phim mới chưa từng xuất hiện trong tập huấn luyện (tránh sinh ra giá trị NaN).

### **9.4. Mô hình cơ sở đơn máy (Single-node Baseline): Scikit-learn TruncatedSVD**

Để làm nổi bật giá trị cốt lõi của môn học về **Lưu trữ và Xử lý Dữ liệu lớn**, nhóm xây dựng một mô hình cơ sở đơn máy sử dụng thuật toán **TruncatedSVD (Singular Value Decomposition)** từ thư viện Scikit-learn.

* **Tính tương đồng toán học:** Cả Spark ALS và Scikit-learn TruncatedSVD đều thuộc họ phương pháp **Phân rã ma trận (Matrix Factorization)**, tìm kiếm phép chiếu chiều thấp biểu diễn ma trận thưa User-Movie. Việc chọn SVD làm baseline bảo đảm sự công bằng tuyệt đối về mặt thuật toán.
* **Cơ chế triển khai:** Chuyển đổi tập đánh giá sang ma trận thưa dạng nén theo dòng (`scipy.sparse.csr_matrix`), huấn luyện phân rã với cùng số chiều $n\_components = rank = 20$.
* **Đặc thù kiến trúc đơn máy:** Scikit-learn đòi hỏi toàn bộ ma trận phải nằm trọn trong RAM của một tiến trình Python duy nhất trên một máy tính, không có cơ chế phân vùng tự động hay chịu lỗi phân tán.

### **9.5. Thực nghiệm Đối sánh Mô hình (Spark ALS vs Scikit-learn SVD)**

Nhóm thiết kế kịch bản thực nghiệm đối sánh trực diện trên 2 quy mô dữ liệu:

1. **Kịch bản 1 - Thực nghiệm trên tập mẫu (1.000.000 dòng):** Cả hai mô hình cùng chạy trên tập dữ liệu 1M dòng để đo đạc công bằng các chỉ số:
   * **Sai số dự đoán (RMSE):** Đánh giá trên cùng tập kiểm thử 20%.
   * **Thời gian huấn luyện (Training Time):** Đo bằng giây từ khi bắt đầu fit mô hình đến khi hoàn tất.
   * **Mức tiêu thụ bộ nhớ (Memory Footprint / Peak RAM):** Đo lường lượng RAM chiếm dụng tối đa trong quá trình nạp và phân rã ma trận.
2. **Kịch bản 2 - Khả năng mở rộng quy mô toàn phần (25.000.000 dòng):**
   * Huấn luyện Spark MLlib ALS trên toàn bộ dữ liệu 25M dòng tận dụng 24 luồng CPU và 10 GB bộ nhớ được cấp phát.
   * Phân tích và chứng minh hiện tượng **Out Of Memory (OOM)** / nghẽn RAM của Scikit-learn khi kích thước ma trận $162.000 \times 62.000$ vượt quá khả năng xử lý trên một máy tính thông thường, qua đó khẳng định tính tất yếu của kiến trúc phân tán Apache Spark trong bài toán Big Data.

| Tiêu chí Đối sánh | Scikit-learn TruncatedSVD (Đơn máy) | Apache Spark MLlib ALS (Hệ phân tán) |
| :--- | :--- | :--- |
| **Bản chất thuật toán** | Phân rã giá trị suy biến (Matrix Factorization) | Phân rã ma trận xen kẽ bình phương tối thiểu (ALS) |
| **Môi trường thực thi** | Đơn tiến trình Python trên 1 máy tính vật lý | Phân tán trên cụm máy / Đa phân vùng (24 partitions) |
| **Cấu trúc lưu trữ trong RAM** | Gom toàn bộ `csr_matrix` vào RAM đơn máy | Phân mảnh ma trận theo các khối RDD (BlockMatrix) |
| **Giới hạn quy mô (Scalability)** | Bị giới hạn cứng bởi RAM vật lý; dễ gặp lỗi OOM | Mở rộng ngang (Horizontal Scaling) theo số Worker |
| **Khả năng chịu lỗi (Fault Tolerance)** | Không có (tiến trình dừng nếu gặp sự cố) | Tự động phục hồi phân vùng bị lỗi qua Spark DAG |
| **Kết quả kỳ vọng trên 25M dòng** | Quá tải bộ nhớ / Thời gian dựng ma trận cực lớn | Xử lý hoàn tất ổn định trong vài phút |

### **9.6. Lưu trữ Mô hình và Xuất dữ liệu Tầng Gold (Serving Layer)**

* **Lưu trữ mô hình:** Mô hình Spark ALS tốt nhất được lưu tuần tự hóa vào thư mục `models/als_best_model/` để tái sử dụng trong suy luận trực tiếp.
* **Tạo danh sách Top-10 gợi ý ngoại tuyến (Offline Batch Recommendations):** Sử dụng hàm `recommendForAllUsers(10)` của Spark MLlib để tính toán trước 10 bộ phim phù hợp nhất cho từng người dùng, tự động loại trừ các phim người dùng đã đánh giá.
* **Liên kết dữ liệu và ghi tầng Gold:** Join kết quả với `movies_cleaned.parquet` để bổ sung tiêu đề và thể loại, sau đó ghi dưới định dạng Snappy Parquet tại `data/gold/user_recommendations_top10.parquet`. Khi đó Dashboard có thể tra cứu kết quả tức thì ($O(1)$) mà không cần tính toán lại.
* **Giải pháp cho bài toán Cold-Start (Kỹ thuật ALS Folding-in cho Người dùng mới):**
  * Đối với người dùng mới toanh chưa có mã định danh trong tập dữ liệu 25M, hệ thống không kích hoạt lại quy trình huấn luyện toàn phần trên cụm Spark (vì gây lãng phí tài nguyên tính toán lớn).
  * Thay vào đó, hệ thống trích xuất ma trận đặc trưng các bộ phim (**Item Factors** $V \in \mathbb{R}^{M \times k}$) đã được huấn luyện sẵn từ mô hình ALS.
  * Sau khi người dùng mới chấm điểm nhanh $m$ bộ phim ($m \ll M$) trong bước Onboarding, vector sở thích của người dùng mới $u_{new} \in \mathbb{R}^k$ được giải nghiệm tối ưu trực tiếp bằng bài toán bình phương tối thiểu có điều chuẩn (Ridge Regression):
    $$u_{new} = \left(V_{sub}^T V_{sub} + \lambda I\right)^{-1} V_{sub}^T r_{new}$$
  * Phép tính này chỉ mất dưới $0.01$ giây trên CPU thông thường và lập tức tính được điểm dự đoán $\hat{r} = u_{new} \cdot V^T$, sinh ra danh sách Top-10 phim được cá nhân hóa hoàn toàn theo gu thưởng thức của người dùng mới.

### **9.7. Đánh giá chất lượng mô hình**

Sử dụng các chỉ số:
* **RMSE (Root Mean Squared Error):** Đo sai số giữa điểm đánh giá dự đoán và điểm đánh giá thực tế trên tập kiểm thử.  
* **Precision@10 & Recall@10:** Đo lường độ chuẩn xác và độ bao phủ của danh sách 10 phim được đề xuất.  
* **Thời gian huấn luyện và thời gian truy vấn:** Đo lường hiệu năng thực thi của hệ thống.

## **10\. Xây dựng dashboard trực quan hóa kết quả**

### **10.1. Công nghệ sử dụng**

Sử dụng Streamlit để xây dựng giao diện và Matplotlib để trực quan hóa dữ liệu, kết quả đánh giá mô hình.

### **10.2. Các chức năng chính**

* **Tổng quan dữ liệu (EDA):** Hiển thị số lượng người dùng, phim, lượt đánh giá và phân bố điểm đánh giá qua các năm.  
* **Đối sánh mô hình:** Trình bày bảng chỉ số RMSE, thời gian huấn luyện, mức tiêu thụ RAM giữa Spark ALS và Scikit-learn SVD.  
* **Tra cứu gợi ý phim:** Cho phép chọn người dùng có sẵn trong tập dữ liệu và hiển thị danh sách Top-10 phim được đề xuất từ tầng Gold.  
* **Demo Cá nhân hóa Người dùng mới (Interactive Cold-Start Onboarding):**
  * Cung cấp giao diện tương tác trực tiếp: Cho phép người dùng tick chọn các thể loại phim yêu thích và thể loại muốn loại trừ.
  * Hiển thị danh sách các phim nổi bật thuộc thể loại đã chọn để người dùng đánh giá nhanh (Đã xem & Thích / Không thích / Chưa xem).
  * Tính toán tức thì vector sở thích bằng thuật toán Folding-in và xuất danh sách Top-10 phim đề xuất trực quan.
* **So sánh hiệu năng Big Data:** Biểu diễn thời gian xử lý giữa các cấu hình phân vùng dữ liệu và định dạng CSV vs Parquet.
* **Giám sát luồng thời gian thực:** Trực quan hóa danh sách phim thịnh hành (Trending Movies) từ Spark Structured Streaming.

### **10.3. Kết quả đầu ra**

Dashboard cho phép theo dõi dữ liệu, đánh giá chất lượng mô hình, xem danh sách phim gợi ý và so sánh hiệu năng xử lý của Apache Spark.

## **11\. Kế hoạch thực hiện dự kiến**

| Tuần | Nội dung công việc | Kết quả dự kiến |
| :---: | ----- | ----- |
| 1 | Tìm hiểu đề tài, Apache Spark, Spark SQL và thuật toán ALS; xác định yêu cầu và kiến trúc hệ thống. | Xác định phạm vi, yêu cầu và kiến trúc hệ thống. |
| 2 | Thu thập dữ liệu MovieLens 25M, cài đặt môi trường và thử nghiệm PySpark. | Có dữ liệu và môi trường xử lý hoạt động. |
| 3–4 | Nạp, kiểm tra, làm sạch, chuẩn hóa dữ liệu; phân tích dữ liệu bằng Spark SQL và lưu trữ dưới dạng Parquet. | Hoàn thành quy trình tiền xử lý và có dữ liệu phục vụ huấn luyện. |
| 5–6 | Xây dựng, huấn luyện mô hình ALS phân tán và đối sánh với mô hình cơ sở Scikit-learn TruncatedSVD; đánh giá bằng RMSE, thời gian và bộ nhớ. | Có mô hình tối ưu, bảng đối sánh hiệu năng hệ phân tán và kết quả chất lượng. |
| 7 | Xây dựng chức năng gợi ý Top-10 phim cho người dùng và module giải quyết Cold-Start tương tác (Onboarding & ALS Folding-in). | Hoàn thành chức năng gợi ý cá nhân hóa và thuật toán suy luận nhanh cho người dùng mới. |
| 8 | Thử nghiệm hiệu năng Spark; so sánh CSV với Parquet và các cấu hình phân vùng dữ liệu. | Có số liệu và biểu đồ so sánh hiệu năng. |
| 9 | Xây dựng dashboard hoàn chỉnh bằng Streamlit (EDA, Benchmark, Đối sánh ALS vs SVD, Demo tương tác Cold-Start, Streaming) và kiểm thử hệ thống. | Hoàn thành ứng dụng trực quan đa chức năng phục vụ thuyết trình. |
| 10 | Tổng hợp kết quả thực nghiệm, hoàn thiện báo cáo, mã nguồn và slide thuyết trình. | Hoàn thành project và chuẩn bị thuyết trình. |

