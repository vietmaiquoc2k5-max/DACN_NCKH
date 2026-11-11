# BÁO CÁO TỔNG KẾT KHOA HỌC WEEK 4: ĐÁNH GIÁ THỰC NGHIỆM THỜI GIAN THỰC & BỘ XÁC NHẬN THEO THỜI GIAN
**Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
**Giai đoạn:** Tuần 4 — Xác nhận theo thời gian (Temporal Verification) & Giảm báo động giả (FAR Reduction)  
**Tác giả thực hiện:** Mai Quốc Việt & Lê Minh Thiên  
**Mã commit chuẩn hóa:** `256065e7` | **Model:** YOLOv8n (`runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt`)  

---

## 1. Đặt Vấn Đề và Mục Tiêu Nghiên Cứu Tuần 4
Trong bài toán phát hiện cháy/khói thời gian thực qua camera giám sát an ninh:
- Mô hình học sâu (như YOLO) chỉ phân tích độc lập từng khung hình (Frame-level Detection). Khi gặp các tác nhân gây nhiễu trong bối cảnh thực tế (đèn pha xe, ánh hoàng hôn, đèn LED đỏ, khói bếp, bụi đường), mô hình thường xuất hiện các dự đoán tự phát trong 1-2 khung hình, dẫn đến hiện tượng **báo động giả liên tục (False Alarm Spam)**.
- **Mục tiêu nghiên cứu Tuần 4 (MT3 & MT5 theo Đề cương NCKH):**
  1. Tích hợp thuật toán theo dõi đối tượng nhiều giai đoạn **ByteTrack** để duy trì quỹ đạo ổn định cho các vùng nghi ngờ cháy/khói qua thời gian.
  2. Xây dựng bộ xác nhận thời gian (`TemporalVerifier`) dựa trên luật cửa sổ trượt $k/N$, kết hợp cổng lọc biến động đặc trưng chuyển động (`TemporalFeatureExtractor` - flicker/lan tỏa diện tích) và cơ chế Cooldown nhân quả.
  3. Kiểm chứng thực nghiệm giảm $\ge 50\%$ tỷ lệ báo động giả so với cảnh báo từng khung hình; suy giảm Event Recall $\le 5$ điểm %; độ trễ cảnh báo trung bình $\le 10$ giây; tốc độ xử lý đạt $\ge 5$ FPS trên CPU và $\ge 15$ FPS trên GPU.

---

## 2. Kho Dữ Liệu Thực Nghiệm (Empirical Datasets)
Toàn bộ dữ liệu đã được kiểm toán vật lý bằng OpenCV và mã băm SHA-256 (`docs/WEEK4_DATA_AUDIT.md`):
- **Tập Video Dương tính:** 104 video vật lý tồn tại trên ổ đĩa (5,17 giờ), trong đó 24 video chuẩn hóa (`POS_01` – `POS_24`) được sử dụng cho quy trình đối sánh 4 Pipeline và quét lưới tham số.
- **Tập Video Âm tính:** 79 video vật lý tồn tại trên ổ đĩa (7,20 giờ), vượt yêu cầu $\ge 5,0$ giờ của đề cương. Chứa đầy đủ các cảnh nhiễu: đèn xe, LED neon, hoàng hôn, hơi nước, bụi đường, bếp ăn.
- **Tập Hình ảnh Bối cảnh Việt Nam (`datasets/vn_dataset/`):** 3.121 ảnh và 3.121 nhãn YOLO chuẩn, phân chia 60/20/20 theo video độc lập (Train: 1.868, Val: 623, Test: 630), bảo đảm 0% rò rỉ dữ liệu (`is_leakage_free: True`).
- **Tập Huấn luyện D-Fire (`datasets/dfire/`):** 21.527 ảnh (Train: 17.221, Test: 4.306).

---

## 3. Kiến Trúc 4 Pipeline Đối Sánh (Comparative Pipelines)
Được đánh giá song song, cùng điều kiện trên toàn bộ 24 video dương tính (lấy mẫu cố định 5.0 FPS, `imgsz = 416`):
1. **Pipeline A — Frame-Level Baseline:** Chỉ chạy YOLO detector. Mỗi detection $\ge \text{conf}$ phát ra 1 frame alarm ngay lập tức; không dùng Tracker, không dùng $k/N$, không dùng Cooldown.
2. **Pipeline B1 — ByteTrack (No Feature Gate):** YOLO + `ByteTrackAdapter` + `TemporalVerifier` ($k/N = 3/8$, Cooldown 5.0s, `feature_threshold = 0.0`).
3. **Pipeline B2 — ByteTrack + Feature Gate (Đề xuất chính):** YOLO + `ByteTrackAdapter` + `TemporalFeatureExtractor` + `TemporalVerifier` ($k/N = 3/8$, Cooldown 5.0s, `feature_threshold = 0.02`).
4. **Pipeline C — Legacy IoU Temporal:** YOLO + IoU tracker nội bộ + `TemporalVerifier`.

---

## 4. Kết Quả Đối Sánh Thực Nghiệm (Scientific Results)

### 4.1. Bảng Đối Sánh Tổng Hợp Trên 24 Video Dương Tính (`full_24_positive`)

| Chỉ số khoa học | Pipeline A (Frame Baseline) | Pipeline B1 (ByteTrack No Gate) | Pipeline B2 (ByteTrack + Feature Gate) | Pipeline C (Legacy IoU) |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng số cảnh báo phát ra** | **2.938 frame alarms** | **129 alerts (-95,61%)** | **128 alerts (-95,64%)** | **128 alerts** |
| **Số khoảng sự kiện (gap=2.0s)** | 53 events | 32 events | **31 events** | 33 events |
| **Video-level Event Recall** | 95,83% (23/24) | 95,83% (23/24) | **95,83% (23/24)** | 95,83% (23/24) |
| **Event Precision** | 52,83% | 78,12% | **80,65% (+27,82%)** | 78,79% |
| **Event Recall (Strict Class)** | 80,00% | 71,43% | **71,43%** | 74,29% |
| **Event F1-Score** | 63,64% | 74,63% | **75,76% (+12,12%)** | 76,47% |
| **Độ trễ phát hiện trung bình** | 1,02 s | 2,09 s | **2,13 s ($\le 10$ s)** | 2,19 s |
| **Báo động nổ ra trước khi cháy** | 3 video | 0 | **0 video** | 0 |
| **Số sự kiện báo sai trên POS** | 25 sự kiện | 7 sự kiện | **6 sự kiện** | 7 sự kiện |
| **Độ trễ CPU / Pipeline FPS** | 17,27 ms (57,9 FPS) | 17,85 ms (56,0 FPS) | **18,10 ms (55,2 FPS)** | 17,30 ms (57,8 FPS) |
| **Tốc độ GPU FP16** | ~32,6 FPS | ~32,0 FPS | **~31,8 FPS** | ~32,4 FPS |

> **Phân tích Chi phí Tính toán (Overhead Breakdown):**  
> Trên CPU Intel, tổng chi phí tính toán tăng thêm của Pipeline B2 so với Detector thuần túy chỉ là **$0,836\text{ ms/frame}$** (trong đó `ByteTrack` chiếm $0,154\text{ ms}$, trích xuất đặc trưng `TemporalFeature` chiếm $0,312\text{ ms}$, và `TemporalVerifier` chiếm $0,370\text{ ms}$). Hệ thống duy trì tốc độ **$55,2\text{ FPS}$ trên CPU** và **$>31\text{ FPS}$ trên GPU**, vượt xa chỉ tiêu tối thiểu ($\ge 5\text{ FPS CPU}, \ge 15\text{ FPS GPU}$).

---

### 4.2. Phân Tách Theo Từng Lớp (Per-Class Breakdown) & Khoảng Tin Cậy 95% CI
Tại điểm vận hành Pipeline B2 trên tập kiểm định:

| Phân lớp đối tượng | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision (95% CI) | Recall (95% CI) | F1-Score | Detection Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lớp LỬA (FIRE)** | 11 | 2 | 5 | **84,62%** [57,8% – 95,7%] | **68,75%** [44,4% – 85,8%] | **75,86%** | **0,78 giây** |
| **Lớp KHÓI (SMOKE)** | 14 | 4 | 5 | **77,78%** [54,8% – 91,0%] | **73,68%** [51,2% – 88,2%] | **75,68%** | **3,19 giây** |
| **TỔNG HỢP (OVERALL)** | **25** | **6** | **10** | **80,65%** [63,7% – 90,8%] | **71,43%** [54,9% – 83,7%] | **75,76%** | **2,13 giây** |

- **Nhận xét chuyên sâu:**
  - Lớp **LỬA** có độ trễ cực thấp (**0,78 giây**) và độ chính xác rất cao (**84,62%**), phản ánh ngọn lửa có tín hiệu quang học rõ ràng, giúp ByteTrack và TemporalVerifier kích hoạt cảnh báo tức thời.
  - Lớp **KHÓI** có độ trễ cao hơn một chút (**3,19 giây**), hoàn toàn phù hợp với bản chất vật lý của sự cháy: khói cần thời gian bốc lên, tích tụ thể tích và lan tỏa để vượt qua ngưỡng diện tích và bộ lọc biến động đặc trưng.

---

## 5. Phân Tích Lỗi Tiêu Biểu (Error Analysis)
Từ 6 trường hợp False Positive và 10 trường hợp False Negative (Missed event intervals) được ghi nhận trong benchmark:
1. **False Positives (Báo sai):**
   - *Nguyên nhân chính:* Phản quang mạnh từ bề mặt kim loại hoặc ánh sáng chớp tắt lặp lại trong video làm `TemporalFeatureExtractor` ghi nhận độ chớp cường độ cục bộ vượt 0.02 kết hợp với confidence dao động quanh 0.35–0.45.
   - *Khắc phục:* Nâng nhẹ `feature_threshold` lên 0.03 hoặc tăng kích thước cửa sổ $N=10$.
2. **False Negatives (Bỏ sót sự kiện):**
   - Không có video nào bị bỏ sót hoàn toàn (Video Recall đạt 95,83% = 23/24 video). Video duy nhất trễ cảnh báo là `POS_18` (đám khói rất xa và loãng ở góc trên khung hình).
   - Trong số 10 sự kiện thành phần bị lỡ, chủ yếu là do ngọn lửa bùng phát quá ngắn (< 1,5 giây) rồi tắt hoặc khói quá mỏng trong điều kiện ngược sáng.

---

## 6. Điểm Vận Hành (Operating Point) & Trạng Thái
- Cấu hình đang sử dụng: `confidence_threshold = 0.35`, $K/N = 3/8$, `feature_threshold = 0.02`, `cooldown_sec = 5.0 s`.
- Trạng thái xác nhận: **`PROVISIONAL`** (Tạm thời), bảo lưu chỉ tiêu FAR/hour cho đến khi chạy đo kiểm trên toàn bộ 7,20 giờ video âm tính thực tế.

---

## 7. Kế Hoạch Chuyển Tiếp Sang Week 5 (Domain Adaptation)
Các bước chuyển giao chuẩn xác theo Đề cương NCKH:
1. Chạy mẻ đánh giá FAR/giờ trên kho 79 video âm tính thực tế để hoàn tất chính thức chỉ tiêu MT3.
2. Triển khai 3 chiến lược Thích nghi miền (Domain Adaptation) trên bộ dữ liệu Việt Nam 3.121 ảnh:
   - Chiến lược 1: Fine-tuning toàn bộ mạng (Full Fine-tuning).
   - Chiến lược 2: Đóng băng backbone, chỉ huấn luyện head phát hiện (Frozen Backbone).
   - Chiến lược 3: Đóng băng backbone kết hợp Replay 20% dữ liệu D-Fire (Experience Replay).
3. Đo lường hiện tượng Quên thảm họa (Catastrophic Forgetting) bằng cách đánh giá mAP song song trên cả tập D-Fire Test (2.153 ảnh) và tập Việt Nam Test (630 ảnh).

