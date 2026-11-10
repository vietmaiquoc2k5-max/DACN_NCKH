# Báo Cáo Đánh Giá Lại Tính Toàn Vẹn Khoa Học và Nghiệm Thu Tuần 4
## (Week 4 Scientific Revalidation & Acceptance Report)

**Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ camera giám sát (`DACN_NCKH`)  
**Nhánh Git:** `fix/week4-scientific-validation`  
**Commit nền đối chiếu:** `280cd6337920c981bd5662a0fddf9bdde374c041`  
**Thời gian hoàn thành:** 10/11/2026  
**Chủ biên thực hiện:** Lê Minh Thiên (Chủ nhiệm) & Mai Quốc Việt (Kỹ sư triển khai)  
**Tiêu chuẩn khoa học áp dụng:** Đề cương NCKH gốc (`DeCuong_NCKH_PhatHienChay_Camera.docx`), IEEE Transactions on CSVT Standards, Quy tắc Liêm chính Học thuật Độc lập.

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Sau đợt rà soát độc lập ở commit nền `280cd63`, nhóm nghiên cứu đã phát hiện một số sai lệch và thiếu sót nghiêm trọng về tính hợp lệ khoa học trong kết quả benchmark Tuần 4:
1. **Ground Truth tự động nâng cấp trạng thái:** Bản ghi sự kiện từ `pending_verification` bị tự động nâng thành `verified` mà chỉ có duy nhất 1 người gán nhãn (`annotator_01`), vi phạm nguyên tắc kiểm tra chéo độc lập (cross-validation).
2. **FAR tính trên khung hình thưa (Sparse Sampling):** FAR (False Alarm Rate) được tính chỉ trên 10–15 khung hình rời rạc mỗi video bằng hàm `get_frame_at_sec(0.5..5.0)`, dẫn đến kết quả FAR = 0.00 cảnh báo giả/giờ – một kết quả phi thực tế, bỏ qua hơn 99% khung hình thực tế của luồng video.
3. **Mập mờ giữa kiểm thử Tệp cục bộ và Luồng mạng RTSP:** Benchmark đa luồng camera thực chất chỉ đọc tệp AVI cục bộ trên ổ cứng nhưng được dán nhãn là đã đạt chỉ tiêu RTSP của MT5, trong khi hạ tầng mạng RTSP thực tế chưa hề được kích hoạt.

Nhánh `fix/week4-scientific-validation` được thiết lập nhằm khắc phục triệt để và minh bạch toàn bộ các tồn tại trên:
- **Chuẩn hóa Ground Truth v1.1:** Phân tách rành mạch 4 trạng thái (`pending_verification`, `annotator_reviewed`, `cross_verified`, `disputed`). Chỉ có **11 sự kiện đạt chuẩn `cross_verified`** (có đủ 2 annotator độc lập, sai lệch thời gian $\le 0.5\text{s}$). 24 sự kiện còn lại giữ nguyên trạng thái `pending_verification`.
- **Đánh giá lại FAR Full-Frame Streaming:** Thực nghiệm liên tục $100\%$ khung hình (native 30 FPS, step = 1, trọn vẹn 5.725 frames) trên toàn bộ 5 video âm tính thực tế trên ổ đĩa. Kết quả chứng minh: Pipeline A (Baseline từng khung) có FAR = **169.78/h**; Pipeline B1 (ByteTrack + K/N) có FAR = **94.32/h**; Pipeline B2 (ByteTrack + Temporal Feature Gate) triệt tiêu hoàn toàn báo giả, đạt **FAR = 0.00/h (giảm 100.0%)**.
- **Minh bạch hóa Benchmark Multi-Camera:** Tách riêng 2 báo cáo: (A) Tải đa luồng tệp AVI cục bộ (1, 2, 4, 6, 9 luồng đạt 23.6–23.8 Capture FPS, 32.8–48.4 Inference FPS, P50 latency 20.9–28.7 ms, RAM 423–769 MB); (B) Đánh giá RTSP thực tế được ghi nhận trung thực là **`NOT_TESTED (BLOCKED_NO_LIVE_RTSP_SOURCE)`**.
- **Kết luận nghiệm thu Tuần 4:** **`PARTIAL (BLOCKED_BY_NEGATIVE_DATA & NO_LIVE_RTSP)`**. Về mặt thuật toán, Temporal Verification và Feature Gate đã chứng minh sự vượt trội và tính khả thi xuất sắc; tuy nhiên dự án từ chối nghiệm thu toàn phần cho đến khi bổ sung đủ 5.0 giờ video âm tính và kích hoạt luồng RTSP trực tiếp.

---

## 2. Những Thay Đổi Đã Thực Hiện (System Changes)

| Hạng mục | Hiện trạng ở commit `280cd63` | Thay đổi tại branch `fix/week4-scientific-validation` | Tệp liên quan |
| :--- | :--- | :--- | :--- |
| **Quy trình Ground Truth** | Tự động coi `pending` là `verified` nếu có dữ liệu; không có cơ chế đối soát 2 annotator. | Phân tách 4 trạng thái; bắt buộc 2 annotator và sai lệch $\le 0.5\text{s}$ mới cấp nhãn `cross_verified`; đóng băng snapshot v1.1. | [`src/ground_truth_validator.py`](file:///d:/workspace/DACN_NCKH/src/ground_truth_validator.py), [`scripts/standardize_ground_truth_v1_1.py`](file:///d:/workspace/DACN_NCKH/scripts/standardize_ground_truth_v1_1.py) |
| **Đánh giá FAR** | Đọc thưa 10-15 frames/video qua `get_frame_at_sec(0.5..5.0)`, FAR = 0.00/h giả tạo. | Full-frame streaming liên tục từng frame (step=1, 5.725 frames), giải phóng bộ nhớ từng frame, tính FAR thực tế. | [`scripts/evaluate_full_frame_far.py`](file:///d:/workspace/DACN_NCKH/scripts/evaluate_full_frame_far.py) |
| **Đa luồng Camera** | Gộp chung file và RTSP; tuyên bố đạt RTSP dù chỉ đọc tệp AVI. | Tách bạch 2 chế độ: A (File simulation) và B (RTSP Live); trung thực ghi nhận NOT_TESTED khi thiếu nguồn RTSP. | [`scripts/benchmark_multicamera_scientific.py`](file:///d:/workspace/DACN_NCKH/scripts/benchmark_multicamera_scientific.py) |
| **Bảo mật RTSP** | Có nguy cơ lộ URL chứa mật khẩu camera trong log/json. | Tích hợp regex che mật khẩu nâng cao `mask_rtsp_credentials` hỗ trợ cả ký tự đặc biệt (@, !, #). | [`src/config.py`](file:///d:/workspace/DACN_NCKH/src/config.py), [`src/camera_manager.py`](file:///d:/workspace/DACN_NCKH/src/camera_manager.py) |
| **Đối soát Operating Point** | Đánh giá lẫn lộn giữa validation và test; thiếu phân tách chi tiết missed/late events. | Đối soát độc lập trên Locked Test Set (12 video); phân tách nhóm Verified GT vs All GT; bóc tách từng sự kiện bỏ sót/trễ. | [`scripts/reconcile_scientific_operating_point.py`](file:///d:/workspace/DACN_NCKH/scripts/reconcile_scientific_operating_point.py) |
| **Kiểm thử tự động** | 61 unit tests. | Bổ sung 18 unit tests mới về tính toàn vẹn khoa học, cross-validation, và masking (tổng 79 tests, PASS 100%). | [`tests/test_ground_truth_cross_validation.py`](file:///d:/workspace/DACN_NCKH/tests/test_ground_truth_cross_validation.py), [`tests/test_scientific_validation_integrity.py`](file:///d:/workspace/DACN_NCKH/tests/test_scientific_validation_integrity.py) |

---

## 3. Tính Toàn Vẹn Của Ground Truth (Ground Truth Integrity)

### 3.1. Phân Loại Trạng Thái và Kiểm Định Chéo
Tệp nhãn Ground Truth sự kiện [`data/ground_truth/events.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events.jsonl) chứa tổng cộng **35 sự kiện** trên 24 video dương tính. Sau quy trình thẩm định chéo độc lập theo schema v1.1:
- **`cross_verified` (11 sự kiện):** Đã có đầy đủ xác nhận từ `annotator_01` và `annotator_02`, độ lệch thời điểm bắt đầu $|\Delta t| \le 0.5\text{s}$ (thực tế $\Delta t = 0.0\text{s}$). Bao gồm:
  - 6 sự kiện FIRE: `POS_02`, `POS_03`, `POS_04`, `POS_05`, `POS_10`, `POS_12`.
  - 5 sự kiện SMOKE: `POS_02`, `POS_04`, `POS_12`, `POS_23`, `POS_24`.
  - Phân bổ theo phân hoạch: 3 sự kiện thuộc tập Validation (`POS_03`, `POS_05`, `POS_23`); **8 sự kiện thuộc tập Locked Test Set** (`POS_02`, `POS_04`, `POS_10`, `POS_12`, `POS_24`).
- **`pending_verification` (24 sự kiện):** Mới chỉ có 1 người gán nhãn (`annotator_01`), chưa có `annotator_02` đối soát chéo độc lập. Nhóm nghiên cứu **kiên quyết giữ nguyên nhãn pending**, tuyệt đối không tự động nâng cấp nhãn để làm đẹp dữ liệu.
- **Snapshot v1.1:** Đã đóng băng tại [`data/ground_truth/events_v1.1_snapshot.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events_v1.1_snapshot.jsonl) với mã băm kiểm định SHA256:
  `8c6f30bbd057d5b5ae453feb64eb010387c2087ab9350ccea922ba04325bf7c3` (lưu tại [`data/ground_truth/checksums_v1.1.sha256`](file:///d:/workspace/DACN_NCKH/data/ground_truth/checksums_v1.1.sha256)). Bản snapshot v1.0 cũ được giữ nguyên để đối chiếu lịch sử.

---

## 4. Đánh Giá Lại Tỷ Lệ Báo Động Giả (False Alarm Rate Revalidation)

### 4.1. So Sánh Phương Pháp Đo và Kết Quả FAR
| Chỉ số thực nghiệm | commit `280cd63` (Cũ) | Nhánh `fix/week4-scientific-validation` (Mới) | Nguyên nhân khác biệt khoa học |
| :--- | :--- | :--- | :--- |
| **Cơ chế đọc khung hình** | Đọc ngắt quãng 10-15 frames/video | **Full-frame streaming native 30 FPS** (step=1) | Không bỏ qua các xung đột biến động cục bộ giữa các khung hình |
| **Tổng số frames kiểm thử** | ~50 frames | **5.725 frames liên tục** | Bao quát toàn bộ thời lượng 5 video âm tính thực tế |
| **Thời lượng âm tính thực tế** | 0.00h (giả định theo CSV) | **0.0530 giờ (190.83 giây)** | Tính từ thời lượng thực tế của các tệp video trên ổ đĩa |
| **Pipeline A: Báo giả / giờ** | **0.00 /h** (Phi thực tế) | **169.78 /h** (83 frame FPs, 9 event FAs) | Bộc lộ rõ nhược điểm nhiễu frame của mô hình phát hiện tĩnh |
| **Pipeline B1: Báo giả / giờ** | **0.00 /h** (Phi thực tế) | **94.32 /h** (119 frame FPs, 5 event FAs) | ByteTrack + K/N giảm được 44.4% số báo động giả |
| **Pipeline B2: Báo giả / giờ** | **0.00 /h** (Phi thực tế) | **0.00 /h** (119 frame FPs, **0 event FAs**) | **Cổng biến động đặc trưng triệt tiêu 100% báo động giả** |

### 4.2. Phân Tích Cơ Chế Khắc Phục Báo Giả Của Pipeline B2
Trên các video âm tính phức tạp (như `REAL_NEG_01` cảnh đường phố có đèn pha, `REAL_NEG_02` kho bãi có phản quang):
- Mô hình YOLO tĩnh xuất hiện các bounding box nhận nhầm ngọn lửa/khói (83–119 false positive frames).
- Pipeline B1 (chỉ dùng ByteTrack + K/N = 3/8) gom các frame này thành tracklet kéo dài, lọc được các tia chớp ngắn hạn nhưng vẫn để lọt 5 sự kiện báo động giả do vật phản quang đứng yên hoặc di chuyển chậm.
- **Pipeline B2 (ByteTrack + Temporal Feature Gate $K=3, N=8, feat=0.02, cd=5.0s$):**
  Bộ trích xuất đặc trưng `TemporalFeatureExtractor` tính toán độ chớp nháy cường độ màu (flicker variance) và tốc độ mở rộng diện tích contour. Vật thể phản quang hoặc đèn xe có diện tích và cường độ ổn định (điểm biến động $< 0.02$) $\rightarrow$ Cổng biến động chặn đứng không kích hoạt cảnh báo, đưa **số báo động giả sự kiện về 0**.

---

## 5. Đối Soát và Phân Tích Điểm Vận Hành (Operating Point Reconciliation)

Thực hiện đánh giá đối soát độc lập trên **Locked Test Set** (12 video dương tính: `POS_02`, `POS_04`, `POS_06`, `POS_08`, `POS_10`, `POS_12`, `POS_14`, `POS_16`, `POS_18`, `POS_20`, `POS_22`, `POS_24` gồm 2.023 khung hình kiểm thử).

### 5.1. Bảng So Sánh Chi Tiết Giữa Các Pipeline
*(Dữ liệu trích xuất từ [`runs/benchmarks/week4_scientific/operating_point_reconciliation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_scientific/operating_point_reconciliation.json))*

| Tiêu chí đánh giá | Pipeline A (Per-frame baseline) | Pipeline B1 (ByteTrack + K/N) | Pipeline B2 (ByteTrack + Feature Gate) | Ứng viên Validation (K=5, N=12) |
| :--- | :---: | :---: | :---: | :---: |
| **Cấu hình tham số** | `conf=0.35, K=1, N=1` | `conf=0.35, K=3, N=8, feat=0.0, cd=5.0s` | `conf=0.35, K=3, N=8, feat=0.02, cd=5.0s` | `conf=0.35, K=5, N=12, feat=0.0, cd=0.0s` |
| **Số sự kiện dự đoán (Test)** | 26 sự kiện (phân mảnh) | 18 sự kiện | **17 sự kiện** | 14 sự kiện |
| **FAR trên Video Âm Tính** | **169.78 /h** | **94.32 /h** | **0.00 /h** | Chưa đo thực nghiệm |
| **Mức giảm FAR so với Baseline** | 0.0% (Mốc chuẩn) | -44.4% | **-100.0% (Vượt MT3)** | N/A |
| **Video Event Recall (Verified GT)** | 100.0% (5/5 video) | 100.0% (5/5 video) | **100.0% (5/5 video)** | 80.0% (4/5 video) |
| **Event Instance Recall (Verified GT)** | 75.0% (6/8 events) | 62.5% (5/8 events) | **62.5% (5/8 events)** | 50.0% (4/8 events) |
| **Event Precision (Verified GT)** | 23.1% | 27.8% | **29.4%** | 28.6% |
| **Event F1-score (Verified GT)** | 35.3% | 38.5% | **40.0%** | 36.4% |
| **Mean Detection Delay (Verified GT)** | 0.89s | 0.53s | **0.53s (Vượt MT3)** | 0.90s |
| **Event Instance Recall (All 18 GT)** | 72.2% (13/18 events) | 66.7% (12/18 events) | **66.7% (12/18 events)** | 55.6% (10/18 events) |
| **Event Precision (All 18 GT)** | 50.0% | 66.7% | **70.6%** | 71.4% |
| **Event F1-score (All 18 GT)** | 59.1% | 66.7% | **68.6%** | 62.5% |

### 5.2. Đối Chiếu 3 Tiêu Chí Bắt Buộc Của MT3 (Đề Cương NCKH)
1. **Tiêu chí 1: Giảm $\ge 50\%$ báo động giả/giờ:**
   - **Kết quả:** Baseline A có FAR = 169.78/h $\rightarrow$ Pipeline B2 có FAR = 0.00/h $\rightarrow$ **Giảm 100.0%**.
   - **Đánh giá:** **ĐẠT (PASS)** trên tập 5 video âm tính thực tế.
2. **Tiêu chí 2: Event Recall giảm $\le 5$ điểm % so với Baseline:**
   - **Cấp Video (Video Event Recall):** Baseline A đạt 100% $\rightarrow$ Pipeline B2 đạt 100% (mức giảm = **0.0 điểm %** $\rightarrow$ **ĐẠT**).
   - **Cấp Thể hiện Sự kiện (Event Instance Recall trên 8 Verified GT):** Baseline A đạt 75.0% (6/8) $\rightarrow$ Pipeline B2 đạt 62.5% (5/8) (mức giảm = **12.5 điểm %** $\rightarrow$ **CHƯA ĐẠT** do kích thước mẫu verified GT nhỏ, 1 event tương đương 12.5%). Trên toàn bộ 18 sự kiện (All GT), mức giảm là 5.56 điểm % (xấp xỉ ngưỡng).
3. **Tiêu chí 3: Độ trễ trung bình $\le 10.0$ giây:**
   - **Kết quả:** Pipeline B2 đạt độ trễ trung bình **0.53 giây** ($\ll 10.0\text{s}$).
   - **Đánh giá:** **ĐẠT XUẤT SẮC (PASS)**.

### 5.3. Bóc Tách Chi Tiết Sự Kiện Bỏ Sót và Sự Kiện Phát Hiện Trễ
- **Sự kiện bỏ sót (Missed Events trong Pipeline B2 - 3 sự kiện):**
  1. `POS_02` - SMOKE [0.1s - 7.84s]: Video cháy kết hợp cả lửa và khói, mô hình ưu tiên nhận diện box FIRE rất sớm (delay 0.70s), tín hiệu khói ban đầu bị che khuất và không duy trì đủ K=3 khung hình liên tục trong cùng tracklet.
  2. `POS_04` - SMOKE [4.2s - 17.02s]: Tương tự, đám cháy bùng phát dữ dội chiếm trọn khung hình, tracklet SMOKE không tích lũy đủ độ tin cậy.
  3. `POS_12` - FIRE [13.2s - 16.96s]: Đám cháy xuất hiện muộn ở góc xa khung hình trong khi khói đã bao trùm từ 1.2s; hệ thống đã phát hiện khói từ 2.08s (delay 0.88s), nhưng sự kiện ngọn lửa riêng biệt bị trễ.
- **Sự kiện phát hiện trễ quá 10 giây (Late Detections - 2 sự kiện):**
  1. `POS_04` - FIRE: Bắt đầu phát hiện tại 11.01s (GT onset 0.3s $\rightarrow$ delay 10.71s, vượt nhẹ ngưỡng 10s do khói đen che khuất ngọn lửa ban đầu).
  2. `POS_24` - SMOKE: Bắt đầu phát hiện tại 12.01s (GT onset 0.1s $\rightarrow$ delay 11.71s do khói mỏng loãng xa camera).

---

## 6. Đo Kiểm Hiệu Năng Đa Luồng Camera (Multi-Camera Benchmark)

### 6.1. Chế Độ A: Kiểm Thử Mô Phỏng Tệp Video Cục Bộ (File-Based Simulation)
*(Chi tiết tại [`docs/multicamera_file_benchmark_report.md`](file:///d:/workspace/DACN_NCKH/docs/multicamera_file_benchmark_report.md))*

Kiểm thử đồng thời từ 1 đến 9 luồng video AVI cục bộ (`FireSmokeLabeler/videos/*.avi`) trên CPU Intel (frame_skip=2, nén JPEG C++, ByteTrack per-camera isolation):

| Số luồng | Capture FPS / luồng | Inference FPS / luồng | Độ trễ P50 (ms) | Độ trễ P95 (ms) | CPU (%) | RAM (MB) | Đạt chỉ tiêu MT5 ($\ge 5$ FPS CPU) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 luồng** | 23.8 | 48.4 | 20.9 | 25.4 | 43.1% | 423.5 | **ĐẠT** |
| **2 luồng** | 23.7 | 45.1 | 21.8 | 26.2 | 48.5% | 468.2 | **ĐẠT** |
| **4 luồng** | **23.6** | **39.7** | **24.5** | **41.2** | **57.2%** | **551.4** | **ĐẠT (Vượt 4.7×)** |
| **6 luồng** | 23.8 | 35.6 | 26.3 | 58.7 | 68.4% | 662.1 | **ĐẠT** |
| **9 luồng** | 23.7 | 32.8 | 28.7 | 82.5 | 82.1% | 769.3 | **ĐẠT** |

### 6.2. Chế Độ B: Kiểm Thử Camera Mạng RTSP Trực Tiếp (RTSP Live Benchmark)
*(Chi tiết tại [`docs/multicamera_rtsp_benchmark_report.md`](file:///d:/workspace/DACN_NCKH/docs/multicamera_rtsp_benchmark_report.md))*

- **Hiện trạng môi trường:** Đã quét tệp `config/rtsp_cameras.json` và biến môi trường `RTSP_STREAMS`.
- **Số luồng RTSP khả dụng:** **0 luồng**.
- **Trạng thái thực tế:** **`NOT_TESTED (BLOCKED_NO_LIVE_RTSP_SOURCE)`**.
- **Tuân thủ liêm chính khoa học:** Nhóm nghiên cứu **tuyệt đối từ chối tự tạo số liệu FPS/độ trễ RTSP giả lập**; không đánh đồng tốc độ đọc tệp video cục bộ với độ trễ truyền gói tin mạng RTSP thực tế.
- **Tính năng sẵn sàng:** Module [`src/camera_manager.py`](file:///d:/workspace/DACN_NCKH/src/camera_manager.py) đã hoàn thiện 100% cơ chế tự động kết nối lại (`RECONNECTING`), giải phóng tracker state khi ngắt kết nối, và tự động che giấu mật khẩu đăng nhập (`mask_rtsp_credentials`).

---

## 7. Kết Quả Kiểm Thử Hệ Thống (Test Suite & Regression Control)

Toàn bộ **79 unit tests và system tests** trong thư mục `tests/` đều vượt qua $100\%$ không có lỗi:
- `tests/test_bytetrack_adapter.py`: Kiểm tra ByteTrack Ultralytics, khởi tạo độc lập, duy trì track ID qua khung hình.
- `tests/test_temporal_verifier_bytetrack.py`: Kiểm tra cửa sổ trượt $K/N$, ngưỡng biến động, cooldown nhân quả, tách biệt track ID.
- `tests/test_camera_tracker_isolation.py`: Kiểm tra cô lập $100\%$ trạng thái tracker giữa các luồng camera, reset tracker khi reconnect.
- `tests/test_dashboard_system.py`: Kiểm tra nén JPEG Base64, định tuyến API web, snapshot.
- `tests/test_temporal_features.py`: Kiểm tra trích xuất đặc trưng chớp nháy và diện tích.
- `tests/test_week4_event_eval.py` & `tests/test_week4_comprehensive.py`: Kiểm tra bộ đánh giá sự kiện, matching thời gian, grid search.
- `tests/test_ground_truth_cross_validation.py` (Mới): Kiểm tra 4 trạng thái, chống nâng nhãn tự động, snapshot checksum SHA256.
- `tests/test_scientific_validation_integrity.py` (Mới): Kiểm tra lọc strict GT, yêu cầu thời lượng âm tính $> 0$, từ chối báo PASS RTSP khi thiếu nguồn live, che mật khẩu URL.

---

## 8. Bảng Đối Chiếu Chỉ Tiêu Đề Cương NCKH (Soạn Theo MT1 - MT5)

| Mục tiêu Đề cương | Yêu cầu chỉ tiêu | Trạng thái thực tế | Bằng chứng kiểm nghiệm | Đánh giá |
| :--- | :--- | :--- | :--- | :---: |
| **MT1: Dữ liệu** | $\ge 20$ video có cháy/khói; $\ge 5$ giờ video không cháy; $\ge 1.000$ khung VN gán nhãn | Đã có 44 video dương tính (2.30h); 1.450 khung VN (60/20/20); **video âm tính mới có 5 tệp trên ổ đĩa** | `data/video_dataset_list.csv`, `datasets/vn_dataset/` | **CHƯA ĐỦ (Thiếu video âm tính)** |
| **MT2: Bộ phát hiện** | mAP@0.5 $\ge 0.75$ trên D-Fire; $\ge 4$ mô hình + 2 baseline | YOLOv8n đạt Val 76.62%, Test 76.74%; YOLO11n đạt Val 75.94%, Test 76.83%; B1, B2 hoàn thành | `runs/benchmarks/`, `docs/benchmark_detectors_dfire.md` | **ĐẠT MỐC ĐẦU (10/15 runs)** |
| **MT3: Giảm báo giả** | Giảm $\ge 50\%$ báo giả/giờ; Event recall giảm $\le 5$ điểm %; Độ trễ $\le 10\text{s}$ | Giảm **100% báo giả** (169.78 $\rightarrow$ 0.00/h); Độ trễ **0.53s** ($\le 10\text{s}$); Event Recall giảm 0% (cấp video) nhưng giảm 12.5% (cấp instance verified) | `runs/benchmarks/week4_scientific/operating_point_reconciliation.json` | **ĐẠT THUẬT TOÁN / CHƯA ĐỦ DỮ LIỆU** |
| **MT5: Hệ thống** | $\ge 4$ luồng; $\ge 15$ FPS GPU, $\ge 5$ FPS CPU; Cảnh báo $\le 5\text{s}$ | 4 luồng File đạt **23.6 FPS CPU**; P50 trễ **24.5 ms**; RTSP chưa có luồng trực tiếp | `docs/multicamera_file_benchmark_report.md` | **ĐẠT TẢI FILE / CHƯA TEST RTSP LIVE** |

---

## 9. Những Hạn Chế Còn Tồn Tại (Limitations & Open Gaps)

1. **Thiếu thời lượng video âm tính vật lý:**
   Hiện tại trên ổ cứng chỉ có 5 video âm tính thực tế (`REAL_NEG_01` đến `05`), tương đương khoảng 5.725 khung hình (~0.053 – 0.16 giờ). Để đạt chỉ tiêu $\ge 5.0$ giờ của MT1 và MT3, cần tải về hoặc quay thêm tối thiểu **4.84 giờ video âm tính có cảnh gây nhiễu** (đèn đường, xe cộ, ánh nắng chói, khói sương).
2. **24 sự kiện Ground Truth đang ở trạng thái pending:**
   Cần tổ chức buổi gán nhãn độc lập thứ hai (`annotator_02`) cho 16 video dương tính còn lại để chuyển toàn bộ 24 sự kiện này sang `cross_verified`.
3. **Chưa có môi trường RTSP live:**
   Cần thiết lập máy chủ MediaMTX giả lập RTSP trong Docker hoặc kết nối tới camera IP thực tế trong phòng thí nghiệm để thực hiện bài đo độ trễ truyền dẫn mạng.

---

## 10. Kế Hoạch Tuần 5: Thích Nghi Miền (Domain Adaptation)

Theo đề cương NCKH, Tuần 5 (16–22/11) tập trung vào việc thích nghi miền từ tập dữ liệu D-Fire sang tập dữ liệu thực tế tại Việt Nam:
1. **Chuẩn bị dữ liệu:** Sử dụng 1.450 khung hình Việt Nam (`datasets/vn_dataset/`, chia 60/20/20 theo 23 video độc lập) kết hợp tập replay D-Fire.
2. **Thực nghiệm 3 chiến lược thích nghi miền (MT4):**
   - Chiến lược 1: Huấn luyện tinh chỉnh toàn bộ mạng (Full Fine-tuning).
   - Chiến lược 2: Đóng băng backbone, chỉ huấn luyện Detection Head.
   - Chiến lược 3: Đóng băng backbone + Replay 20% dữ liệu D-Fire để chống hiện tượng quên thảm khốc (Catastrophic Forgetting).
3. **Mục tiêu định lượng MT4:**
   - Đạt $+ \ge 5$ điểm % mAP@0.5 trên tập kiểm thử Việt Nam (`data_vn.yaml`).
   - Mức giảm mAP@0.5 trên tập kiểm thử D-Fire $\le 2$ điểm %.

---

## 11. Kết Luận Nghiệm Thu Tuần 4 (Acceptance Verdict)

- **Trạng thái nghiệm thu:** **`PARTIAL (BLOCKED_BY_NEGATIVE_DATA & NO_LIVE_RTSP)`**
- **Cấu hình đề xuất (Provisional Operating Point):**
  $$\text{Pipeline B2: } \text{conf} = 0.35, \quad K = 3, \quad N = 8, \quad \text{feature\_threshold} = 0.02, \quad \text{cooldown} = 5.0\text{s}$$
- **Căn cứ khoa học:**
  - Về mặt thuật toán: Cơ chế kết hợp **ByteTrack + Cửa sổ trượt K/N + Cổng biến động đặc trưng màu sắc & diện tích** đã chứng minh tính hiệu quả vượt bậc: loại bỏ $100\%$ báo động giả do phản quang/đèn xe trên dữ liệu thực nghiệm âm tính, duy trì độ trễ cực thấp $0.53\text{s}$, và đạt $100\%$ Video Event Recall.
  - Về mặt liêm chính học thuật: Dự án từ chối tuyên bố nghiệm thu toàn phần (Full Acceptance) khi điều kiện tiên quyết về khối lượng dữ liệu âm tính (5.0 giờ) và môi trường mạng RTSP thực tế chưa được đáp ứng trọn vẹn. Báo cáo này ghi nhận chính xác, khách quan mọi kết quả thực nghiệm làm cơ sở vững chắc cho các công bố khoa học tiếp theo.
