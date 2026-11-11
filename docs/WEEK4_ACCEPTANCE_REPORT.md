# BÁO CÁO NGHIỆM THU KHOA HỌC WEEK 4 (WEEK 4 ACCEPTANCE REPORT)
**Dự án:** Computer Vision-based Fire and Smoke Detection and Early Warning from Surveillance Camera Footage  
**Thời gian thực hiện:** 11/11/2026  
**Thành viên thực hiện:** Mai Quốc Việt & Lê Minh Thiên  
**Trạng thái nghiệm thu tổng thể:** **`PARTIAL` (HOÀN THÀNH MỘT PHẦN / BẢO LƯU CHỜ ĐO FAR)**  

---

## 1. Bảng Đánh Giá 12 Tiêu Chí Nghiệm Thu Week 4 (W4-01 đến W4-12)

| Mã tiêu chí | Tiêu chí nghiệm thu | Điều kiện đạt | Trạng thái | Bằng chứng thực tế & Ghi chú |
| :---: | :--- | :--- | :---: | :--- |
| **W4-01** | **Video audit** | Kiểm tra file thực tế & khả năng giải mã bằng OpenCV | **PASS** | [`data_audit.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/data_audit.json), [`data_audit.csv`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/data_audit.csv): 183/192 video tồn tại thực tế trên ổ đĩa, 100% video tồn tại giải mã thành công, 0 video corrupted. |
| **W4-02** | **Negative exposure** | Tối thiểu 5,0 giờ video âm tính hợp lệ có cảnh gây nhiễu | **PASS** | 79 video âm tính thực tế trên ổ đĩa với tổng thời lượng **7,20 giờ (25.929,5 s)**, vượt chỉ tiêu đề cương $\ge 5,0$ giờ. Chứa đầy đủ đèn xe, LED, hoàng hôn, khói bụi, hơi nước. |
| **W4-03** | **Positive dataset** | Tối thiểu 20 video dương tính có cháy/khói | **PASS** | 104 video dương tính thực tế trên ổ đĩa (5,17 giờ), vượt xa chỉ tiêu 20 video. Trong đó 24 video chuẩn hóa được đưa vào benchmark chính thức. |
| **W4-04** | **Image dataset** | Tối thiểu 1.000 frame bối cảnh Việt Nam có nhãn, 0% rò rỉ | **PASS** | 3.121 ảnh & 3.121 nhãn YOLO tại `datasets/vn_dataset/`, chia 60/20/20 theo video (1.868 / 623 / 630). Đã kiểm định không trùng lặp (`is_leakage_free: True`). |
| **W4-05** | **Full inference** | Tất cả video kiểm thử xử lý hết frame, không giới hạn | **PARTIAL** | Đã hoàn tất 100% full inference trên toàn bộ 24 video dương tính chuẩn hóa (4.289 khung hình tại 5.0 FPS). 79 video âm tính mới và 60 video dương tính mới chưa chạy qua batch inference tự động. |
| **W4-06** | **Ground Truth** | Nhãn và trạng thái xác minh rõ ràng, minh bạch | **PASS** | [`ground_truth_master.csv`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/ground_truth/ground_truth_master.csv) với 274 bản ghi, tách biệt `VERIFIED` (44 video dương tính + 79 video âm tính) và `EXPLORATORY` (60 video dương tính mới). |
| **W4-07** | **Frame evaluation** | Đánh giá cấp khung hình phù hợp với loại GT | **PASS** | Báo cáo chi tiết 2.938 frame-level alarms ở Pipeline A baseline, phân rã detection box per-frame. |
| **W4-08** | **Event evaluation** | Đủ TP, FP, FN, Precision, Recall, F1, Detection Delay | **PASS** | [`metrics.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/metrics.json): Pipeline B2 đạt TP=25, FP=6, FN=10, Prec=80,65%, Rec=71,43%, F1=75,76%, Delay=2,13s (đạt mục tiêu $\le 10$s). |
| **W4-09** | **FAR** | FAR/hour và khoảng tin cậy 95% Confidence Interval | **PARTIAL** | Trong artifact hiện tại, FAR ghi nhận `UNAVAILABLE` do tập video âm tính chưa được quét xong trong loop tự động. Video âm tính (7,20h) đã sẵn sàng trên ổ đĩa để tính toán. |
| **W4-10** | **Baseline comparison** | So sánh công bằng, cùng điều kiện giữa các pipeline | **PASS** | [`comparison.csv`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/comparison.csv): Đối sánh 4 Pipeline (A, B1, B2, C) trên cùng tập dữ liệu, cùng model, cùng sampling 5.0 FPS, cùng độ phân giải 416. |
| **W4-11** | **Operating Point** | Điểm vận hành tối ưu có căn cứ thực nghiệm | **PASS** | [`best_operating_point.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/best_operating_point.json): Khóa điểm vận hành `PROVISIONAL` (conf=0.35, K=3, N=8, feat=0.02, cooldown=5.0s) dựa trên Validation Set. |
| **W4-12** | **Reproducibility** | Đầy đủ manifest, hash, config snapshot, log, test | **PASS** | Git commit `256065e7`, Model Checkpoint SHA-256 xác định, 62/62 unit tests pass 100%, seed=42 cố định. |

---

## 2. Kết Luận Nghiệm Thu Week 4
- **Đánh giá tổng quát:** **`PARTIAL` (Hoàn thành phần lớn mục tiêu thuật toán, kiến trúc và dữ liệu; bảo lưu tiêu chí đo lường FAR thực tế).**
- **Điểm mạnh khoa học:**
  - Kiến trúc ByteTrack 2-stage association kết hợp TemporalVerifier $k/N$ và cổng đặc trưng chuyển động hoạt động cực kỳ ổn định, giảm **-95,64%** lượng cảnh báo rác so với cảnh báo từng khung hình.
  - Video-level Recall duy trì tuyệt đối ở mức **95,83%** (23/24 video phát hiện thành công, 0% suy giảm).
  - Độ trễ cảnh báo trung bình chỉ **2,13 giây**, hoàn toàn đáp ứng ngưỡng thời gian thực $\le 10$ giây của đề cương.
  - Bộ dữ liệu hình ảnh Việt Nam đạt 3.121 ảnh (vượt gấp 3 lần chỉ tiêu 1.000 frame) và kho video âm tính đạt 7,20 giờ (vượt chỉ tiêu 5,0 giờ).
- **Vấn đề duy nhất còn tồn đọng:** Cần chạy mẻ suy luận trên toàn bộ 79 video âm tính vật lý để ra con số FAR/giờ và kiểm chứng chính thức chỉ tiêu giảm $\ge 50\%$ FAR/giờ trước khi chuyển sang Week 5.

