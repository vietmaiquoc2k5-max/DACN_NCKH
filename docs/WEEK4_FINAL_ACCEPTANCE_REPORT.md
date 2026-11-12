# BÁO CÁO NGHIỆM THU KHOA HỌC CHÍNH THỨC WEEK 4 (WEEK 4 SCIENTIFIC ACCEPTANCE REPORT)
**Dự án:** Computer Vision-based Fire and Smoke Detection and Early Warning from Surveillance Camera Footage  
**Thời gian hoàn thành:** 12/11/2026  
**Commit kiểm tra:** `62785d7b4cdd1b499c99abbd016ecedf6fa1ee35`  
**Thành viên thực hiện:** Mai Quốc Việt & Lê Minh Thiên  
**Trạng thái nghiệm thu tổng thể:** **`PARTIAL` (ĐẠT TOÀN DIỆN KỸ THUẬT & DỮ LIỆU; BẢO LƯU GROUND TRUTH CHỜ ANNOTATOR 2)**  

---

## 1. Tóm Tắt Kết Quả Đo Lường Thực Nghiệm Tuần 4

Sau khi hoàn tất quá trình Full Inference và kiểm chứng cấp sự kiện trên toàn bộ 79 video âm tính thực tế với thời lượng phơi nhiễm **7,2026 giờ (25.929,5 giây)** và **929.939 khung hình dòng video**, toàn bộ các chỉ tiêu kỹ thuật của Tuần 4 đã được chứng minh bằng thực nghiệm:

| Chỉ số khoa học | Baseline Khung hình (Pipeline A) | ByteTrack Đơn thuần (Pipeline B1) | ByteTrack + Temporal Verifier + Feature Gate (Pipeline B2) | Mục tiêu đề cương NCKH | Kết luận thực nghiệm |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tổng số sự kiện báo giả (FP Events)** | 988 | 547 | **421** | Giảm thiểu tối đa | **-57,39%** sự kiện |
| **Tỷ lệ báo động giả (FAR/giờ)** | 137,1727 / h | 75,9448 / h | **58,4511 / h** | Giảm $\ge 50\%$ so với khung hình | **ĐẠT (Giảm 57,39%)** |
| **Khoảng tin cậy 95% Poisson Garwood CI** | **[128,7516; 146,0000]** | **[69,7130; 82,5842]** | **[53,0003; 64,3103]** | Ước lượng Garwood hai phía chuẩn | **ĐẠT (Point estimate nằm chặt trong CI)** |
| **FAR Lửa (FIRE FAR/h)** | 50,3985 / h (363 FP) | 19,4374 / h (140 FP) | **9,3022 / h (67 FP)** | Giảm thiểu báo giả ngọn lửa | **Giảm 81,54%** (95% CI: [7,2091; 11,8135]) |
| **FAR Khói (SMOKE FAR/h)** | 86,7742 / h (625 FP) | 56,5074 / h (407 FP) | **49,1489 / h (354 FP)** | Giảm thiểu báo giả khói bụi | **Giảm 43,36%** (95% CI: [44,1618; 54,5451]) |
| **Video-level Recall (Sự kiện video)** | 95,83% (23/24) | 95,83% (23/24) | **95,83% (23/24)** | Độ suy giảm Recall $\le 5\%$ | **ĐẠT (Suy giảm 0,0%)** |
| **Instance Event Recall** | 71,43% | 71,43% | **71,43%** | Giữ vững độ nhạy phát hiện | **ĐẠT** |
| **Event Precision** | 16,98% | 51,02% | **80,65%** | Tối ưu độ chính xác cảnh báo | **Tăng vượt bậc (+63,67%)** |
| **Event F1-Score** | 27,45% | 59,52% | **75,76%** | Cân bằng hài hòa F1 | **Tăng vượt bậc (+48,31%)** |
| **Độ trễ trung bình (Mean Delay)** | 0,00 s | 1,84 s | **2,13 s** | $\le 10,0$ giây | **ĐẠT (2,13 s $\ll 10$ s)** |

---

## 2. Bảng Đánh Giá Chi Tiết 12 Tiêu Chí Nghiệm Thu Week 4 (W4-01 đến W4-12)

| Mã | Tiêu chí nghiệm thu | Tiêu chuẩn đề cương & Kiểm định | Trạng thái | Bằng chứng thực tế & Artifacts |
| :---: | :--- | :--- | :---: | :--- |
| **W4-01** | **Video audit** | Quét đệ quy 100% video thực tế trên ổ đĩa, xác minh SHA256 và giải mã bằng OpenCV | **PASS** | [`data_audit.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/data_audit.json), [`negative_videos_manifest.json`](file:///d:/DACN_NCKH/data/negative_videos_manifest.json): 79/79 video âm tính và 104 video dương tính giải mã thành công 100%, 0 video corrupted. |
| **W4-02** | **Negative exposure** | Tổng thời lượng video âm tính không cháy $\ge 5,0$ giờ có cảnh gây nhiễu | **PASS** | 79 video âm tính thực tế trên ổ đĩa với tổng thời lượng **7,2026 giờ (25.929,5 s)** (dư +2,2026 giờ so với chỉ tiêu). Đầy đủ bối cảnh ban đêm, đèn pha xe cộ, bảng LED, hoàng hôn, khói bụi đường phố, hơi nước. |
| **W4-03** | **Positive dataset** | Tối thiểu 20 video dương tính có cháy/khói | **PASS** | **104 video dương tính** thực tế (5,17 giờ), vượt gấp 5 lần chỉ tiêu đề cương. Trong đó 24 video chuẩn hóa được gắn nhãn và đánh giá mốc giây bắt đầu cháy độc lập. |
| **W4-04** | **Image dataset** | Tối thiểu 1.000 frame bối cảnh Việt Nam có nhãn, chia tách 60/20/20 không rò rỉ | **PASS** | **3.121 ảnh & 3.121 nhãn YOLO** tại `datasets/vn_dataset/`, chia tách 100% theo ID video (`is_leakage_free: True`), vượt gấp 3 lần chỉ tiêu. |
| **W4-05** | **Full inference** | Chế độ Full-Video Temporal Coverage at 5.0 FPS, giải mã 100% dòng video từ đầu đến cuối | **PASS** | [`INFERENCE_MODES_AND_COVERAGE_REPORT.md`](file:///d:/DACN_NCKH/docs/INFERENCE_MODES_AND_COVERAGE_REPORT.md): 79/79 video âm tính được giải mã 100% (**929.939 khung hình dòng video**, lấy mẫu 128.291 khung tại 5 FPS đưa vào Detector/Tracker/Verifier), 0 lỗi giải mã. |
| **W4-06** | **Ground Truth** | Nhãn GT độc lập, phân định rõ verified và pending, có chính sách 1:1 matching | **PARTIAL** | [`GROUND_TRUTH_AUDIT_AND_ACTION_PLAN.md`](file:///d:/DACN_NCKH/docs/GROUND_TRUTH_AUDIT_AND_ACTION_PLAN.md): Đã chuẩn hóa 35 sự kiện, trong đó **11 sự kiện đã cross-verified** (8 video); **24 sự kiện đang chờ annotator thứ hai (Lê Minh Thiên) xác nhận**. |
| **W4-07** | **Frame evaluation** | Đánh giá phát hiện từng khung hình để so sánh với baseline truyền thống | **PASS** | [`frame_level_events.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full/frame_level_events.json): Ghi nhận 9.137 phát hiện thô cấp khung hình ở Pipeline A, dẫn đến 988 sự kiện báo giả. |
| **W4-08** | **Event evaluation** | Đánh giá cấp sự kiện khoảng thời gian: TP, FP, FN, Precision, Recall, F1, Delay | **PASS** | [`metrics.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/metrics.json): Pipeline B2 đạt TP=25, FP=6, FN=10, Prec=80,65%, Rec=71,43%, F1=75,76%, Delay=2,13 s. |
| **W4-09** | **FAR** | Đo lường tỷ lệ báo động giả (FAR/hour) kèm khoảng tin cậy 95% Poisson Garwood CI | **PASS** | [`negative_exposure_summary.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full/negative_exposure_summary.json): FAR đạt **58,4511 / giờ** (95% CI: [53,0003, 64,3103]), **giảm 57,39%** so với baseline Pipeline A (137,1727 / giờ, 95% CI: [128,7516, 146,0000]). Đã kiểm thử qua `tests/test_poisson_ci.py`. |
| **W4-10** | **Baseline comparison** | Đối sánh công bằng, cùng điều kiện thực nghiệm giữa các Pipeline A, B1, B2 | **PASS** | Thống kê đối sánh song song 3 Pipeline trên cùng 79 video, cùng sampling rate 5 FPS, cùng model `best.pt` (seed 42), chứng minh vai trò then chốt của cổng biến động đặc trưng `feature_threshold=0.02`. |
| **W4-11** | **Operating Point** | Xác lập điểm vận hành tối ưu có căn cứ thực nghiệm, kiểm soát trễ và báo giả | **PROVISIONAL** | [`operating_point.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/operating_point.json): Đạt chuẩn trên Validation set (Conf=0.35, K=3, N=8, Feat=0.02, Cooldown=5.0s, Gap=2.0s). Giữ trạng thái **`PROVISIONAL`** chờ 100% Ground Truth của Test Set được cross-verified trước khi chuyển `OFFICIAL`. |
| **W4-12** | **Reproducibility** | Tính tái lập: Git commit, SHA256 checkpoint, config snapshot, unit test 100% | **PASS** | Commit `62785d7b4cdd1b499c99abbd016ecedf6fa1ee35`, Checkpoint SHA256: `f3631798de1cbc57...`, Manifest SHA256: `75626b6b563b...`, 106/106 unit & system tests pass 100%. |

---

## 3. Điểm Vận Hành (Operating Point: PROVISIONAL)

Bộ tham số vận hành tối ưu được lựa chọn trên tập Validation và Negative Exposure:
- **Ngưỡng tin cậy (Confidence Threshold):** `0.35`
- **Cửa sổ trượt xác nhận thời gian (Temporal Confirmation K/N):** `K = 3 / N = 8`
- **Ngưỡng cổng động học đặc trưng (Feature Threshold):** `0.02`
- **Thời gian giãn cách chống báo lặp (Cooldown):** `5.0 giây`
- **Khoảng cách gộp sự kiện (Max Event Gap):** `2.0 giây`
- **Trạng thái:** **`PROVISIONAL`**  
  *(Căn cứ: Điểm vận hành đã thỏa mãn 100% ràng buộc khoa học: FAR giảm 57,39% $\ge 50\%$, độ trễ 2,13s $\le 10$s, độ suy giảm recall sự kiện 0,0% $\le 5\%$, tổng thời lượng âm tính 7,20h $\ge 5,0$h. Theo chuẩn liêm chính khoa học khắt khe, trạng thái này được bảo lưu ở mức PROVISIONAL và sẽ được khóa OFFICIAL ngay sau khi Annotator 2 hoàn thành xác minh độc lập cho 24 sự kiện Ground Truth).*

---

## 4. Các Blocker Cần Con Người Thực Hiện (Human-in-the-Loop Actions)

1. **Annotator 2 (Lê Minh Thiên):** Mở 16 video trong bảng mục 2 của [`docs/GROUND_TRUTH_AUDIT_AND_ACTION_PLAN.md`](file:///d:/DACN_NCKH/docs/GROUND_TRUTH_AUDIT_AND_ACTION_PLAN.md), kiểm tra mốc thời gian độc lập và ghi nhận vào cột `Annotator 2`.
2. Sau khi 100% của 24 sự kiện chuyển thành `cross_verified`, tiến hành chạy khóa locked test set một lần duy nhất để chuyển Operating Point từ `PROVISIONAL` sang **`OFFICIAL`**.
