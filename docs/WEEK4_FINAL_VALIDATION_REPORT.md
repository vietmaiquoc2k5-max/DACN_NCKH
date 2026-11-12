# BÁO CÁO TỔNG KẾT NGHIỆM THU KHOA HỌC TUẦN 4
**Tài liệu:** `docs/WEEK4_FINAL_VALIDATION_REPORT.md`  
**Dự án:** Computer Vision-based Fire and Smoke Detection and Early Warning from Surveillance Camera Footage  
**Thời gian lập báo cáo:** 12/11/2026 16:20 UTC+7  
**Commit cơ sở:** `c95907f6b8cf12cc02563dfaab209f304dbc3259`  
**Chủ nhiệm đề tài:** Lê Minh Thiên  
**Người thực hiện & Kiểm thử:** Mai Quốc Việt  

---

## 1. Kết Luận Nghiệm Thu Tuần 4 (Final Acceptance Status)

### ĐÁNH GIÁ: **PARTIAL**

**Lý giải ngắn gọn:**
- **Đạt toàn diện các chỉ tiêu kỹ thuật thực nghiệm:** Đã củng cố độ tin cậy tuyệt đối cho video audit, hoàn tất tính toán streaming SHA-256 trên 79 video âm tính thực tế, vượt mục tiêu thời lượng phơi nhiễm $\ge 5,0$ giờ (đạt 6,9846h untruncated / 7,1312h stream thực / 7,2026h benchmark), hoàn tất full-video temporal sampling inference trên 100% video (929.939 frames decode, 128.291 frames đưa vào mô hình), giảm False Alarm Rate (FAR) **57,39%** (từ 137,17/h xuống 58,45/h, vượt chỉ tiêu MT3 $\ge 50\%$), độ trễ trung bình 2,13s $\le 10$s, recall sự kiện giữ nguyên 95,83% (suy giảm 0,0% $\le 5\%$), 42/42 bài unit/integration test đạt 100% PASS.
- **Lý do giữ mức PARTIAL thay vì COMPLETE:**
  1. **Ground Truth:** Bộ nhãn 35 sự kiện dương tính mới có 11 sự kiện đạt kiểm định chéo kép (`cross_verified`), 24 sự kiện còn ở trạng thái `pending_verification` đang chờ thẩm định viên thứ hai (Annotator 02) rà soát độc lập.
  2. **Operating Point:** Duy trì nghiêm cẩn ở trạng thái `PROVISIONAL` theo nguyên tắc khoa học, không tự ý nâng lên `OFFICIAL` khi 24 sự kiện Ground Truth chưa hoàn tất thẩm định chéo.
  3. **RTSP Live Benchmark:** Ghi nhận trung thực `NOT_TESTED / BLOCKED_NO_LIVE_RTSP_SOURCE` do trạm đo hiện chưa kết nối phần cứng camera IP trực tiếp tại hiện trường.

---

## 2. Dữ Liệu Thực Tế (Dataset Integrity & Scope)

| Danh mục dữ liệu | Số lượng | Thời lượng / Số khung hình | Trạng thái xác minh | Ghi chú khoa học |
|---|---:|---:|---|---|
| **Tổng video âm tính phát hiện trên đĩa** | 79 video | 7,30 GB | 100% Full Streaming SHA-256 | `datasets/negative_videos` (65 video) + `datasets/test_videos/neg` (14 video) |
| **Video âm tính hoàn toàn hợp lệ (Untruncated VALID)** | **77 video** | **6,9846 giờ** (25.144,67 s) | `NORMAL_EOF`, `VALID` | Vượt chỉ tiêu MT1 $\ge 5,0$ giờ (+39,7%) |
| **Video âm tính ngắt luồng sớm (Truncated)** | **2 video** | **0,1465 giờ** (527,47 s stream thực) | `TRUNCATED_STREAM`, `NEEDS_REVIEW` | `VN_NEG_10` (thiếu 3.248 frames) và `VN_NEG_48` (thiếu 5.912 frames) do lỗi packet bitstream |
| **Tổng phơi nhiễm luồng stream thực tế giải mã** | 79 video | **7,1312 giờ** (25.672,14 s) | 929.939 frames giải mã tuần tự đến EOF | Mẫu số phơi nhiễm thực tế vật lý |
| **Tổng phơi nhiễm chuẩn hóa Benchmark Run** | 79 video | **7,2026 giờ** (25.929,53 s) | 79/79 video hoàn tất inference | Mẫu số chuẩn hóa của đợt chạy `run_20261111_233035_neg_full` |
| **Video dương tính chuẩn hóa** | 24 video | 35 sự kiện (16 FIRE, 19 SMOKE) | 11 `cross_verified`, 24 `pending_verification` | `FireSmokeLabeler/videos/*.avi` (`POS_01` – `POS_24`) |
| **Khung hình Việt Nam gán nhãn** | 3.121 ảnh | 3.121 XML/TXT labels | Đã phân chia 60/20/20 độc lập theo video | `datasets/vn_dataset/` có làm mờ khuôn mặt và biển số xe |

---

## 3. Nhật Ký Suy Luận Thực Nghiệm (Full-Video Temporal Sampling Inference)

- **Mã đợt suy luận (Run ID):** `run_20261111_233035_neg_full`
- **Số lượng video đưa vào pipeline:** 79/79 video (Tỷ lệ bao phủ 100%).
- **Số lượng video hoàn tất:** 79/79 video (100% COMPLETED, 0 video lỗi bỏ dở).
- **Tổng số khung hình giải mã (Decoded Frames):** **929.939 frames** đến điểm kết thúc EOF của luồng stream.
- **Tổng số khung hình đưa vào mô hình YOLO (Sampled Frames):** **128.291 frames**.
- **Chính sách lấy mẫu thời gian:** Temporal Sampling tại **5 FPS** (`round(native_fps / 5.0)`).
  > **Lưu ý khoa học:** Đây là suy luận bao phủ 100% thời lượng video tại tần số lấy mẫu 5 FPS (chuẩn giám sát camera), không phải suy luận trên 100% số khung hình gốc (native full-frame).
- **Tổng thời gian xử lý thực tế:** 1.968,42 giây (~32,8 phút).
- **Tốc độ suy luận trung bình:** **65,17 FPS** (Vượt xa chỉ tiêu MT5 $\ge 15$ FPS GPU).

---

## 4. Kết Quả Khoa Học & Bảng So Sánh 3 Pipeline

Bảng đối sánh hiệu năng sự kiện trên tập dữ liệu kiểm chứng độc lập:

| Chỉ số khoa học | Pipeline A (Raw YOLO) | Pipeline B1 (ByteTrack) | Pipeline B2 (Full Verifier) | Chỉ tiêu Đề cương MT3 | Kết luận |
|---|---:|---:|---:|---:|---|
| **Video-level Recall** | 95,83% | 95,83% | **95,83%** | Suy giảm $\le 5\%$ | **ĐẠT** (0% suy giảm) |
| **Event-instance Recall** | 71,43% | 71,43% | **71,43%** | Duy trì mức phát hiện | **ĐẠT** |
| **Event Precision** | 52,38% | 73,33% | **80,65%** | Càng cao càng tốt | **TĂNG +28,27%** |
| **Event F1-Score** | 0,6044 | 0,7237 | **0,7576** | Càng cao càng tốt | **TĂNG +0,1532** |
| **Mean Detection Delay** | 1,42 s | 1,89 s | **2,13 s** | $\le 10,0$ s | **ĐẠT** (rất nhanh) |
| **Số báo động giả (Alarms)** | 988 | 547 | **421** | Giảm $\ge 50\%$ | **GIẢM 57,39%** |
| **FAR (Benchmark: 7,2026h)** | 137,17 / giờ | 75,94 / giờ | **58,45 / giờ** | Giảm $\ge 50\%$ | **ĐẠT** |
| **95% Poisson CI (Benchmark)** | [128,75, 145,98] | [69,75, 82,55] | **[53,00, 64,33]** | Garwood Exact CI | Đã kiểm chứng toán học |
| **FAR (Stream thực: 7,1312h)** | 138,55 / giờ | 76,71 / giờ | **59,04 / giờ** | Giảm $\ge 50\%$ | **ĐẠT** (Giảm 57,39%) |
| **FAR (Untruncated: 6,9846h)** | 141,45 / giờ | 78,32 / giờ | **60,28 / giờ** | Giảm $\ge 50\%$ | **ĐẠT** (Giảm 57,39%) |

---

## 5. Điểm Vận Hành (Operating Point)

- **Ngưỡng tin cậy phát hiện (Confidence Threshold):** `0.35`
- **Số khung hình xác nhận (K / N Window):** `3 / 8`
- **Ngưỡng cổng biến động đặc trưng (Feature Threshold):** `0.02`
- **Thời gian hồi phục (Cooldown):** `5.0 giây`
- **Khoảng cách gộp sự kiện tối đa (Max Event Gap):** `2.0 giây`
- **Trạng thái điểm vận hành:** **`PROVISIONAL`** (Tạm thời khóa, chưa chuyển OFFICIAL).
- **Lý do giữ PROVISIONAL:** Đang chờ thẩm định viên thứ hai kiểm định chéo độc lập 24 sự kiện pending trong tập Ground Truth.

---

## 6. Đánh Giá Chi Tiết 14 Tiêu Chí Nghiệm Thu (Acceptance Criteria Matrix)

| Mã | Tiêu chí | Điều kiện nghiệm thu | Trạng thái | Bằng chứng kiểm chứng |
|---|---|---|---|---|
| **A1** | Video audit | Có full SHA-256 streaming và decode verification tuần tự đến EOF | **PASS** | `src/video_audit.py`, `negative_videos_manifest.json` v2.3 |
| **A2** | Cache | Có bài test chứng minh cache invalidation khi sửa file cùng kích thước | **PASS** | `test_08_cache_invalidation_on_change`, `test_09_cache_invalidation_same_size_different_sha` |
| **A3** | Manifest | Một nguồn dữ liệu chuẩn duy nhất, có Overwrite Protection Guard | **PASS** | `scripts/run_week4_complete_validation.py`, `test_28_manifest_no_downgrade_verified` |
| **A4** | Negative dataset | Exposure tính lại từ thực tế, có danh sách video đưa vào mẫu số | **PASS** | Bảng đối soát 3 mẫu số phơi nhiễm (6,9846h / 7,1312h / 7,2026h) |
| **A5** | Inference | 100% video hợp lệ được đối soát khớp số frame với inference summary | **PASS** | `runs/benchmarks/week4_final/inference_reconciliation_report.json` |
| **A6** | Sampling | Xác nhận policy lấy mẫu thời gian 5 FPS và phân biệt với native frame | **PASS** | Báo cáo nêu rõ 128.291 sampled frames / 929.939 stream frames |
| **A7** | Ground Truth | Các event ảnh hưởng đến metric được xác minh theo protocol | **PARTIAL** | 11/35 sự kiện `cross_verified`, 24 sự kiện `pending_verification` trong `gt_review_queue.csv` |
| **A8** | FAR | Định nghĩa event-level và exposure scope thống nhất | **PASS** | `runs/benchmarks/week4_final/far_report_verified.json` |
| **A9** | Poisson CI | Exact Poisson 95% CI được kiểm chứng độc lập bằng phân phối Gamma | **PASS** | `src/event_eval.py`, 7/7 unit tests trong `tests/test_poisson_ci.py` |
| **A10** | Operating Point | Được chọn trên validation, khóa cấu hình và test độc lập | **PARTIAL** | Cấu hình đã khóa, giữ trạng thái `PROVISIONAL` do A7 |
| **A11** | Provenance | Các artifact có thể truy xuất nguồn gốc SHA-256 và commit | **PASS** | `runs/benchmarks/week4_final/provenance_audit.json` |
| **A12** | Testing | Bộ test suite chạy thành công trên môi trường thực tế | **PASS** | 42/42 tests PASS trong `tests/test_video_audit_and_consistency.py` |
| **A13** | Documentation | Báo cáo đầy đủ, khoa học, không có mâu thuẫn số liệu | **PASS** | `docs/VIDEO_AUDIT_RELIABILITY_REPORT.md` & báo cáo này |
| **A14** | RTSP Live | Live RTSP benchmark camera thực tế | **NOT_TESTED** | Đánh dấu `BLOCKED_NO_LIVE_RTSP_SOURCE`, không tạo dữ liệu giả |

---

## 7. Bằng Chứng Dữ Liệu & Artifacts

- **Báo cáo kỹ thuật chi tiết:**
  - [`docs/VIDEO_AUDIT_RELIABILITY_REPORT.md`](file:///d:/DACN_NCKH/docs/VIDEO_AUDIT_RELIABILITY_REPORT.md)
  - [`docs/WEEK4_FINAL_VALIDATION_REPORT.md`](file:///d:/DACN_NCKH/docs/WEEK4_FINAL_VALIDATION_REPORT.md)
- **Dữ liệu cấu trúc JSON & CSV:**
  - [`data/negative_videos_manifest.json`](file:///d:/DACN_NCKH/data/negative_videos_manifest.json) (Manifest v2.3 chuẩn hóa)
  - [`runs/benchmarks/week4_final/far_report_verified.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/far_report_verified.json) (Báo cáo FAR & CI)
  - [`runs/benchmarks/week4_final/inference_reconciliation_report.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/inference_reconciliation_report.json) (Đối soát suy luận)
  - [`runs/benchmarks/week4_final/operating_point_audit.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/operating_point_audit.json) (Kiểm toán Operating Point)
  - [`runs/benchmarks/week4_final/provenance_audit.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/provenance_audit.json) (Truy xuất nguồn gốc)
  - [`runs/benchmarks/week4_final/rtsp_benchmark_status.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/rtsp_benchmark_status.json) (Trạng thái RTSP)
  - [`runs/benchmarks/week4_final/ground_truth/gt_review_queue.csv`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/ground_truth/gt_review_queue.csv) (Hàng đợi kiểm định 24 sự kiện)
- **Mã nguồn và Unit Tests:**
  - [`src/video_audit.py`](file:///d:/DACN_NCKH/src/video_audit.py)
  - [`tests/test_video_audit_and_consistency.py`](file:///d:/DACN_NCKH/tests/test_video_audit_and_consistency.py) (42 tests PASS)
  - [`tests/test_poisson_ci.py`](file:///d:/DACN_NCKH/tests/test_poisson_ci.py) (7 tests PASS)

---

## 8. Đề Xuất Các Bước Tiếp Theo (Chuyển Giao Tuần 5)

Để chính thức hoàn tất Tuần 4 và sẵn sàng chuyển giao sang **Tuần 5 — Domain Adaptation & Forgetting Measurement**:
1. **Phân công thẩm định viên thứ hai (Annotator 02):** Rà soát độc lập 24 sự kiện trong `runs/benchmarks/week4_final/ground_truth/gt_review_queue.csv` để chuyển sang `cross_verified`.
2. **Khóa chính thức Operating Point (OFFICIAL):** Sau khi 24 sự kiện được kiểm định, kích hoạt lệnh chuyển trạng thái Operating Point từ `PROVISIONAL` sang `OFFICIAL`.
3. **Chuẩn bị dữ liệu huấn luyện Tuần 5:** Sử dụng tập ảnh `datasets/vn_dataset` (3.121 ảnh đã gán nhãn và chia 60/20/20 độc lập theo video) và 20% tập D-Fire replay để triển khai 3 chiến lược thích nghi miền (Toàn bộ / Đóng băng backbone / Đóng băng + Replay 20% D-Fire) theo đúng Đề cương NCKH.

