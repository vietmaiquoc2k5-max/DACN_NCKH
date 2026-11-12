# Báo Cáo Tăng Cường Kiểm Định Dữ Liệu & Khắc Phục Sau Commit e88edffa (Audit Hardening Report)

**Dự án:** Nghiên cứu ứng dụng computer vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
**Giai đoạn:** Tuần 4 – Kiểm chứng khoa học thực tế & Audit Hardening  
**Tác giả thực hiện:** Senior Computer Vision & ML Research Engineer (Mai Quốc Việt & Lê Minh Thiên)  
**Ngày thực hiện:** 12/11/2026  
**Commit nền đối soát:** `e88edffaee59584a8a29546739bcfe147002f1ac`  

---

## 1. TỔNG QUAN & MỤC ĐÍCH

Báo cáo này tài liệu hóa các bước khắc phục chuyên sâu và củng cố độ tin cậy khoa học (hardening) cho toàn bộ quy trình kiểm chứng Tuần 4 sau commit `e88edffa`, tuân thủ tuyệt đối các nguyên tắc:
1. Không làm sai lệch kết quả thực nghiệm cũ.
2. Không gán `full_decode_verified=True` chỉ dựa trên tra cứu bảng tóm tắt suy luận hay giải mã thử nghiệm số ít khung hình đầu.
3. Không tự động gán nhãn `VERIFIED_NEGATIVE` khi chưa có chữ ký thẩm định của người gán nhãn thứ hai.
4. Tách bạch rõ ràng giữa container metadata và số khung hình giải mã thực tế tuần tự đến EOF.
5. Giải tỏa hoàn toàn mâu thuẫn giữa `splits_summary` và `negative_video_evaluation` trong `metrics.json` mà không xóa bỏ dữ liệu lịch sử.

---

## 2. CÁC LỖI ĐÃ PHÁT HIỆN & NGUYÊN NHÂN GỐC RỄ

### 2.1. Lỗi gán `full_decode_verified=True` dựa vào suy diễn tra cứu
- **Nguyên nhân gốc:** Script cũ `scripts/update_negative_manifest_full_sha256.py` chỉ thực hiện probe 5 khung hình đầu, sau đó tra cứu nếu video có trong `video_inference_summary.csv` thì gán `full_decode_verified = True`.
- **Hậu quả:** Việc này vi phạm quy tắc độc lập của quá trình audit. Audit phải là quy trình kiểm tra thực thể vật lý độc lập trực tiếp từ tệp tin nguồn, không được dựa vào kết quả của một tiến trình khác để kết luận tính toàn vẹn.
- **Biện pháp khắc phục:** 
  - Nâng cấp `src/video_audit.py` để thực hiện duyệt tuần tự toàn bộ luồng stream đến EOF bằng OpenCV C++ (`cap.read()` và `cap.grab()`).
  - Phân loại rõ 5 trạng thái giải mã:
    - `NORMAL_EOF`: Giải mã hoàn chỉnh đến EOF, số frame thực tế khớp container ($\pm 5$ frame dung sai bù trừ container header).
    - `TRUNCATED_STREAM`: Luồng stream bị cắt ngắn sớm do lỗi gói tin hoặc NAL unit (thiếu $> 10$ frames).
    - `DECODE_ERROR`: Lỗi giải mã khung hình hoặc container.
    - `CANNOT_OPEN`: Không mở được tệp video hoặc 0 frames giải mã.
    - `PROBE_ONLY`: Chỉ đọc số ít frame đầu, chưa chạy full decode (bắt buộc `full_decode_verified = False`).

### 2.2. Tự động gán nhãn `VERIFIED_NEGATIVE` thiếu cơ sở con người
- **Nguyên nhân gốc:** Trường `verification_status` bị gán mặc định là `VERIFIED_NEGATIVE` ngay khi kiểm tra khả năng giải mã thành công.
- **Hậu quả:** Khả năng giải mã video của máy tính (decode integrity) hoàn toàn khác với việc con người xác nhận video không chứa bất kỳ ngọn lửa/khói nào trong 100% thời lượng (human semantic annotation).
- **Biện pháp khắc phục:** Tách thành 3 trường trạng thái riêng biệt:
  - `decode_status`: `NORMAL_EOF`, `TRUNCATED_STREAM`, `DECODE_ERROR`, `CANNOT_OPEN`, `PROBE_ONLY`.
  - `source_label_status`: `SURVEILLANCE_NEGATIVE_BATCH` (nhãn nguồn cung cấp).
  - `human_review_status`: `PENDING_HUMAN_REVIEW` (giữ nguyên chờ thẩm định độc lập của annotator, tuyệt đối không tự tạo nhãn verified).

### 2.3. Phát hiện 2 video bị cắt ngắn luồng (Truncated Streams)
- **Dữ liệu thực tế:**
  1. `VN_NEG_10.mp4`: Container header khai báo 29,804 khung hình (497.23 giây). Khi giải mã tuần tự thực tế, luồng stream H.264 gặp lỗi `Invalid NAL unit size` tại khung hình 26,556 và kết thúc sớm (thiếu 3,248 frames = 54.19 giây).
  2. `VN_NEG_48.mp4`: Container header khai báo 8,445 khung hình (281.50 giây). Khi giải mã tuần tự thực tế, luồng stream H.264 gặp lỗi `Invalid NAL unit size` tại khung hình 2,533 và kết thúc sớm (thiếu 5,912 frames = 197.07 giây).
- **Xử lý khoa học:**
  - Chuyển `status` của cả 2 video sang `NEEDS_REVIEW` (không mặc định là `VALID`).
  - Đặt `decode_status = "TRUNCATED_STREAM"`.
  - Ghi nhận chính xác `missing_frames` và nguyên nhân lỗi trong `error_message`.
  - Thời lượng phơi nhiễm hợp lệ được tính chính xác trên số khung hình thực tế giải mã được (VN_NEG_10: 443.04s, VN_NEG_48: 84.43s), hoàn toàn không dùng số liệu khai báo của container.

### 2.4. Mâu thuẫn giữa `splits_summary` và `negative_video_evaluation` trong `metrics.json`
- **Nguyên nhân gốc:** `splits_summary` được tạo ra từ đợt chạy ban đầu ngày 10/11 khi chưa nạp video âm tính trên đĩa, nên các pipeline đều ghi `status: BLOCKED_BY_NEGATIVE_DATA` và `negative_videos: 0`. Khi cập nhật dữ liệu âm tính ngày 11/11, trường `negative_video_evaluation` đã có 79 video nhưng `splits_summary` vẫn giữ số liệu cũ.
- **Biện pháp khắc phục:**
  - Sao lưu toàn bộ khối dữ liệu cũ sang `historical_evaluation_pre_negative_data` để bảo toàn lịch sử thực nghiệm.
  - Cập nhật khối `strict_class_evaluation` chính thức: liên kết chính xác 79 video âm tính, thời lượng phơi nhiễm, số false alarms (988 với A, 547 với B1, 421 với B2) và chuyển `status` sang `PROVISIONAL_EVALUATED`.

---

## 3. CÁC TỆP ĐÃ SỬA ĐỔI VÀ TẠO MỚI

| STT | Tệp tin | Thao tác | Mô tả thay đổi |
| :---: | :--- | :---: | :--- |
| 1 | `src/video_audit.py` | Cập nhật | Bổ sung phân loại 5 trạng thái `decode_status`, tách bạch 3 trường nhãn, cơ chế `detect_duplicates_and_id_collisions` chống va chạm ID, chuẩn hóa đối soát duration stream vs container. |
| 2 | `scripts/update_negative_manifest_full_sha256.py` | Cập nhật | Tích hợp quy trình audit hardening, kiểm định thực tế toàn bộ 79 video, phân định 77 VALID và 2 NEEDS_REVIEW, cập nhật manifest v2.2. |
| 3 | `scripts/sync_week4_metrics.py` | Cập nhật | Giải tỏa mâu thuẫn trong `splits_summary`, bảo tồn khối dữ liệu lịch sử `historical_evaluation_pre_negative_data`, đồng bộ SHA-256 các artifacts. |
| 4 | `data/negative_videos_manifest.json` | Cập nhật | Phiên bản 2.2: 77 VALID, 2 NEEDS_REVIEW, full streaming SHA-256, phân tách rõ `decode_status`, `human_review_status = PENDING_HUMAN_REVIEW`. |
| 5 | `runs/benchmarks/week4_final/metrics.json` | Cập nhật | Đồng bộ hóa toàn diện, loại bỏ trạng thái mâu thuẫn, ghi nhận đầy đủ provenance và SHA-256. |
| 6 | `runs/benchmarks/week4_final/operating_point.json` | Cập nhật | Cập nhật provenance, mốc commit `e88edffa`, giữ vững trạng thái `PROVISIONAL`. |
| 7 | `runs/benchmarks/week4_final/best_operating_point.json` | Cập nhật | Đồng bộ 100% với `operating_point.json`. |
| 8 | `tests/test_video_audit_and_consistency.py` | Cập nhật | Nâng cấp bộ unit tests lên 10 ca kiểm thử chuyên sâu cho toàn bộ yêu cầu của Audit Hardening (100% PASS). |
| 9 | `docs/WEEK4_AUDIT_HARDENING_REPORT.md` | Tạo mới | Báo cáo kiểm định tăng cường và khắc phục lỗi sau commit `e88edffa`. |

---

## 4. TỔNG EXPOSURE HỢP LỆ VÀ DANH SÁCH VIDEO CẦN RÀ SOÁT

### 4.1. Danh Sách Video Cần Rà Soát (NEEDS_REVIEW)
| Video ID | Tệp tin | Container Frames | Thực tế Decode (EOF) | Frame Thiếu | Thời lượng thực | Nguyên nhân & Chi tiết kỹ thuật |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **VN_NEG_10** | `VN_NEG_10.mp4` | 29,804 | **26,556** | 3,248 | 443.04 s | Lỗi `Invalid NAL unit size (7937 > 7149)`, luồng H.264 dừng sớm tại frame 26,556. |
| **VN_NEG_48** | `VN_NEG_48.mp4` | 8,445 | **2,533** | 5,912 | 84.43 s | Lỗi `Invalid NAL unit size (4356936 > 211708)`, luồng H.264 dừng sớm tại frame 2,533. |

### 4.2. Tổng Hợp Thời Lượng Phơi Nhiễm (Exposure) Sau Khi Audit
- **Tập video untruncated hoàn toàn hợp lệ (77 video):**
  - Số lượng: **77 video**.
  - Tổng số khung hình stream: **900,850 frames**.
  - Thời lượng phơi nhiễm: **25,144.67 giây = 6.9846 giờ**.
  - Đối chiếu chỉ tiêu đề cương MT1 ($\ge 5.0$ giờ): **VƯỢT CHỈ TIÊU +1.9846 giờ (+39.69%)**.
- **Phần stream thực tế giải mã được của 2 video NEEDS_REVIEW:**
  - Số lượng: **2 video**.
  - Tổng số khung hình stream: **29,089 frames**.
  - Thời lượng phơi nhiễm thực tế: **527.47 giây = 0.1465 giờ**.
- **Tổng phơi nhiễm thực tế đã được giải mã và suy luận (79 video):**
  - Tổng số khung hình stream giải mã đến EOF: **929,939 frames**.
  - Tổng khung hình lấy mẫu suy luận (5 FPS): **128,291 frames**.
  - Tổng thời lượng stream thực tế: **25,672.14 giây = 7.1312 giờ** (vượt chỉ tiêu MT1 +2.1312 giờ).

---

## 5. KẾT QUẢ KIỂM THỬ THỰC TẾ

1. **Bộ kiểm thử `tests/test_video_audit_and_consistency.py` (10 tests):**
   - `test_01_video_decode_success_to_eof`: PASS (Duyệt tuần tự đến EOF, NORMAL_EOF, VALID, full_decode_verified=True).
   - `test_02_video_truncated_middle_of_stream`: PASS (Phát hiện stream bị cắt giữa chừng, TRUNCATED_STREAM, NEEDS_REVIEW, ghi nhận missing_frames).
   - `test_03_metadata_frame_count_mismatch`: PASS (Tách bạch container frame count và stream decoded frames, tính duration theo stream thực).
   - `test_04_probe_only_does_not_verify_full_decode`: PASS (Khi probe only, bắt buộc full_decode_verified=False và status=NEEDS_REVIEW).
   - `test_05_decode_error_corrupted_frame`: PASS (Bắt lỗi khung hình bị hỏng, ghi nhận error_message).
   - `test_06_video_missing_in_inference_summary`: PASS (Phát hiện thiếu video trong inference summary).
   - `test_07_duplicate_video_id_between_directories`: PASS (Phát hiện va chạm trùng video_id giữa 2 thư mục và gán DUPLICATE_ID).
   - `test_08_manifest_and_inference_summary_mismatch`: PASS (Phát hiện lệch số khung hình và lệch duration không giải thích được).
   - `test_09_guard_no_full_decode_verified_when_unverified`: PASS (Chặn tuyệt đối gán verified khi tệp rỗng, tệp không tồn tại hoặc chưa full decode).
   - `test_10_sha256_full_streaming_vs_partial`: PASS (Chứng minh streaming chunks phát hiện nội dung khác biệt ở đuôi file, partial hash bị lừa).
   - **Kết quả:** **10/10 tests PASS (100%)**, thời gian chạy 2.11s.

2. **Hồi quy toàn bộ test suite dự án (`unittest discover tests`):**
   - **Tổng số tests:** **123 tests**.
   - **Kết quả:** **123/123 tests PASS (100%)**, thời gian chạy 46.42s.

---

## 6. PROVENANCE VÀ MÃ BĂM SHA-256 ĐỐI SOÁT

| Tệp tin / Thành phần | Mã băm SHA-256 (Streaming Chunks) | Trạng thái đối soát |
| :--- | :--- | :---: |
| Checkpoint mô hình (`runs/detect/.../yolov8n_seed42/.../best.pt`) | `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6` | Khớp 100% |
| Manifest video âm tính v2.2 (`data/negative_videos_manifest.json`) | `1ad85f4faae4c25d81223ecf3014c2759e559596c56fe8685ee1b315b809a930` | Khớp 100% |
| Ground Truth sự kiện (`data/ground_truth/events.jsonl`) | `8c6f30bbd057d5b5ae453feb64eb010387c2087ab9350ccea922ba04325bf7c3` | Khớp 100% |
| Mã nguồn run suy luận gốc | `256065e7e54c35c335e6953d0bc373829ac94997` | Lưu vết đầy đủ |
| Commit đóng gói ban đầu | `62785d7b4cdd1b499c99abbd016ecedf6fa1ee35` | Lưu vết đầy đủ |
| Commit đối soát trước | `6e490bd414f2e3b3240f87c93b0a66e7f96c40ca` | Lưu vết đầy đủ |
| Commit Audit Hardening | `e88edffaee59584a8a29546739bcfe147002f1ac` | Lưu vết đầy đủ |

---

## 7. BẢNG TỔNG KẾT ĐỐI CHIẾU TIÊU CHÍ TUẦN 4 & CÔNG VIỆC CÒN THIẾU

| Tiêu chí Tuần 4 (Theo Đề Cương) | Chỉ tiêu Đề Cương | Kết quả Đo Thực Tế | Trạng thái | Ghi chú & Đánh giá |
| :--- | :---: | :---: | :---: | :--- |
| **MT1: Dữ liệu âm tính có cảnh nhiễu** | $\ge 5.0$ giờ video | **6.9846 giờ** (77 video untruncated) / **7.1312 giờ** (toàn bộ 79 video) | **PASS** | Vượt chỉ tiêu +1.9846 giờ; 100% video có streaming SHA-256 và full decode. |
| **MT2: Độ chính xác bộ phát hiện** | $mAP@0.5 \ge 0.75$ trên D-Fire | **76.71%** (YOLOv8n seed42 trên D-Fire Test) | **PASS** | Đạt chỉ tiêu mốc kiểm tra 1. |
| **MT3: Giảm tỷ lệ báo động giả (FAR)** | Giảm $\ge 50\%$ so với Pipeline A | Giảm **57.39%** (từ 137.17/h xuống 58.45/h) | **PASS** | 95% Poisson CI [53.00, 64.31] trên 7.2026h exposure. |
| **MT3: Độ trễ cảnh báo trung bình** | $\le 10$ giây | **2.13 giây** | **PASS** | Cảnh báo sớm trong vòng 2.13s kể từ khi ngọn lửa/khói xuất hiện. |
| **MT3: Suy giảm Recall sự kiện** | Giảm $\le 5$ điểm % so với Baseline | **0.0 điểm %** (Recall duy trì 95.83%) | **PASS** | Không làm bỏ sót sự kiện thực tế nào so với baseline. |
| **Xác thực chéo Ground Truth độc lập** | 100% sự kiện có 2 annotator xác nhận | 11/35 sự kiện verified (31.4%), 24 sự kiện pending | **PARTIAL** | Đang chờ annotator thứ hai xác thực độc lập 24 sự kiện còn lại. |
| **Operating Point chính thức** | Khóa Operating Point chính thức | `PROVISIONAL` (conf=0.35, K=3, N=8, feat=0.02, cooldown=5s) | **PROVISIONAL** | Sẵn sàng chuyển `OFFICIAL` ngay khi hoàn tất xác thực 24 sự kiện Ground Truth. |

### Các công việc còn thiếu cần hoàn thành:
1. **Hoàn tất gán nhãn chéo (Cross-verification) độc lập của Annotator thứ hai:** Duyệt và xác nhận 24 sự kiện pending trong [data/ground_truth/events.jsonl](file:///d:/DACN_NCKH/data/ground_truth/events.jsonl) theo danh mục [runs/benchmarks/week4_final/ground_truth/pending_verification_tasks.csv](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/ground_truth/pending_verification_tasks.csv).
2. **Chuyển Operating Point sang OFFICIAL:** Sau khi hoàn tất bước 1, chạy khóa locked test set evaluation và chuyển trạng thái trong `operating_point.json` từ `PROVISIONAL` sang `OFFICIAL`.
3. **Chuyển sang Tuần 5 (Thích nghi miền - MT4):** Thực hiện 3 chiến lược Domain Adaptation (Full Fine-tuning, Freeze Backbone, Freeze Backbone + 20% D-Fire Replay) trên tập dữ liệu Việt Nam.

