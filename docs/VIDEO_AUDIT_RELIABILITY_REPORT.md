# BÁO CÁO ĐỘ TIN CẬY VÀ KIỂM ĐỊNH TOÀN VẸN VIDEO AUDIT TUẦN 4
**Tài liệu:** `docs/VIDEO_AUDIT_RELIABILITY_REPORT.md`  
**Dự án:** Computer Vision-based Fire and Smoke Detection and Early Warning from Surveillance Camera Footage  
**Thời gian lập báo cáo:** 12/11/2026 16:15 UTC+7  
**Commit cơ sở kiểm định:** `c95907f6b8cf12cc02563dfaab209f304dbc3259`  
**Môi trường:** Python 3.10 / OpenCV 4.x / PyTorch 2.5.1 / Windows 11  

---

## 1. Tóm Tắt Tổng Quan (Executive Summary)

Báo cáo này công bố kết quả rà soát, tái cấu trúc và củng cố toàn diện (Hardening) hệ thống kiểm định tính toàn vẹn của tệp video (`src/video_audit.py`), bộ nhớ đệm giải mã (`VideoDecodeCacheManager`), danh mục dữ liệu âm tính (`data/negative_videos_manifest.json`), và đối soát 100% với nhật ký suy luận thực nghiệm (`runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full/video_inference_summary.csv`).

Toàn bộ các lỗ hổng tiềm ẩn đã được khắc phục triệt để:
- Không còn việc gán `full_decode_verified=True` khi chỉ probe một số khung hình đầu.
- Tính toán SHA-256 hoàn toàn theo cơ chế streaming chunks (1MB), phân biệt chính xác tệp bị sửa đổi phần đuôi hoặc tệp cùng kích thước nhưng khác nội dung.
- Phân biệt minh thị 3 chiều trạng thái: kỹ thuật luồng stream (`decode_status`), nguồn gốc thu thập (`source_label_status`), và kiểm duyệt ngữ nghĩa độc lập của con người (`human_review_status`).
- Thiết lập cơ chế bảo vệ chống ghi đè hạ cấp (Overwrite Protection Guard) ngăn script kiểm thử probe cũ vô hiệu hóa kết quả kiểm định chi tiết.
- Hoàn tất bộ kiểm thử 42 chỉ tiêu tự động (100% PASS) bao quát toàn bộ các ca kiểm thử băm, cache, giải mã, manifest, FAR Poisson CI và tính tương thích ngược.

---

## 2. Commit Được Kiểm Tra & Hiện Trạng Repository

- **Commit kiểm tra:** `c95907f6b8cf12cc02563dfaab209f304dbc3259`
- **Nhánh Git:** `main`
- **Trọng số mô hình chính thức:** `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt` (SHA-256: `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`)
- **Tập video âm tính:** 79 video trên đĩa (7.30 GB, 929.939 frames stream thực tế) phân bổ tại `datasets/negative_videos` (65 video) và `datasets/test_videos/neg` (14 video).

---

## 3. Danh Sách Lỗi Phát Hiện & Phân Loại Mức Độ

| Mã lỗi | Mức độ | Thành phần | Mô tả vấn đề | Nguyên nhân gốc rễ | Biện pháp xử lý |
|---|---|---|---|---|---|
| **BUG-01** | **CRITICAL** | `scripts/run_week4_complete_validation.py` | Hàm legacy `audit_negative_videos_dir` chỉ đọc 5 frame đầu (probe), tự động gán `full_decoded_frames = frame_count` và `verification_status = VERIFIED_NEGATIVE`, có thể ghi đè manifest chuẩn. | Code kế thừa từ Tuần 3 giả định video mở được là âm tính hợp lệ và metadata container là chính xác. | Tái cấu trúc hàm gọi trực tiếp `audit_single_video()`, thêm Overwrite Protection Guard chặn ghi đè nếu manifest đã `version >= 2.0` và đã `full_stream_decode_verified`. |
| **BUG-02** | **CRITICAL** | `src/video_audit.py` & `.video_decode_cache.json` | Cache khóa theo `video_id` đơn lẻ và chỉ kiểm tra `file_size_bytes`, có nguy cơ va chạm giữa các thư mục và không phát hiện file bị sửa cùng dung lượng. | Thiết kế cache sơ khai chưa tính đến trường hợp file trùng tên hoặc tấn công/sửa đổi cùng kích thước. | Chuyển khóa cache sang `relative_path`, bắt buộc thẩm định full streaming SHA-256 trước khi trả về cache hit, thiết lập `schema_version = 2`. |
| **BUG-03** | **HIGH** | `src/video_audit.py` | Lớp `DecodeStatus(str, Enum)` trả về chuỗi `DecodeStatus.NORMAL_EOF` trong Python thay vì `NORMAL_EOF`. | Cách thức chuyển đổi chuỗi mặc định của `Enum` trong một số phiên bản Python runtime. | Cài đặt tường minh `__str__(self) -> str: return self.value` đảm bảo tương thích 100% với chuỗi ký tự. |
| **BUG-04** | **HIGH** | `scripts/update_negative_manifest_full_sha256.py` | Script nạp trực tiếp file cache thô thay vì sử dụng đối tượng `VideoDecodeCacheManager`. | Đồng bộ hóa cache chưa tập trung qua một module quản lý chuẩn. | Chuyển toàn bộ pipeline sang `VideoDecodeCacheManager`, tự động lưu cache an toàn sau khi duyệt xong toàn bộ tập dữ liệu. |
| **BUG-05** | **MEDIUM** | `src/event_eval.py` | `exact_poisson_ci` trả về `(0.0, 0.0)` khi `exposure_hours <= 0` thay vì ném ngoại lệ toán học. | Thiếu ràng buộc kiểm tra mẫu số thời gian phơi nhiễm. | Cập nhật ném `ValueError` rõ ràng khi `exposure_hours <= 0`. |
| **BUG-06** | **MEDIUM** | `data/negative_videos_manifest.json` | Thiếu sự phân biệt tường minh giữa 3 định nghĩa thời gian phơi nhiễm (7.2026h, 7.1312h, 6.9846h). | Báo cáo ban đầu chỉ nêu một con số phơi nhiễm container. | Bổ sung đầy đủ 3 trường phơi nhiễm vào manifest schema v2.3 kèm lý giải toán học và kỹ thuật bitstream. |

---

## 4. Chính Sách Kỹ Thuật (Engineering Policies)

### 4.1. Chính Sách Dung Sai Container (Explicit Container Tolerance Policy)
Khác biệt giữa số khung hình do Container Metadata khai báo (`reported_frame_count` từ `CAP_PROP_FRAME_COUNT`) và số khung hình thực tế giải mã được đến EOF (`full_decoded_frames`):
$$\Delta = \text{reported\_frame\_count} - \text{full\_decoded\_frames}$$

1. **$\Delta = 0$:** `NORMAL_EOF`, trạng thái `VALID`. Khớp tuyệt đối 100%.
2. **$1 \le \Delta \le 5$ (`CONTAINER_PADDING_TOLERANCE`):** `NORMAL_EOF`, trạng thái `VALID`. Bù trừ sai số làm tròn chỉ mục container MP4/AVI (moov atom rounding, B-frame PTS/DTS alignment theo chuẩn ISO/IEC 14496-12).
3. **$6 \le \Delta \le 10$ (`CONTAINER_BORDERLINE_TOLERANCE`):** `NORMAL_EOF`, trạng thái `VALID` kèm cảnh báo ghi log `WARNING`. Bù trừ xả bộ đệm tái sắp xếp khung hình B (B-frame reorder buffer flush) và GOP biến thiên ở cuối tệp.
4. **$\Delta > 10$ (`TRUNCATED_STREAM_THRESHOLD`):** `TRUNCATED_STREAM`, trạng thái `NEEDS_REVIEW`. Bắt buộc coi là luồng bị ngắt sớm do lỗi packet bitstream NAL.
5. **$\text{reported\_frame\_count} = 0$:** `NEEDS_REVIEW`. Thiếu dữ liệu metadata container, không thể đối chiếu an toàn.

### 4.2. Chính Sách Thẩm Định Cache (Cache Validation Policy)
Mỗi truy vấn cache phải trải qua 5 bước nghiêm ngặt:
1. Tệp phải tồn tại trên đĩa vật lý.
2. Khóa cache phải khớp `relative_path` chuẩn hóa (forward slash).
3. `schema_version` của bản ghi phải $\ge 2$ và có `sha256` hợp lệ (không chứa tiền tố lỗi `err_`). Nếu không, lập tức `CACHE_INVALIDATED`.
4. `file_size_bytes` phải khớp tuyệt đối với `stat().st_size`.
5. Nếu `validate_sha256=True`, tính lại streaming SHA-256 toàn phần. Nếu mã băm khác nhau (phát hiện sửa đổi cùng kích thước), lập tức `CACHE_INVALIDATED`.

### 4.3. Ngữ Nghĩa Số Lượng Khung Hình (Frame Count Semantics)
- `reported_frame_count`: Số khung hình khai báo bởi container header.
- `full_decoded_frames` (`decoded_frame_count`): Số khung hình thực tế giải mã thành công đến điểm kết thúc. Đây là Ground Truth kỹ thuật duy nhất.
- `audit_probe_frames`: Số khung hình đọc thăm dò khởi động ban đầu (mặc định 5).
- `processed_sampled_frames`: Số khung hình trích xuất đưa vào mô hình YOLO theo chính sách lấy mẫu thời gian (5 FPS).

---

## 5. Đối Soát Phơi Nhiễm & Trạng Thái Video Bất Thường

### 5.1. Bảng Đối Soát 3 Mẫu Số Phơi Nhiễm
| Mẫu số phơi nhiễm | Giá trị (Giây) | Giá trị (Giờ) | Phạm vi áp dụng | FAR Pipeline B2 (421 cảnh báo) |
|---|---:|---:|---|---:|
| **Canonical Benchmark Exposure** | 25.929,53 s | **7,2026 h** | Mẫu số chuẩn hóa của đợt chạy full inference 79 video | **58,4511 / giờ** (95% CI: [53,00, 64,33]) |
| **Physical Stream Decoded Exposure** | 25.672,14 s | **7,1312 h** | Thời lượng luồng stream thực tế giải mã được đến điểm ngắt | **59,0361 / giờ** (95% CI: [53,53, 64,97]) |
| **Untruncated Strictly Valid Exposure** | 25.144,67 s | **6,9846 h** | Chỉ tính 77 video hoàn toàn không bị ngắt luồng | **60,2755 / giờ** (95% CI: [54,65, 66,34]) |

Cả 3 mẫu số phơi nhiễm đều vượt xa chỉ tiêu $\ge 5,0$ giờ quy định trong Mục tiêu MT1 của Đề cương NCKH. Mức giảm báo giả của Pipeline B2 so với Baseline A (988 báo động) đạt **57,39%**, vượt chỉ tiêu $\ge 50\%$.

### 5.2. Hồ Sơ Kỹ Thuật 2 Video Truncated (Trạng Thái NEEDS_REVIEW)
1. **`VN_NEG_10` (`datasets/negative_videos/VN_NEG_10.mp4`):**
   - Container metadata: 29.804 frames (497,23 giây).
   - Thực tế giải mã: 26.556 frames (443,04 giây).
   - Số frame thiếu: 3.248 frames (54,19 giây) do lỗi hỏng khối NAL unit ở cuối tệp.
2. **`VN_NEG_48` (`datasets/negative_videos/VN_NEG_48.mp4`):**
   - Container metadata: 8.445 frames (281,50 giây).
   - Thực tế giải mã: 2.533 frames (84,43 giây).
   - Số frame thiếu: 5.912 frames (197,07 giây) do lỗi packet H.264 stream kết thúc sớm.

---

## 6. Kết Quả Kiểm Thử Tự Động (Test Coverage & Results)

Đã chạy toàn diện 42 unit test và integration test độc lập (`tests/test_video_audit_and_consistency.py`):
- **Video Hash Tests (01-06):** 6/6 PASS.
- **Cache Tests (07-14):** 8/8 PASS.
- **Decode Tests (15-23):** 9/9 PASS.
- **Manifest Tests (24-30):** 7/7 PASS.
- **FAR & Exact Poisson CI Tests (31-37):** 7/7 PASS.
- **Integration Tests (38-42):** 5/5 PASS.
- **Tổng kết:** **42/42 tests PASS (100%)**.

---

## 7. Các Hạn Chế Còn Tồn Tại & Kiến Nghị

1. **Ground Truth còn 24 sự kiện pending:** Trong tổng số 35 sự kiện dương tính trên 24 video, hiện có 11 sự kiện đã được kiểm định chéo (`cross_verified`), 24 sự kiện đang chờ annotator thứ hai đối soát trong tệp hàng đợi `runs/benchmarks/week4_final/ground_truth/gt_review_queue.csv`.
2. **Operating Point duy trì PROVISIONAL:** Do các sự kiện Ground Truth chưa đạt 100% kiểm định chéo kép độc lập, điểm vận hành được khóa chặt ở trạng thái `PROVISIONAL` nhằm bảo đảm tính trung thực khoa học tuyệt đối.
3. **RTSP Live Benchmark:** Đánh dấu `NOT_TESTED / BLOCKED_NO_LIVE_RTSP_SOURCE` do trạm kiểm định hiện tại chưa kết nối camera IP phần cứng trực tiếp. Pipeline đa luồng sẵn sàng khi có nguồn phát MediaMTX.

---

## 8. Lệnh Tái Hiện Thực Nghiệm (Reproduction Commands)

```powershell
# Chạy bộ 42 bài kiểm thử tính toàn vẹn video audit và cache:
python -m unittest tests/test_video_audit_and_consistency.py -v

# Chạy kiểm thử khoảng tin cậy Poisson chính xác:
python -m unittest tests/test_poisson_ci.py -v

# Cập nhật manifest và đối soát với dữ liệu thực nghiệm:
python scripts/update_negative_manifest_full_sha256.py
```

