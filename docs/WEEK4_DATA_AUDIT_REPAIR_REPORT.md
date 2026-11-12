# Báo Cáo Sửa Đổi Kiểm Định Dữ Liệu, Giải Mã Toàn Bộ, SHA-256 & Tính Nhất Quán Artifacts Tuần 4

**Dự án:** Nghiên cứu ứng dụng computer vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
**Giai đoạn:** Tuần 4 – Kiểm chứng khoa học thực tế & Xác thực dữ liệu âm tính  
**Tác giả thực hiện:** Senior Computer Vision & ML Research Engineer (Mai Quốc Việt & Lê Minh Thiên)  
**Ngày thực hiện:** 12/11/2026  
**Commit nền đối soát:** `6e490bd414f2e3b3240f87c93b0a66e7f96c40ca`  

---

## 1. TỔNG QUAN VÀ MỤC ĐÍCH

Báo cáo này tài liệu hóa chi tiết toàn bộ quá trình rà soát, phát hiện sai lệch và khắc phục dứt điểm 6 vấn đề tồn đọng trong quy trình kiểm chứng dữ liệu thực nghiệm Tuần 4 theo nguyên tắc liêm chính học thuật:
1. **Script audit cũ chỉ đọc 2 MB đầu tệp** để tính SHA-256 dẫn đến nguy cơ xung đột mã băm và không bảo đảm tính toàn vẹn.
2. **Script audit cũ chỉ probe 5 khung hình đầu** mà không duyệt tuần tự đến EOF, dẫn đến hiểu nhầm giữa số khung hình khai báo trong metadata container (`reported_frame_count`) và số khung hình giải mã thực tế (`full_decoded_frames`).
3. **Trường metadata `decoded_frames` gây nhầm lẫn** giữa số khung hình probe thử nghiệm và số khung hình stream thực tế.
4. **`metrics.json` bị lệch pha thời gian** (chứa số liệu cũ ngày 10/11 khi chưa nạp video âm tính).
5. **Chưa phân định rạch ròi chế độ suy luận**: *Full-Video Temporal Sampling Inference at 5 FPS* và *Exhaustive Frame-by-Frame Inference at Native FPS*.
6. **Provenance chưa tách bạch**: giữa commit chứa mã nguồn thực thi run (`256065e7`), commit đóng gói kết quả ban đầu (`62785d7b`), và commit sửa đổi kiểm định (`6e490bd4`).

---

## 2. CHI TIẾT CÁC LỖI ĐÃ PHÁT HIỆN & NGUYÊN NHÂN GỐC RỄ

### 2.1. Lỗi mã băm SHA-256 đọc từng phần (Partial Hash)
- **Hiện trạng trước sửa đổi:** Trong `scripts/run_week4_complete_validation.py` (dòng 91–98), mã băm SHA-256 chỉ đọc đúng $2 \times 1024 \times 1024$ bytes đầu tiên của file:
  ```python
  with open(vpath, "rb") as f:
      h.update(f.read(2 * 1024 * 1024))
  sha256_val = h.hexdigest()
  ```
- **Hậu quả:** Đối với các file video dung lượng lớn (ví dụ `VN_NEG_01.mp4` dung lượng 1.05 GB), giá trị mã băm lưu trong manifest (`cb5174b2...`) thực chất chỉ là băm của 2 MB đầu, hoàn toàn không đại diện cho 1.05 GB dữ liệu.
- **Khắc phục:** Viết module `src/video_audit.py` với hàm `compute_streaming_sha256(filepath, chunk_size=1MB)` duyệt toàn bộ stream dữ liệu theo block 1 MB vào SHA-256. Mã băm thực tế của `VN_NEG_01.mp4` được chuẩn hóa thành `2a149fd6839b87956025e11d30743f8bb324f2b6dc8b43162d59bf5e207c874a`.

### 2.2. Lỗi lệch số khung hình giữa Container Header và Luồng Thực Tế (Stream EOF)
- **Hiện trạng trước sửa đổi:** Container MP4/AVI lưu trường `CAP_PROP_FRAME_COUNT` ở header. Tuy nhiên, luồng nén H.264 thực tế có thể kết thúc sớm do lỗi NAL unit, rớt gói tin hoặc EOF container lệch.
- **Bằng chứng thực nghiệm điển hình:**
  - `VN_NEG_10.mp4`: Container header báo `29,804` frames. Nhưng khi giải mã tuần tự qua OpenCV, tại frame `26,556` xuất hiện lỗi:
    `[h264] Invalid NAL unit size (7937 > 7149). Error splitting the input into NAL units.` và dừng giải mã.
  - Script audit cũ chỉ probe 5 frame nên gán mù quáng `decoded_frames = 29,804`.
  - Trong khi đó, run suy luận thực tế `run_20261111_233035_neg_full` đọc tuần tự đến EOF ghi nhận chính xác `total_decoded_frames = 26,556`!
  - Phát hiện tổng cộng **60/79 video** có sự chênh lệch nhỏ giữa container header và số frame thực giải mã đến EOF.
- **Khắc phục:** Chuẩn hóa tách bạch 2 trường:
  - `reported_frame_count`: Số frame container khai báo.
  - `full_decoded_frames`: Số frame thực sự giải mã thành công tuần tự đến EOF (Tổng = 929,939 frames).
  - `full_decode_errors`: Số frame lệch do luồng ngắt sớm.
  - `full_decode_verified`: Đánh dấu `True` có kiểm chứng bởi run thực tế.

### 2.3. Lỗi bất đồng bộ trong `runs/benchmarks/week4_final/metrics.json`
- **Hiện trạng:** Tệp `runs/benchmarks/week4_final/metrics.json` lưu mốc thời gian `2026-11-10 02:52:26` với `available_negative_videos: 0`, trong khi báo cáo và run thực nghiệm đã hoàn tất 79 video (7.2026 giờ) vào đêm 11/11.
- **Khắc phục:** Chạy script `scripts/sync_week4_metrics.py` đồng bộ hóa toàn diện dữ liệu âm tính, 3 Pipeline (A, B1, B2), khoảng tin cậy Poisson và xác lập trạng thái minh bạch `PARTIAL (PENDING_24_GT_VERIFICATION)`.

---

## 3. CÁC TỆP ĐÃ TẠO MỚI VÀ SỬA ĐỔI

| STT | Tệp | Hành động | Nội dung thay đổi |
| :---: | :--- | :---: | :--- |
| 1 | `src/video_audit.py` | Tạo mới | Cung cấp `compute_streaming_sha256`, `audit_single_video`, `detect_duplicates_by_hash`, `verify_manifest_vs_inference`. |
| 2 | `scripts/update_negative_manifest_full_sha256.py` | Tạo mới | Quét toàn bộ 79 video, tính full streaming SHA-256, đối soát 100% với `video_inference_summary.csv` và cập nhật manifest. |
| 3 | `scripts/sync_week4_metrics.py` | Tạo mới | Đồng bộ hóa `runs/benchmarks/week4_final/metrics.json`, `operating_point.json`, `best_operating_point.json` với đầy đủ provenance. |
| 4 | `tests/test_video_audit_and_consistency.py` | Tạo mới | Bộ unit test tự động gồm 8 ca kiểm thử chuyên sâu cho toàn bộ quy trình audit video và đồng bộ. |
| 5 | `scripts/run_week4_complete_validation.py` | Sửa đổi | Thay thế cơ chế partial 2MB hash bằng `compute_streaming_sha256`, chuẩn hóa các trường metadata. |
| 6 | `data/negative_videos_manifest.json` | Cập nhật | 79 video có full streaming SHA-256, trường `reported_frame_count`, `full_decoded_frames`, `audit_probe_frames`, `full_decode_verified`. |
| 7 | `runs/benchmarks/week4_final/metrics.json` | Cập nhật | Đồng bộ đầy đủ 79 video âm tính, 7.2026 giờ exposure, FAR 3 pipeline và provenance. |
| 8 | `runs/benchmarks/week4_final/operating_point.json` | Cập nhật | Bổ sung mã băm manifest SHA-256 mới và làm rõ chế độ Full-Video Temporal Sampling at 5 FPS. |
| 9 | `runs/benchmarks/week4_final/best_operating_point.json` | Cập nhật | Đồng bộ hoàn toàn với `operating_point.json`. |

---

## 4. KẾT QUẢ ĐỐI SOÁT VÀ XÁC MINH DỮ LIỆU THỰC TẾ

### 4.1. Tập Dữ Liệu Video Âm Tính Thực Tế
- **Tổng số tệp quét:** 79 tệp video (`.mp4`, `.avi`).
- **Tổng dung lượng đĩa:** 7.30 GB (tốc độ đọc kiểm tra trung bình: 578.8 MB/s).
- **Mã băm SHA-256:** 100% tệp (79/79) được tính toán bằng streaming chunks (1 MB), xác nhận không có bất kỳ tệp nào bị đọc thiếu.
- **Trùng lặp (Duplicate):** 0 tệp trùng lặp (79 hash duy nhất).
- **Video lỗi (Corrupted):** 0 tệp lỗi mở luồng.
- **Video hợp lệ:** 79 tệp (100%).
- **Tổng khung hình luồng stream giải mã đến EOF:** **929,939 khung hình**.
- **Tổng thời lượng exposure hợp lệ:** **25,929.53 giây = 7.2026 giờ**.
  - Chỉ tiêu MT1/MT3 đề cương: $\ge 5.0$ giờ.
  - Vượt chỉ tiêu: **+2.2026 giờ** (Thặng dư +44.05%).

### 4.2. Phân Định Rõ Ràng Hai Chế Độ Suy Luận
Hệ thống xác định rạch ròi quy cách kỹ thuật của lần chạy suy luận `run_20261111_233035_neg_full`:
- **Chế độ thực thi:** **Full-Video Temporal Sampling Inference at 5 FPS**.
- **Giải thích kỹ thuật:** 
  - Toàn bộ luồng video (100% thời lượng từ giây 0 đến giây cuối cùng) được OpenCV duyệt tuần tự qua từng khung hình đến EOF (`cap.read()` và `cap.grab()`). Tổng số khung hình stream được giải mã/duyệt là **929,939 khung hình**.
  - Bộ dò YOLOv8n suy luận định kỳ trên các khung hình lấy mẫu tại tần số 5 FPS (`step = round(native_fps / 5.0)`). Tổng số khung hình đưa qua YOLO và ByteTrack là **128,291 khung hình**.
  - Bộ kiểm định thời gian `TemporalVerifier` duy trì cửa sổ trượt $K=3, N=8$ trên luồng 5 FPS này, bảo đảm độ phủ thời gian 100% trên toàn bộ 7.2026 giờ exposure mà không bỏ sót bất kỳ phân đoạn thời gian nào.
- **Cam kết khoa học:** Không tuyên bố là *Exhaustive Frame-by-Frame Inference at Native FPS* (vốn không khả thi trong giám sát camera thực tế 24/7 do lãng phí tài nguyên tính toán đối với các khung hình trùng lặp ở 60 FPS).

### 4.3. Đối Soát 1:1 Giữa Manifest và Inference Summary
Sử dụng hàm kiểm tra tự động `verify_manifest_vs_inference()`:
- Số video trong Manifest: 79 video.
- Số video trong `video_inference_summary.csv`: 79 video.
- Missing in inference: 0.
- Extra in inference: 0.
- Lệch số khung hình giải mã (`full_decoded_frames` vs `total_decoded_frames`): **0 khung hình**.
- Lệch thời lượng exposure (`duration_seconds` vs `actual_duration_sec`): **0.00 giây**.
- **Kết quả đối soát:** `is_perfect_match: True` (Khớp tuyệt đối 100%).

---

## 5. KẾT QUẢ KIỂM THỬ TỰ ĐỘNG (UNIT TESTS)

Bộ kiểm thử `tests/test_video_audit_and_consistency.py` đã thực thi thành công 8/8 ca kiểm thử:
1. `test_01_sha256_full_streaming_vs_partial`: PASS (Chứng minh hash 2MB đầu bị lừa, streaming hash phát hiện sai khác ở đuôi).
2. `test_02_video_full_decode_to_eof`: PASS (Giải mã 25/25 frames, gán `full_decode_verified=True`, đếm chính xác frames).
3. `test_03_video_decode_error_middle_of_stream`: PASS (Xử lý mượt mà khi video bị cắt ngắn giữa stream, đếm đúng frame thực tế giải mã).
4. `test_04_video_probe_only_does_not_verify_full_decode`: PASS (Khi `full_decode=False`, bảo đảm `full_decode_verified=False`).
5. `test_05_video_duplicate_detection_by_full_hash`: PASS (Phát hiện bản sao chính xác và gán `duplicate_of`).
6. `test_06_metadata_frame_count_mismatch`: PASS (Ghi nhận cả container count và stream count, tính duration theo stream count).
7. `test_07_manifest_and_inference_summary_mismatch`: PASS (Phát hiện chính xác thiếu video hoặc lệch số frame).
8. `test_08_guard_no_full_decode_verified_when_unverified`: PASS (Chặn gán verified khi file lỗi, file rỗng hoặc chỉ probe).

Chạy toàn bộ test suite của toàn dự án (`unittest discover tests`):
- **Tổng số tests:** **121 unit & system tests**.
- **Kết quả:** **121/121 PASS (100%)**, thời gian chạy 37.11s.

---

## 6. PROVENANCE ĐẦY ĐỦ VÀ MINH BẠCH

Để đảm bảo tính lặp lại (reproducibility) trong nghiên cứu khoa học, bảng sau đây ghi nhận chính xác dấu vết xuất xứ (provenance) của toàn bộ dữ liệu:

| Thuộc tính Provenance | Giá trị Xác Nhận |
| :--- | :--- |
| **Run ID Suy Luận Âm Tính** | `run_20261111_233035_neg_full` |
| **Source Run Commit SHA** | `256065e7e54c35c335e6953d0bc373829ac94997` (commit tại thời điểm chạy thực nghiệm) |
| **Initial Artifact Commit SHA** | `62785d7b4cdd1b499c99abbd016ecedf6fa1ee35` (commit đóng gói kết quả ban đầu) |
| **Audit Repair Base Commit SHA** | `6e490bd414f2e3b3240f87c93b0a66e7f96c40ca` (commit nền kiểm định và sửa đổi) |
| **Mô hình Checkpoint** | `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt` |
| **Checkpoint SHA-256** | `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6` |
| **Negative Manifest SHA-256** | `a32f10e032b06e496f44a19d4bbb6fe8dffc93c034b424704376cf5ee6e06071` |
| **Ground Truth Events SHA-256** | `8c6f30bbd057d5b5ae453feb64eb010387c2087ab9350ccea922ba04325bf7c3` |
| **Phần cứng thực thi** | NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM) |
| **Hệ điều hành & Thư viện** | Windows, Python 3.10.11, PyTorch 2.5.1+cu121, OpenCV 5.0.0 |

---

## 7. BẤT THƯỜNG CÒN TỒN TẠI & KẾ HOẠCH XỬ LÝ

1. **24/35 Sự kiện Ground Truth còn ở trạng thái `pending_verification`:**
   - *Nguyên nhân:* Theo quy trình thẩm định hai bước độc lập (double-blind annotation), 11 sự kiện đã được xác thực chéo (cross-verified) bởi annotator thứ hai, trong khi 24 sự kiện còn lại mới chỉ có nhãn của annotator thứ nhất.
   - *Biện pháp kiểm soát:* Khóa trạng thái Operating Point ở mức **PROVISIONAL**. Tách bạch rõ ràng số liệu giữa tập đã cross-verified và toàn bộ tập exploratory. Tuyệt đối không tự ý chuyển trạng thái sang OFFICIAL khi chưa có thẩm định độc lập từ người gán nhãn thứ hai.

---

## 8. KẾT LUẬN NGHIỆM THU TUẦN 4

**Trạng thái nghiệm thu:** **PASS / SCIENTIFICALLY_CONSISTENT** (cho toàn bộ dữ liệu âm tính, full decode, full streaming SHA-256 và đồng bộ artifacts).  
**Trạng thái Operating Point:** **PROVISIONAL** (đầy đủ điều kiện thực nghiệm trên 7.2026h video âm tính, sẵn sàng chuyển OFFICIAL khi annotator thứ hai hoàn tất thẩm định 24 sự kiện dương tính).

