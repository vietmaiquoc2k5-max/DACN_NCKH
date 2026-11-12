# BÁO CÁO TOÀN DIỆN VỀ ĐỘ TIN CẬY AUDIT VIDEO, XÁC MINH EOF VÀ ĐỒNG BỘ FAR TUẦN 4
**Comprehensive Video Audit Reliability, Stream Termination Verification, Provenance & FAR Reconciliation Report**

- **Dự án**: Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát
- **Kỹ sư thực hiện**: Mai Quốc Việt (`vietmaiquoc2k5-max/DACN_NCKH`)
- **Thời điểm hoàn tất**: 12/11/2026 – 17:35:00 (GMT+7)
- **Base Commit**: `bc636ea1e923fcc5588e42a572877b8b5af506a7`
- **Kết luận nghiệm thu Week 4**: `PARTIAL_ACCEPTANCE` (Hạ tầng kỹ thuật, cơ sở toán học, phân loại luồng và kiểm thử đạt 100%; bảo vệ nguyên tắc khoa học giữ Operating Point ở mức `PROVISIONAL` chờ Annotator 02 nghiệm thu 24 sự kiện Ground Truth)

---

## 1. CÁC LỖI ĐÃ PHÁT HIỆN TRƯỚC VÀ SAU COMMIT bc636ea

Qua quá trình rà soát độc lập và kiểm định khoa học, các vấn đề kỹ thuật sau đã được định vị chính xác:

1. **Lỗi xác minh EOF giả định từ OpenCV (`cap.grab() == False`)**:
   - `cap.grab()` trả về `False` khi luồng video kết thúc, nhưng không phân biệt được đó là kết thúc bình thường (normal EOF), đứt gói mạng (packet loss), lỗi cú pháp NAL unit hay file bị cắt cụt giữa chừng.
   - Code cũ từng gán `full_decode_verified = (actual_decoded_frames > 0)`, khiến hai video âm tính bị cắt ngắn giữa chừng (`VN_NEG_10` thiếu 3.248 frames và `VN_NEG_48` thiếu 5.912 frames) bị gán nhầm là đã giải mã thành công 100%.

2. **Chính sách dung sai (Tolerance Policy) thiếu căn cứ thời gian thực**:
   - Các ngưỡng 5 frame và 10 frame nếu áp dụng cố định mà không xét đến tốc độ khung hình (FPS) sẽ gây sai lệch nghiêm trọng. Ở 5 FPS, 5 frames tương đương 1.0 giây cảnh quay an ninh bị mất.

3. **Thiếu kiểm tra tính hợp lệ của FPS và Container Frame Count**:
   - Khi metadata container bị lỗi (ví dụ `reported_frame_count <= 0` hoặc `FPS <= 0`), logic cũ vẫn dùng hàm `max(1.0, fps)` để tính thời gian mà không chuyển sang trạng thái cảnh báo `NEEDS_REVIEW`.

4. **FFprobe / FFmpeg chưa hỗ trợ chế độ giải mã toàn diện (Full Stream Decode)**:
   - Trước đây hàm `probe_video_with_ffprobe` chỉ đọc metadata từ header mà không có cơ chế decode toàn bộ luồng để kiểm tra mã thoát (`exit_code`) và nhật ký lỗi `stderr`.

5. **Sai lệch mẫu số phơi nhiễm FAR (Exposure Mismatch)**:
   - Thời lượng khai báo container danh nghĩa (7,2026 giờ) bao gồm cả 251,26 giây của các frame bị hỏng không giải mã được. Việc lấy mẫu số 7,2026h mà không loại trừ phần frame hỏng gây mất tính đồng bộ với tử số cảnh báo.

6. **Lệch vết nguồn gốc (Provenance Discrepancy)**:
   - Một số artifact lưu trữ mã commit cũ `c95907f` hoặc `091352f` trong khi HEAD thực tế của repository đã tiến tới `bc636ea1`.

---

## 2. NGUYÊN NHÂN GỐC RỄ (ROOT CAUSE ANALYSIS)

| Mã vấn đề | Vị trí mã nguồn | Nguyên nhân kỹ thuật sâu xa | Hậu quả |
| :--- | :--- | :--- | :--- |
| **RC-01** | `src/video_audit.py::audit_single_video` | Sử dụng biểu thức `(actual_decoded_frames > 0)` để xác định `full_decode_verified`. | Video bị đứt luồng ở frame 14/35 vẫn bị gắn cờ đã xác minh giải mã thành công 100%. |
| **RC-02** | `src/video_audit.py::classify_decode_termination` | Không kiểm tra `fps <= 0` và `reported <= 0` trước khi so sánh độ lệch. | Khi metadata container hỏng, hệ thống vẫn chấp nhận dung sai và trả về `VALID`. |
| **RC-03** | `src/video_audit.py` | Thiếu module bọc FFmpeg/FFprobe thực hiện lệnh null-sink decode (`-f null -`). | Không phát hiện được các lỗi bitstream H.264/H.265 ẩn mà OpenCV bỏ qua mà không ném exception. |
| **RC-04** | `tests/test_video_audit_and_consistency.py` | `test_17` chấp nhận cả `"NORMAL_EOF"` khi inject rác vào luồng. `test_22` chỉ so sánh biến tĩnh `missing <= 10`. | Bộ kiểm thử không bắt được các hồi quy logic trong quy trình phân loại luồng. |
| **RC-05** | `runs/benchmarks/week4_final/far_report_verified.json` | Mẫu số phơi nhiễm gộp cả 251,26 giây của video lỗi `VN_NEG_10` và `VN_NEG_48`. | Mẫu số phơi nhiễm không phản ánh đúng lượng thời gian thực sự được thuật toán quét qua. |

---

## 3. CÁC THAY ĐỔI MÃ NGUỒN (CODE MODIFICATIONS)

### 3.1. Sửa đổi `src/video_audit.py`
1. **Bổ sung `run_external_stream_decode_check`**:
   - Sử dụng `ffmpeg -v error -i <file> -map 0:v:0 -f null -` hoặc `ffprobe -v error -count_frames ...`.
   - Bắt mã thoát `exit_code`, nhật ký lỗi `stderr`, phát hiện từ khóa bất thường (`error`, `corrupt`, `truncat`, `failed`).
   - Nếu hệ thống không cài đặt FFmpeg/FFprobe, trả về cờ rõ ràng `"tool_available": False` và kích hoạt OpenCV PTS fallback có ghi chú.
2. **Tái cấu trúc hàm `classify_decode_termination`**:
   - Thêm tham số `external_check`. Nếu công cụ ngoại vi báo `exit_code != 0`, chuyển ngay `TRUNCATED_STREAM` hoặc `DECODE_ERROR`.
   - Bổ sung kiểm tra nghiêm ngặt:
     - `fps <= 0.0 or math.isnan(fps) or math.isinf(fps) or fps > 240.0` $\rightarrow$ `NEEDS_REVIEW`, `full_decode_verified = False`.
     - `reported_frame_count <= 0` $\rightarrow$ `NEEDS_REVIEW`, `full_decode_verified = False`.
     - `actual_decoded_frames == 0` $\rightarrow$ `DECODE_ERROR`, `status = CORRUPTED`.
   - Ràng buộc dung sai thời gian:
     - $1 \le \Delta \le 5$: Yêu cầu $\Delta t = \Delta / \text{FPS} \le 0.25\text{ s}$ (1 GOP / B-frame PTS rounding). Vượt quá $\rightarrow$ `NEEDS_REVIEW`.
     - $6 \le \Delta \le 10$: Yêu cầu $\Delta t \le 0.35\text{ s} \land r = \Delta / N_{\text{reported}} \le 1.0\%$. Vượt quá $\rightarrow$ `NEEDS_REVIEW`.
     - $\Delta > 10$: Bắt buộc `TRUNCATED_STREAM`, `status = NEEDS_REVIEW`, `full_decode_verified = False`.
3. **Cập nhật `audit_single_video` và `_make_error_record`**:
   - Ghi nhận `stream_verification_backend`: phản ánh đúng `external_ffmpeg_full_decode`, `ffprobe_stream_metadata`, hoặc `opencv_pts_frame_analysis (ffprobe_not_found_fallback)`.
   - Lưu trữ trường `external_tool_check` trong kết quả audit.

### 3.2. Cập nhật `tests/test_video_audit_and_consistency.py`
- Mở rộng bộ kiểm thử từ 42 test lên **48 test chuyên sâu** (kết hợp cùng 7 test trong `test_poisson_ci.py` thành tổng cộng **55 test cases**).
- Thêm các bài test kiểm tra pipeline thực tế:
  - `test_44_audit_pipeline_invalid_fps`: Xác nhận pipeline gắn đúng `NEEDS_REVIEW` và `duration = 0.0` khi FPS không hợp lệ.
  - `test_45_audit_pipeline_zero_reported_frames`: Xác nhận pipeline gắn đúng `NEEDS_REVIEW` khi container báo 0 frames.
  - `test_46_audit_pipeline_ffprobe_not_installed`: Xác nhận pipeline tự động chuyển sang OpenCV fallback an toàn mà không bị crash.
  - `test_47_audit_pipeline_external_tool_error_exit_code`: Xác nhận khi external tool trả exit_code != 0, pipeline gắn đúng `TRUNCATED_STREAM`/`DECODE_ERROR`.
  - `test_48_audit_pipeline_silent_eof_missing_frames`: Xác nhận khi decoder dừng im lặng thiếu frame, pipeline gắn đúng `TRUNCATED_STREAM`.

---

## 4. KẾT QUẢ TỪNG TEST THỰC TẾ (100% PASS)

Toàn bộ **55 bài test** được thực thi trực tiếp bằng môi trường Python chính thức của repository:
`d:\DACN_NCKH\.venv\Scripts\python.exe -m unittest tests/test_video_audit_and_consistency.py tests/test_poisson_ci.py -v`

```text
Ran 55 tests in 26.218s

OK
[mjpeg @ 000001b17f2dab40] overread 8
[mjpeg @ 000001b17f30b700] No JPEG data found in image
[mov,mp4,m4a,3gp,3g2,mj2 @ 000001b17f30e9c0] moov atom not found
[mjpeg @ 000001b17f30b700] overread 8
[mjpeg @ 000001b17f30e9c0] overread 1
[mjpeg @ 000001b17f30e9c0] dht: len 118 is too large
[mjpeg @ 000001b17f30e9c0] huffman table decode error
```

### Danh mục kết quả 55 kiểm thử:

| STT | Tên Test Case | Mục tiêu kiểm định | Kết quả |
| :---: | :--- | :--- | :---: |
| 1 | `test_01_hash_small_file` | SHA-256 file nhỏ ổn định | **PASS** |
| 2 | `test_02_hash_large_file_chunked` | Streaming SHA-256 chunk 1MB không tràn RAM | **PASS** |
| 3 | `test_03_hash_empty_file` | File 0 byte trả về đúng SHA-256 chuỗi rỗng | **PASS** |
| 4 | `test_04_hash_unreadable_file` | Ném ngoại lệ khi file không tồn tại | **PASS** |
| 5 | `test_05_hash_same_content_different_path` | Cùng nội dung khác đường dẫn có SHA-256 trùng | **PASS** |
| 6 | `test_06_hash_same_size_different_content` | Cùng kích thước khác nội dung có SHA-256 khác | **PASS** |
| 7 | `test_07_cache_hit_unchanged_file` | Cache hit khi kích thước và SHA-256 không đổi | **PASS** |
| 8 | `test_08_cache_invalidation_on_change` | Invalidate cache khi file thay đổi kích thước | **PASS** |
| 9 | `test_09_cache_invalidation_same_size_different_sha` | Invalidate cache khi file sửa nội dung cùng kích thước | **PASS** |
| 10 | `test_10_cache_old_schema_no_sha256` | Invalidate cache bản ghi cũ schema version < 2 | **PASS** |
| 11 | `test_11_cache_missing_sha256_field` | Không tái sử dụng cache thiếu trường SHA-256 | **PASS** |
| 12 | `test_12_cache_custom_root_dir` | Cache key relative_path đúng theo root_dir | **PASS** |
| 13 | `test_13_cache_relative_path_normalization` | Chuẩn hóa Windows backslash thành POSIX slash | **PASS** |
| 14 | `test_14_cache_key_collision_different_dirs` | Không va chạm key giữa các file cùng tên khác thư mục | **PASS** |
| 15 | `test_15_decode_normal_eof` | Video hợp lệ giải mã tuần tự đến EOF bình thường | **PASS** |
| 16 | `test_16_decode_truncated_video` | Video bị cắt ngắn thiếu > 10 frames: TRUNCATED_STREAM | **PASS** |
| 17 | `test_17_decode_midstream_error` | Rác giữa luồng: dừng sớm, từ chối NORMAL_EOF | **PASS** |
| 18 | `test_18_decode_cannot_open` | File rác không thể mở: CANNOT_OPEN, CORRUPTED | **PASS** |
| 19 | `test_19_decode_probe_only` | Chế độ full_decode=False: PROBE_ONLY | **PASS** |
| 20 | `test_20_decode_metadata_frame_count_wrong` | Metadata frame count sai: dùng frame decode tính thời lượng | **PASS** |
| 21 | `test_21_decode_full_decode_verified_requires_eof` | full_decode_verified=True chỉ khi đến EOF | **PASS** |
| 22 | `test_22_decode_borderline_tolerance` | Kiểm thử chặt 4 kịch bản thời gian/tỷ lệ dung sai cận | **PASS** |
| 23 | `test_23_decode_partial_video` | Video chỉ decode được một số frame: ghi nhận chính xác | **PASS** |
| 24 | `test_24_manifest_duplicate_video_id` | Phát hiện và gắn nhãn DUPLICATE_ID | **PASS** |
| 25 | `test_25_manifest_duplicate_sha256` | Phát hiện và gắn nhãn DUPLICATE cho file trùng SHA-256 | **PASS** |
| 26 | `test_26_manifest_missing_file` | Phát hiện file có trong manifest nhưng thiếu trên đĩa | **PASS** |
| 27 | `test_27_manifest_no_auto_verified_negative` | Không tự động gán nhãn VERIFIED_NEGATIVE | **PASS** |
| 28 | `test_28_manifest_no_downgrade_verified` | Bảo vệ không hạ cấp manifest đã verified | **PASS** |
| 29 | `test_29_manifest_legacy_migration` | Tương thích ngược schema cũ an toàn | **PASS** |
| 30 | `test_30_manifest_invalid_schema` | Xử lý an toàn bản ghi rác | **PASS** |
| 31 | `test_31_far_zero_alarms` | Poisson CI khi 0 báo giả: lower=0.0, upper>0.0 | **PASS** |
| 32 | `test_32_far_one_alarm` | Poisson CI khi 1 báo giả: lower>0.0 | **PASS** |
| 33 | `test_33_far_multiple_alarms` | Poisson CI số lượng lớn: không tràn số | **PASS** |
| 34 | `test_34_far_zero_exposure` | Phơi nhiễm T=0 ném ValueError rõ ràng | **PASS** |
| 35 | `test_35_far_negative_exposure` | Phơi nhiễm T<0 ném ValueError rõ ràng | **PASS** |
| 36 | `test_36_poisson_ci_matches_scipy` | Đối chiếu khớp chính xác phân phối Gamma/Chi-Square | **PASS** |
| 37 | `test_37_far_consistent_exposure_scope` | Kiểm thử nhất quán mẫu số trên 2 phạm vi phơi nhiễm | **PASS** |
| 38 | `test_38_integration_manifest_audit_inference` | Đối soát phát hiện sai lệch số frame và thời lượng | **PASS** |
| 39 | `test_39_integration_repeated_run_same_result` | Tính tái lập: 2 lần audit độc lập cho kết quả giống hệt | **PASS** |
| 40 | `test_40_integration_metrics_sync` | Đồng bộ số liệu không tạo số liệu giả | **PASS** |
| 41 | `test_41_integration_legacy_script_no_overwrite` | Bảo vệ ghi đè manifest | **PASS** |
| 42 | `test_42_integration_provenance_mismatch_detected` | Phát hiện sai lệch commit giữa các artifact | **PASS** |
| 43 | `test_43_decode_missing_frames_without_decoder_error` | Phân loại khi decoder dừng im lặng không ném ngoại lệ | **PASS** |
| 44 | `test_44_audit_pipeline_invalid_fps` | Pipeline thực tế: FPS=0 chuyển NEEDS_REVIEW | **PASS** |
| 45 | `test_45_audit_pipeline_zero_reported_frames` | Pipeline thực tế: reported=0 chuyển NEEDS_REVIEW | **PASS** |
| 46 | `test_46_audit_pipeline_ffprobe_not_installed` | Pipeline thực tế: OpenCV fallback an toàn khi thiếu FFprobe | **PASS** |
| 47 | `test_47_audit_pipeline_external_tool_error_exit_code` | Pipeline thực tế: External tool lỗi exit code != 0 | **PASS** |
| 48 | `test_48_audit_pipeline_silent_eof_missing_frames` | Pipeline thực tế: Decoder dừng sớm thiếu frame | **PASS** |
| 49-55 | `test_poisson_ci (7 bài test)` | Đơn điệu, độ rộng khoảng tin cậy, các mức phơi nhiễm | **PASS** |

---

## 5. BẰNG CHỨNG XÁC MINH EOF (STREAM TERMINATION EVIDENCE)

1. **Không suy diễn EOF từ `cap.grab() == False`**:
   - Khi `cap.grab()` trả về `False`, hệ thống bắt buộc đối chiếu số frame thực tế decode được $N_{\text{actual}}$ với số frame khai báo trong container $N_{\text{reported}}$.
   - Nếu $N_{\text{actual}} < N_{\text{reported}} - 10$, hệ thống gắn cờ `TRUNCATED_STREAM` và đặt `full_decode_verified = False`, bất kể bộ giải mã có ném ngoại lệ hay không.
2. **Không phụ thuộc đơn lẻ vào `CAP_PROP_FRAME_COUNT`**:
   - Khi container metadata báo 0 khung hình hoặc FPS không hợp lệ, hệ thống từ chối đánh dấu `VALID` và chuyển thẳng về `NEEDS_REVIEW`.
3. **Cơ chế bọc FFmpeg/FFprobe**:
   - Khi có công cụ ngoại vi trong PATH, hệ thống kiểm tra toàn vẹn luồng bằng cách decode toàn bộ stream tới null sink, theo dõi exit code và stderr.
   - Khi không có công cụ ngoại vi, hệ thống ghi rõ `"stream_verification_backend": "opencv_pts_frame_analysis (ffprobe_not_found_fallback)"` để bảo đảm tính minh bạch khoa học.

---

## 6. ĐỒNG BỘ MẪU SỐ PHƠI NHIỄM VÀ TÍNH TOÁN FAR CHÍNH XÁC

Tuyệt đối không cộng thời lượng của các frame không đọc được vào mẫu số phơi nhiễm. Hai video lỗi luồng được cô lập như sau:
- **`VN_NEG_10`**: Thiếu 3.248 frames ($54,19\text{s}$). Pipeline B2: 0 cảnh báo. Pipeline A: 10 cảnh báo.
- **`VN_NEG_48`**: Thiếu 5.912 frames ($197,07\text{s}$). Pipeline B2: 4 cảnh báo. Pipeline A: 95 cảnh báo.
- **Tổng thời lượng hỏng bị loại trừ**: $251,26\text{ giây} = 0,0698\text{ giờ}$ (9.160 khung hình).

### Bảng kết quả FAR trên 2 phạm vi đối chiếu khoa học:

| Chỉ số kỹ thuật | Scope 1: Toàn bộ luồng giải mã (79 video) | Scope 2: Nghiêm ngặt 77 video không lỗi luồng | Mục tiêu Đề cương NCKH (MT3) |
| :--- | :---: | :---: | :---: |
| **Số lượng video** | 79 video | 77 video | $\ge 5\text{ giờ}$ video không cháy |
| **Thời gian phơi nhiễm ($T$)** | **7,1312 giờ** (25.672,14 s) | **6,9846 giờ** (25.144,67 s) | $\ge 5,0\text{ giờ}$ (Đạt: dôi dư $+1,98\text{h}$) |
| **Báo giả Pipeline A (Raw YOLO)** | 988 cảnh báo | 883 cảnh báo | Baseline từng khung hình |
| **FAR Pipeline A** | $138,5461\text{ /h}$ | $126,4210\text{ /h}$ | — |
| **95% Poisson CI (Pipeline A)** | $[130,04;\ 147,47]$ | $[118,22;\ 135,04]$ | Garwood Exact |
| **Báo giả Pipeline B1 (ByteTrack)** | 547 cảnh báo | 541 cảnh báo | Giảm bước 1 |
| **FAR Pipeline B1** | $76,7052\text{ /h}$ | $77,4561\text{ /h}$ | — |
| **Báo giả Pipeline B2 (Temporal)** | **421 cảnh báo** | **417 cảnh báo** | Pipeline đề xuất chính thức |
| **FAR Pipeline B2** | **$59,0361\text{ /h}$** | **$59,7028\text{ /h}$** | — |
| **95% Poisson CI (Pipeline B2)** | **$[53,53;\ 64,95]$** | **$[54,11;\ 65,72]$** | Garwood Exact |
| **Tỷ lệ giảm báo giả vs Pipeline A** | **$57,39\%$** | **$52,77\%$** | **Giảm $\ge 50\%$** |
| **Đánh giá chỉ tiêu MT3** | **ĐẠT (Vượt 7,39 điểm %)** | **ĐẠT (Vượt 2,77 điểm %)** | **PASS CẢ 2 PHẠM VI** |

---

## 7. TRUY XUẤT NGUỒN GỐC VÀ TÍNH ĐỒNG BỘ ARTIFACT (PROVENANCE)

Tất cả các tệp cấu hình, báo cáo và chỉ số được đồng bộ hóa với commit gốc:
- **Source & Audit Commit**: `bc636ea1e923fcc5588e42a572877b8b5af506a7`
- **Model Checkpoint**: `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt`
  - SHA-256: `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`
- **Manifest Negative Videos**: `data/negative_videos_manifest.json` (Version 2.3)
  - SHA-256: `5413a0328fc6aea70da0f5b22f96514b8bab780a45737f22be0dea896768bfb1`
- **Inference Summary ID**: `run_20261111_233035_neg_full` (128.291 processed frames tại 5 FPS temporal sampling)
- **Danh sách artifact đồng bộ**:
  - `runs/benchmarks/week4_final/provenance_audit.json`
  - `runs/benchmarks/week4_final/far_report_verified.json`
  - `runs/benchmarks/week4_final/operating_point_audit.json`
  - `runs/benchmarks/week4_final/rtsp_benchmark_status.json`
  - `runs/benchmarks/week4_final/metrics.json`
  - `runs/benchmarks/week4_final/ground_truth/ground_truth_master.csv`

---

## 8. CÁC VẤN ĐỀ CHƯA GIẢI QUYẾT (REMAINING LIMITATIONS)

1. **Ground Truth Verification (24 sự kiện pending)**:
   - Trong 35 sự kiện dương tính, 11 sự kiện đã được xác nhận chéo bởi 2 annotator độc lập.
   - 24 sự kiện còn lại vẫn ở trạng thái `PENDING_HUMAN_REVIEW` trong tệp [gt_review_queue.csv](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/ground_truth/gt_review_queue.csv).
   - **Cam kết khoa học**: Không tự động đánh dấu VERIFIED khi chưa có sự xác nhận của Annotator 02.
2. **Operating Point**:
   - Khóa chặt ở trạng thái **`PROVISIONAL`** (Tạm thời có kiểm chứng toán học), chưa nâng lên `OFFICIAL`.
3. **RTSP Multi-stream**:
   - Ghi nhận trạng thái **`NOT_TESTED / BLOCKED_NO_LIVE_RTSP_SOURCE`**, không giả lập số liệu bằng file video để tuyên bố đạt MT5. Việc triển khai hệ thống đa luồng thực tế bằng MediaMTX sẽ được tiến hành đúng tiến độ Tuần 7.

---

## 9. KẾT LUẬN NGHIỆM THU TUẦN 4

- **Trạng thái nghiệm thu chính thức**: **`PARTIAL_ACCEPTANCE`**.
- **Cơ sở kỹ thuật và toán học**: Hoàn thành 100% các tiêu chí về Video Audit, Stream Termination Classification, Tolerance Policy, Full-video Sampling Inference, False Alarm Rate và Exact Poisson Confidence Interval.
- **Tiêu chuẩn khoa học**: Tuân thủ nghiêm ngặt nguyên tắc trung thực số liệu, minh bạch các hạn chế về Ground Truth và nguồn phát RTSP, sẵn sàng chuyển tiếp sang nhiệm vụ Tuần 5 (Thích nghi miền - Domain Adaptation).

