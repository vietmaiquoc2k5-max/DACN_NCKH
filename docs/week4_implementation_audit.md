# BÁO CÁO AUDIT TRIỂN KHAI THỰC NGHIỆM TUẦN 4
## Hệ Thống Phát Hiện & Xác Nhận Cháy Khói Thời Gian Thực (YOLO + ByteTrack + Temporal Verification)

- **Repository:** `vietmaiquoc2k5-max/DACN_NCKH`
- **Branch:** `main`
- **Thời gian thực hiện audit:** 2026-11-10
- **Tiêu chuẩn đối chiếu:** Đề cương NCKH (Mục tiêu MT1, MT2, MT3) & Hướng dẫn kỹ thuật Tuần 4

---

## 1. Hiện Trạng Thành Phần Đã Triển Khai (What Exists)

1. **Mô hình & Phát hiện đối tượng (`src/detector.py`):**
   - Lớp `SharedFireSmokeDetector` hỗ trợ nạp checkpoint PyTorch `best.pt` (huấn luyện trên D-Fire, lớp 0: SMOKE, lớp 1: FIRE).
   - Đã tối ưu suy luận: `fuse()` conv+bn, GPU warmup, FP16 inference, kiểm tra tính hợp lệ của metadata lớp nhãn (từ chối mô hình không hợp lệ như `yolov8n.pt` COCO).
2. **Theo dõi đối tượng (`src/bytetrack_adapter.py`):**
   - Đã tích hợp ByteTrack chính thức 2 giai đoạn (Ultralytics official tracker).
   - Duy trì `track_id` ổn định qua các khung hình mất dấu ngắn hạn (`track_buffer=15`).
   - Tách biệt không gian track giữa 2 lớp `FIRE` và `SMOKE` (`per_class=True`), không gộp nhầm track.
3. **Trích xuất đặc trưng biến động thời gian (`src/temporal_features.py`):**
   - `TemporalFeatureExtractor` duy trì lịch sử theo `(camera_id, track_id)`.
   - Trích xuất: độ chớp ROI (flicker intensity variance), tốc độ mở rộng diện tích khói (`area_growth_rate`), tốc độ dịch chuyển tâm (`centroid_speed`).
4. **Bộ xác minh thời gian (`src/temporal_verifier.py`):**
   - Cửa sổ trượt $K/N$ độc lập cho từng `track_id` (kết hợp `tracks_assigned=True`).
   - Cổng biến động đặc trưng `feature_threshold`: chặn đối tượng tĩnh khi Gate BẬT (`feat_thr > 0.0`), cho qua khi Gate TẮT (`feat_thr = 0.0`).
   - Vùng loại trừ ROI Masking.
   - Cơ chế Cooldown nhân quả (`cooldown_sec`) ngăn lặp cảnh báo trên cùng track mà không chặn sự kiện mới sau khi hết cooldown.
5. **Công cụ đánh giá sự kiện (`src/event_eval.py` & wrapper `src/week4_event_eval.py`):**
   - Định nghĩa dataclass `Event` và `FrameDetection`.
   - Thuật toán gộp cảnh báo thành sự kiện `merge_frame_alarms_to_events` với `max_event_gap_sec`.
   - Khớp sự kiện dự đoán với Ground Truth `evaluate()` theo `tolerance_sec=10.0s`.
   - Đánh giá các chỉ số: Video-level Event Recall, Event-instance Recall, Event Precision, Event F1, Mean Detection Delay, False Alarms per Hour (FAR/giờ).
   - Quét lưới siêu tham số `grid_search()` với nguyên tắc so sánh công bằng: đối sánh từng ứng viên với baseline tại chính xác cùng mức `confidence_threshold`.
6. **Bộ kiểm thử tự động (`tests/`):**
   - 40 unit tests trong `tests/` pass 100% (`test_bytetrack_adapter.py`, `test_camera_tracker_isolation.py`, `test_temporal_features.py`, `test_temporal_verifier_bytetrack.py`, `test_week4_event_eval.py`).
   - 5 system tests trong `tests/test_dashboard_system.py` pass 100%.

---

## 2. Những Thành Phần Còn Thiếu (What Is Missing)

1. **Thiếu script xuất Frame-level JSONL chuyên dụng (`scripts/export_week4_frame_jsonl.py`):**
   - Trước đây việc xuất JSONL được nhúng trong script benchmark tổng hợp `benchmark_week4_bytetrack.py` và chỉ xuất `frame_level_events.json` / `temporal_confirmed_events.json`.
   - Chưa có script chuẩn hóa độc lập để xuất hai tệp JSONL khung hình chi tiết: `frame_level_detections.jsonl` (raw detections từ YOLO, không chứa track_id giả) và `tracked_frame_detections.jsonl` (tracked detections từ ByteTrack kèm `track_id` và `temporal_score`).
2. **Thiếu script kiểm tra tính hợp lệ dữ liệu JSONL (`scripts/validate_week4_frame_jsonl.py`):**
   - Chưa có công cụ độc lập kiểm tra tự động: tính toàn vẹn cú pháp JSONL, tính tăng đơn điệu của timestamp, sự hiện diện đầy đủ của các frame rỗng (`detections: []`), tính hợp lệ của confidence và track_id.
3. **Hệ thống Ground Truth sự kiện độc lập chưa hoàn thiện:**
   - Hiện tại benchmark Tuần 4 đang phụ thuộc vào hàm `build_ground_truth_events()` tạo GT tự động từ `fire_start_sec`, `smoke_start_sec` trong CSV và gán `end_sec = video_duration`.
   - Chưa có tệp lưu trữ Ground Truth sự kiện độc lập chuẩn hóa `data/ground_truth/events.jsonl` với metadata người gán nhãn (`annotator_id`), khoảng thời gian thực tế $[start\_sec, end\_sec]$ và trạng thái xác minh (`verified` / `pending`).
   - Chưa có tài liệu hướng dẫn quy trình gán nhãn `data/ground_truth/README.md`.
   - Chưa có script kiểm tra tính đúng đắn của Ground Truth `scripts/validate_ground_truth.py`.
4. **Thiếu dữ liệu video âm tính vật lý trên ổ đĩa:**
   - Trong `data/video_dataset_list.csv` có danh mục 28 video âm tính (`NEG_25..49` và `VN_NEG_01, 03, 04`) tương đương 2.8648 giờ, nhưng thực tế **0 video âm tính tồn tại trên ổ đĩa**.
   - Chưa có dữ liệu video âm tính vật lý để tính FAR thực nghiệm.

---

## 3. Các Lỗi & Điểm Cần Chỉnh Sửa Phát Hiện Được (Detected Issues)

1. **Vấn đề giả định `end_sec` của Ground Truth:**
   - Khi dùng mốc thời gian từ CSV, `end_sec` bị gán mặc định bằng `duration_sec` của video. Điều này có thể không phản ánh đúng các video mà đám cháy bị dập tắt hoặc khói tan giữa chừng. Cần phân tách Ground Truth độc lập lưu trữ tại `data/ground_truth/events.jsonl`.
2. **Đồng bộ hóa nguồn nạp Ground Truth trong `src/event_eval.py`:**
   - `src/event_eval.py` đã có hàm `load_events()`, nhưng cần bổ sung hàm `load_ground_truth_events()` có khả năng đọc linh hoạt từ `data/ground_truth/events.jsonl` hoặc dự phòng từ metadata video list với cờ báo rõ ràng về nguồn gốc dữ liệu.
3. **Phân tách rõ ràng giữa Raw Detections và Tracked Detections:**
   - Trong raw detections (`frame_level_detections.jsonl`), detection không được chứa `track_id` giả lập (phải để `track_id = null` hoặc không có trường này).
   - Trong tracked detections (`tracked_frame_detections.jsonl`), `track_id` phải là số nguyên dương thực tế do ByteTrack cấp phát.
4. **Lưu trữ kết quả kiểm định Test Set:**
   - Cần xuất riêng tệp `runs/benchmarks/week4_final/locked_test_evaluation.json` và báo cáo phân tích điểm vận hành `docs/week4_operating_point_report.md`.

---

## 4. Danh Sách Các Tệp Cần Chỉnh Sửa & Tạo Mới

| Tác vụ | Đường dẫn tệp | Trạng thái | Mục đích |
| :--- | :--- | :---: | :--- |
| **Audit** | `docs/week4_implementation_audit.md` | Tạo mới | Ghi nhận hiện trạng toàn diện trước khi thực thi |
| **JSONL Exporter** | `scripts/export_week4_frame_jsonl.py` | Tạo mới | Xuất `frame_level_detections.jsonl` và `tracked_frame_detections.jsonl` |
| **JSONL Validator** | `scripts/validate_week4_frame_jsonl.py` | Tạo mới | Kiểm tra tính hợp lệ và xuất `frame_jsonl_validation.json` |
| **Ground Truth Data** | `data/ground_truth/events.jsonl` | Tạo mới | Lưu trữ Ground Truth độc lập theo chuẩn schema |
| **Ground Truth Guide** | `data/ground_truth/README.md` | Tạo mới | Quy trình gán nhãn, kiểm tra chéo và khóa Ground Truth |
| **Ground Truth Validator**| `scripts/validate_ground_truth.py` | Tạo mới | Kiểm tra logic $[start\_sec, end\_sec]$, duration và nhãn |
| **Evaluator Update** | `src/event_eval.py` | Cập nhật | Bổ sung nạp Ground Truth độc lập từ JSONL |
| **Operating Point Report**| `docs/week4_operating_point_report.md` | Tạo mới | Phân tích chi tiết Grid Search và lựa chọn điểm vận hành |
| **Locked Test Eval** | `runs/benchmarks/week4_final/locked_test_evaluation.json` | Tạo mới | Báo cáo chi tiết đánh giá trên tập Test độc lập |
| **Acceptance Report**| `docs/WEEK4_FINAL_ACCEPTANCE_REPORT.md` | Tạo mới | Báo cáo nghiệm thu chính thức Tuần 4 |
| **Unit Tests** | `tests/test_week4_comprehensive.py` | Tạo mới | Bổ sung bộ test bao phủ JSONL, empty frames, GT, FAR, leakage |

---

## 5. Dữ Liệu Không Thể Tự Khôi Phục (Irrecoverable Data)

- **28 tệp video âm tính vật lý (`NEG_25..49`, `VN_NEG_01, 03, 04`):**
  - Không có file trên ổ cứng máy local (`0.00 giờ`).
  - Hệ thống tuân thủ đạo đức nghiên cứu khoa học: **Tuyệt đối không tự sinh video giả, không gán FAR = 0**.
  - Trạng thái FAR được duy trì minh bạch: `UNAVAILABLE_MISSING_NEGATIVE_VIDEOS` (`status = "BLOCKED_BY_NEGATIVE_DATA"`).
  - Điểm vận hành Operating Point được đánh dấu trung thực: `NO_FEASIBLE_OPERATING_POINT`.

