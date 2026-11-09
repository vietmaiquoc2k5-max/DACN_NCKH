# Báo Cáo Thực Nghiệm & Nghiệm Thu Tuần 4: Frame-Level Baseline vs. ByteTrack + Temporal Verification

- **Ngày thực hiện:** `03/11/2026`
- **Branch:** `tuần-4`
- **Script thực thi:** `scripts/benchmark_week4_bytetrack.py`
- **Công cụ đánh giá:** `src/event_eval.py` (wrapper tương thích: `src/week4_event_eval.py`)
- **Thư mục kết quả chuẩn:** `runs/benchmarks/week4_final/`
- **Trạng thái nghiệm thu Tuần 4:** **`PARTIAL (BLOCKED_BY_NEGATIVE_DATA)`**

---

## 1. Cấu Hình Thực Nghiệm & Tính Tái Lập

1. **Model Checkpoint:** `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt` (Mô hình YOLOv8n tốt nhất đã kiểm chứng ở Tuần 3 với Test mAP@0.5 = 76.78% trên D-Fire).
2. **Tham số cố định giữa các Pipeline:**
   - Tần số lấy mẫu khung hình: `5.0 FPS`
   - Độ phân giải suy luận: `imgsz = 416`
   - Ngưỡng độ tin cậy cơ sở: `confidence_threshold = 0.35`
   - Khoảng cách gộp sự kiện tối đa: `max_event_gap_sec = 2.0s`
   - Hạt giống ngẫu nhiên: `seed = 42`
3. **Định nghĩa 3 Pipeline so sánh độc lập (cô lập 100% trạng thái):**
   - **Pipeline A — `frame_level_baseline`:** Chỉ chạy YOLO detector. Mỗi khung hình có bất kỳ phát hiện `FIRE`/`SMOKE` nào đạt `confidence_threshold >= 0.35` lập tức tạo một cảnh báo cấp khung hình (`raw_frame_alarm`). **Không dùng ByteTrack, không dùng luật K/N, không dùng TemporalVerifier, không dùng Cooldown.** Các cảnh báo cấp khung hình liên tiếp cùng lớp cách nhau $\le 2.0\text{ s}$ được gộp thành các khoảng `Event` để đánh giá cấp sự kiện.
   - **Pipeline B — `temporal_confirmed_bytetrack_features`:** Chạy YOLO detector + `ByteTrackAdapter` (`per_class=True`) + `TemporalFeatureExtractor` + `TemporalVerifier` ($K=3, N=8$, `cooldown_sec = 5.0s`). Chỉ phát cảnh báo xác nhận khi cùng một `track_id` xuất hiện ít nhất $3/8$ khung hình trong cửa sổ trượt và vượt qua cổng biến động thời gian.
   - **Pipeline C — `legacy_iou_temporal`:** Chạy YOLO detector + bộ ghép cặp IoU nội bộ của `TemporalVerifier` ($K=3, N=8$, `cooldown_sec = 5.0s`) để đo lường riêng lợi ích của thuật toán ByteTrack so với IoU tracking truyền thống.

---

## 2. Hiện Trạng Dữ Liệu Video Kiểm Thử & Lý Do `BLOCKED_BY_NEGATIVE_DATA`

Kiểm toán thực tế từ danh mục `data/video_dataset_list.csv` (72 video) đối chiếu với hệ thống tệp trên ổ đĩa:

| Phân nhóm dữ liệu | Số lượng trong `video_dataset_list.csv` | Hiện có trên ổ đĩa | Thời lượng hiện có | Yêu cầu Đề cương (MT1/MT3) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Dương tính (Có cháy/khói)** | 44 (`POS_01..24`, `VN_POS_01..20`) | **24 video** (`POS_01..24` tại `FireSmokeLabeler/videos/`) | `863.98s` (`0.24h`, `4,307` frames mẫu) | $\ge 20$ video | **ĐẠT (24/20 video)** |
| **Video Âm tính (Không cháy / Cảnh nhiễu)** | 28 (`NEG_25..49`, `VN_NEG_01,03,04`) | **0 video** | `0.00s` (`0.00h`) | $\ge 5.0$ giờ | **THIẾU (`BLOCKED_BY_NEGATIVE_DATA`)** |

- **Manifest video âm tính bị thiếu:** Đã xuất đầy đủ tại `runs/benchmarks/week4_final/missing_negative_videos_manifest.csv` và `missing_negative_videos_manifest.json` (gồm 28 video âm tính tổng thời lượng `10,313.16s` = `2.8648 giờ` chưa có file vật lý trên ổ đĩa, cùng yêu cầu bổ sung thêm `2.1352 giờ` video âm tính mới để đạt đủ $\ge 5.0$ giờ theo Đề cương).
- **Tuân thủ quy tắc trung thực khoa học:** Do không có video âm tính trên ổ đĩa, chỉ số `false_alarms_per_hour` (FAR/giờ trên tập âm tính) được ghi nhận chính thức là **`UNAVAILABLE` (`BLOCKED_BY_NEGATIVE_DATA`)** và tuyệt đối không được gán bằng `0.0`.

---

## 3. Kết Quả Thực Nghiệm Đối Sánh 3 Pipeline

### 3.1. Trên Toàn Bộ 24 Video Dương Tính (`full_24_positive` — 863.98 giây)

| Chỉ số đánh giá | Pipeline A: Frame-Level Baseline (Không K/N, Không Tracker) | Pipeline B: Temporal-Confirmed (ByteTrack + Features + K=3/N=8) | Pipeline C: Legacy IoU Temporal (IoU Tracker + K=3/N=8) | Đánh giá tác động của Pipeline B |
| :--- | :---: | :---: | :---: | :--- |
| **Tổng số cảnh báo phát ra (Raw Frame Alarms vs DB Triggers)** | **2,938** frame alarms | **126** DB triggers | **128** DB triggers | **Giảm 95.71% số lần phát cảnh báo lặp** |
| **Số khoảng sự kiện sau khi gộp (`max_event_gap_sec = 2.0s`)** | **53** events | **29** events | **33** events | **Giảm 45.28% sự kiện phân mảnh** |
| **Video-level Event Recall (Strict Class Match)** | `95.83%` (23/24) | `95.83%` (23/24) | `95.83%` (23/24) | **Recall giảm 0.00 điểm %** (Đạt chuẩn $\le 5\%$) |
| **Video-level Primary Recall (Any Fire/Smoke Class)** | `95.83%` (23/24) | `95.83%` (23/24) | `95.83%` (23/24) | **Giữ nguyên 23/24 video** (chỉ `POS_09` model không bắt được) |
| **Event-instance Recall** | `80.00%` (28/35) | `71.43%` (25/35) | `74.29%` (26/35) | Lọc bỏ các nháy đơn lẻ 1–2 khung hình |
| **Event Precision** | `52.83%` (28/53) | **`86.21%`** (25/29) | `78.79%` (26/33) | **Tăng +33.38 điểm % so với Pipeline A (+7.42% so với IoU)** |
| **Event F1-Score** | `63.64%` | **`78.13%`** | `76.47%` | **Tăng +14.49 điểm % so với Pipeline A** |
| **Mean Detection Delay (Strict Class Match)** | `1.02 s` | **`3.56 s`** | `2.19 s` | **Đạt chuẩn Đề cương ($\le 10.0\text{ s}$)** |
| **Mean Detection Delay (Từ mốc khởi phát sớm nhất)** | `0.97 s` | **`3.41 s`** | `1.79 s` | **Đạt chuẩn Đề cương ($\le 10.0\text{ s}$)** |
| **Cảnh báo nổ ra trước mốc cháy (`pre_event_alarms` trên video dương)** | `3` | **`0`** | `0` | **Loại bỏ 100% cảnh báo sớm trước khi cháy** |
| **Sự kiện dư thừa không khớp GT trên video dương (`unmatched_positive_predictions`)** | `25` | **`4`** | `7` | **Giảm 84.0% sự kiện nhiễu/phân mảnh (từ 25 xuống 4)** |
| **Negative False Alarms per Hour (FAR/giờ trên video âm tính)** | `UNAVAILABLE` | `UNAVAILABLE` | `UNAVAILABLE` | `BLOCKED_BY_NEGATIVE_DATA` |
| **Detector-only Latency / FPS (CPU)** | `17.76 ms` (`56.3 FPS`) | `17.76 ms` (`56.3 FPS`) | `17.76 ms` (`56.3 FPS`) | Đạt chuẩn $\ge 5$ FPS CPU |
| **End-to-End Pipeline Latency / FPS (CPU)** | `17.76 ms` (`56.3 FPS`) | **`18.54 ms` (`53.9 FPS`)** | `17.79 ms` (`56.2 FPS`) | Tổng overhead toàn pipeline chỉ **+0.78 ms/frame** |

> **Phân rã thời gian xử lý (Latency Breakdown) của Pipeline B trên CPU:**
> - Suy luận YOLO (`SharedFireSmokeDetector.detect_frame`): `17.760 ms/frame`
> - Theo dõi đa đối tượng (`ByteTrackAdapter.update`): `0.513 ms/frame`
> - Trích xuất đặc trưng động học (`TemporalFeatureExtractor.update`): `0.237 ms/frame`
> - Xác minh cửa sổ trượt K/N (`TemporalVerifier.update`): `0.032 ms/frame`
> - **Tổng cộng End-to-End Pipeline B:** `18.542 ms/frame` (`53.9 FPS` trên CPU).

---

## 4. Quét Tham Số (Grid Search) Trên Validation Set & Kiểm Định Trên Test Set

Để tránh hiện tượng quá khớp (overfitting) khi chọn siêu tham số, 24 video dương tính được chia cố định thành 2 tập độc lập:
- **Validation Set (12 video lẻ):** `POS_01, POS_03, POS_05, POS_07, POS_09, POS_11, POS_13, POS_15, POS_17, POS_19, POS_21, POS_23` (`456.65s`).
- **Test Set khóa cố định (12 video chẵn):** `POS_02, POS_04, POS_06, POS_08, POS_10, POS_12, POS_14, POS_16, POS_18, POS_20, POS_22, POS_24` (`407.33s`).

### 4.1. Không gian quét lưới tham số (432 tổ hợp)
- `confidence_threshold` $\in \{0.25, 0.35, 0.45\}$
- $K \in \{2, 3, 4, 5\}$
- $N \in \{5, 8, 10, 12\}$ (với điều kiện $K \le N$)
- `feature_threshold` $\in \{0.0, 0.01, 0.02\}$
- `cooldown_sec` $\in \{0.0, 5.0, 12.0\}$

### 4.2. Kết quả chọn điểm vận hành (`runs/benchmarks/week4_final/best_operating_point.json`)
- **Trạng thái nghiệm thu điểm vận hành:** `NO_FEASIBLE_OPERATING_POINT` (`blocked_by_negative_data = True`).
- **Nguyên nhân:** Chưa có $\ge 5.0$ giờ video âm tính trên ổ đĩa để xác minh ràng buộc giảm $\ge 50\%$ FAR/giờ.
- **Điểm vận hành tạm thời (Provisional Candidate) tốt nhất trên Validation Set (`conf = 0.45, K = 2, N = 10, feature_threshold = 0.0, cooldown_sec = 0.0`):**
  - Trên Validation Set (12 video): Video Event Recall = `91.67%` (drop `0.00%`), Event Precision = `86.67%`, Event F1 = `81.25%`, Mean Detection Delay = `0.55s`.
  - Khi khóa tham số và đánh giá trên **Test Set (12 video)**: Video Event Recall = **`100.00%` (12/12)**, Event Precision = **`86.67%`**, Event F1 = **`78.79%`**, Mean Detection Delay = **`5.49s`** ($\le 10\text{ s}$).
- **Điểm vận hành cân bằng chống báo lặp (`conf = 0.35, K = 3, N = 8, cooldown_sec = 5.0s`) trên Test Set (12 video):**

| Tập dữ liệu | Pipeline | Raw Alarms / DB Triggers | Merged Events | Video Event Recall | Event Precision | Event F1 | Mean Delay |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Set (12 video)** | Pipeline A (Frame-level baseline) | 1,920 | 26 | `91.67%` | `53.85%` | `65.12%` | `0.26 s` |
| **Validation Set (12 video)** | Pipeline B (ByteTrack + K=3/N=8) | **79 (-95.89%)** | **16 (-38.46%)** | `91.67%` | **`81.25%`** | **`78.79%`** | `1.10 s` |
| **Test Set (12 video)** | Pipeline A (Frame-level baseline) | 1,018 | 27 | `100.00%` | `51.85%` | `62.22%` | `1.85 s` |
| **Test Set (12 video)** | Pipeline B (ByteTrack + K=3/N=8) | **47 (-95.38%)** | **13 (-51.85%)** | **`100.00%`** | **`92.31%`** | **`77.42%`** | `6.03 s` |

---

## 5. Danh Mục Tệp Đầu Ra Đã Kiểm Chứng

Toàn bộ kết quả thực nghiệm có thể tái lập 100% bằng lệnh `.\.venv\Scripts\python.exe scripts/benchmark_week4_bytetrack.py` và được lưu trữ tại `runs/benchmarks/week4_final/`:

1. `runs/benchmarks/week4_final/frame_level_events.json`
2. `runs/benchmarks/week4_final/temporal_confirmed_events.json`
3. `runs/benchmarks/week4_final/comparison.csv`
4. `runs/benchmarks/week4_final/metrics.json`
5. `runs/benchmarks/week4_final/grid_search.csv`
6. `runs/benchmarks/week4_final/grid_search.json`
7. `runs/benchmarks/week4_final/best_operating_point.json`
8. `runs/benchmarks/week4_final/missing_negative_videos_manifest.csv` & `missing_negative_videos_manifest.json`
9. `runs/benchmarks/week4_final/benchmark_report.md`
