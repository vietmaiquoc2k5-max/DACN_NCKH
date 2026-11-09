# Báo Cáo Thực Nghiệm & Nghiệm Thu Tuần 4: Frame-Level Baseline vs. ByteTrack + Temporal Verification

- **Thời gian thực thi:** `2026-11-10 02:52:26`
- **Model Checkpoint:** `runs\detect\runs\benchmarks\yolov8n_seed42\weights\best.pt` (mAP@0.5 = 76.78% trên D-Fire Test)
- **Tần số lấy mẫu khung hình:** `5.0 FPS` (`imgsz = 416`, `seed = 42`)
- **Trạng thái nghiệm thu Tuần 4:** **`PARTIAL (BLOCKED_BY_NEGATIVE_DATA)`**

---

## 1. Hiện Trạng Dữ Liệu Video Kiểm Thử Trên Ổ Đĩa

| Phân loại | Số video trong `video_dataset_list.csv` | Số video thực có trên ổ đĩa | Thời lượng thực có | Chỉ tiêu Đề cương (MT1/MT3) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Dương tính (Có cháy/khói)** | 44 (`POS_01..24`, `VN_POS_01..20`) | **24** (`POS_01..24`) | `863.98s` (`0.24h`) | ≥ 20 video | **ĐẠT (24/20 video)** |
| **Video Âm tính (Không cháy / Cảnh nhiễu)** | 28 (`NEG_25..49`, `VN_NEG_01,03,04`) | **0** | `0.00s` (`0.00h`) | ≥ 5.0 giờ | **THIẾU (`BLOCKED_BY_NEGATIVE_DATA`)** |

> Danh sách chi tiết 28 video âm tính bị thiếu trên ổ đĩa (`2.8622 giờ` trong catalog + `2.1378 giờ` cần thu thập bổ sung) đã được xuất tại `missing_negative_videos_manifest.csv` và `missing_negative_videos_manifest.json`.
> Báo cáo riêng về video âm tính đã được ghi nhận tại `negative_video_metrics.json`.

---

## 2. Bảng Đối Sánh 4 Pipeline Trên Toàn Bộ 24 Video Dương Tính (`full_24_positive`)

Cấu hình điểm vận hành đánh giá: `confidence_threshold = 0.35`, `K = 3`, `N = 8`, `cooldown_sec = 5.0s`, `max_event_gap_sec = 2.0s`.

| Chỉ số thực nghiệm | Pipeline A: Frame-Level Baseline (Chỉ YOLO, không K/N) | Pipeline B1: ByteTrack + K/N (Không Feature Gate, feat_thr=0.0) | Pipeline B2: ByteTrack + K/N + Feature Gate (feat_thr=0.02) | Pipeline C: Legacy IoU Temporal (IoU + K/N) |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng cảnh báo phát ra (Raw Alarms / DB Triggers)** | **2938** frame alarms | **129** DB alerts (**-95.61%**) | **128** DB alerts (**-95.64%**) | **128** DB alerts |
| **Số khoảng sự kiện sau khi gộp (`gap = 2.0s`)** | **53** events | **32** events | **31** events | **33** events |
| **Video-level Event Recall (Strict Match)** | `100.00%` | `100.00%` | `100.00%` | `100.00%` |
| **Video-level Primary Recall (Any Class)** | `95.83%` (23/24) | `95.83%` (23/24) | `95.83%` (23/24) | `95.83%` (23/24) |
| **Event-instance Recall** | `81.82%` | `72.73%` | `72.73%` | `72.73%` |
| **Event Precision** | `16.98%` | `25.00%` | **`25.81%`** | `24.24%` |
| **Event F1-Score** | `28.12%` | `37.21%` | **`38.10%`** | `36.36%` |
| **Mean Detection Delay (Strict Class Match)** | `0.60 s` | `0.50 s` | `0.52 s` (≤ 10 s) | `0.47 s` |
| **Mean Detection Delay (Từ mốc khởi phát sớm)** | `0.97 s` | `1.85 s` | `1.87 s` (≤ 10 s) | `1.79 s` |
| **Cảnh báo nổ ra trước mốc cháy (`pre_event_alarms`)** | `3` | `0` | **`0`** | `0` |
| **Sự kiện không khớp GT (`unmatched_positive`)** | `44` | `24` | **`23`** | `25` |
| **Negative False Alarms per Hour (FAR/giờ)** | `UNAVAILABLE` | `UNAVAILABLE` | `UNAVAILABLE` | `UNAVAILABLE` |
| **Detector-only Latency / FPS (CPU)** | `18.30 ms` (`54.6 FPS`) | `18.30 ms` (`54.6 FPS`) | `18.30 ms` (`54.6 FPS`) | `18.30 ms` (`54.6 FPS`) |
| **End-to-End Pipeline Latency / FPS (CPU)** | `18.30 ms` (`54.6 FPS`) | `18.89 ms` (`52.9 FPS`) | **`19.14 ms` (`52.2 FPS`)** | `18.33 ms` (`54.5 FPS`) |

> **Phân rã chi phí tính toán tăng thêm (Overhead) của Pipeline B2 trên CPU:**
> - `ByteTrackAdapter.update`: `0.561 ms/frame`
> - `TemporalFeatureExtractor.update`: `0.262 ms/frame`
> - `TemporalVerifier.update`: `0.017 ms/frame`
> - **Tổng Overhead toàn pipeline:** `0.840 ms/frame`

---

## 3. Bảng Đối Sánh 4 Pipeline Trên Tập Validation (12 Video Lẻ - `val_12_positive`)

| Chỉ số thực nghiệm | Pipeline A: Frame Baseline | Pipeline B1: ByteTrack (No Gate) | Pipeline B2: ByteTrack + Feature Gate | Pipeline C: Legacy IoU |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng cảnh báo phát ra** | **1920** frame alarms | **79** DB alerts (**-95.89%**) | **79** DB alerts (**-95.89%**) | **78** DB alerts |
| **Số sự kiện sau gộp (gap=2.0s)** | **26** events | **16** events | **16** events | **17** events |
| **Video Recall (Strict Match)** | `100.00%` | `100.00%` | `100.00%` | `100.00%` |
| **Event Precision** | `11.54%` | `18.75%` | **`18.75%`** | `17.65%` |
| **Event F1-Score** | `20.69%` | `31.58%` | **`31.58%`** | `30.00%` |
| **Mean Detection Delay** | `0.00 s` | `0.43 s` | `0.50 s` | `0.43 s` |

---

## 4. Bảng Đối Sánh 4 Pipeline Trên Tập Test Độc Lập (12 Video Chẵn - `test_12_positive`)

| Chỉ số thực nghiệm | Pipeline A: Frame Baseline | Pipeline B1: ByteTrack (No Gate) | Pipeline B2: ByteTrack + Feature Gate | Pipeline C: Legacy IoU |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng cảnh báo phát ra** | **1018** frame alarms | **50** DB alerts (**-95.09%**) | **49** DB alerts (**-95.19%**) | **50** DB alerts |
| **Số sự kiện sau gộp (gap=2.0s)** | **27** events | **16** events | **15** events | **16** events |
| **Video Recall (Strict Match)** | `100.00%` | `100.00%` | `100.00%` | `100.00%` |
| **Event Precision** | `22.22%` | `31.25%` | **`33.33%`** | `31.25%` |
| **Event F1-Score** | `34.29%` | `41.67%` | **`43.48%`** | `41.67%` |
| **Mean Detection Delay** | `0.89 s` | `0.53 s` | `0.53 s` | `0.49 s` |

---

## 5. Kết Quả Quét Lưới Tham Số (Grid Search) Trên Validation Set & Đánh Giá Trên Test Set

- **Validation Set (12 video lẻ):** `POS_01, POS_03, POS_05, POS_07, POS_09, POS_11, POS_13, POS_15, POS_17, POS_19, POS_21, POS_23` (`456.65s`).
- **Test Set khóa cố định (12 video chẵn):** `POS_02, POS_04, POS_06, POS_08, POS_10, POS_12, POS_14, POS_16, POS_18, POS_20, POS_22, POS_24` (`407.33s`).
- **Tách biệt 100%:** Hai tập Validation và Test không có video nào trùng lặp (`val_ids.isdisjoint(test_ids) == True`).
- **Nguyên tắc so sánh Baseline:** Mỗi ứng viên `(conf, K, N, ...)` được so sánh với Frame-level baseline tại chính ngưỡng `conf` đó (`baseline_by_conf`).
- **Tổng số cấu hình đã quét:** `432` tổ hợp (`confidence_threshold` ∈ {0.25, 0.35, 0.45}, $K$ ∈ {2, 3, 4, 5}, $N$ ∈ {5, 8, 10, 12}, `feature_threshold` ∈ {0.0, 0.01, 0.02}, `cooldown_sec` ∈ {0.0, 5.0, 12.0}).
- **Kết luận chọn điểm vận hành:** `NO_FEASIBLE_OPERATING_POINT`
  - Lý do: NO_FEASIBLE_OPERATING_POINT (BLOCKED_BY_NEGATIVE_DATA: Chưa có >= 5 giờ video âm tính độc lập trên ổ đĩa để xác minh điều kiện giảm >= 50% FAR/giờ. Đã chọn cấu hình provisional tối ưu Recall và Delay trên tập Validation dương tính).
- **Cấu hình Provisional tối ưu trên Validation Set:** `conf=0.35, K=3, N=8, feat_thr=0.02, cooldown=5.0s`.
