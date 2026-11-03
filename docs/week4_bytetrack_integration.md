# TÀI LIỆU TÍCH HỢP BYTETRACK VÀ PIPELINE TEMPORAL VERIFICATION – TUẦN 4

**Dự án:** Nghiên cứu ứng dụng computer vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
**Tác giả triển khai:** Mai Quốc Việt (Branch: `tuần-4`)  
**Ngày cập nhật:** 03/11/2026  

---

## 1. Tổng Quan & Mục Tiêu Kỹ Thuật

Tuần 4 của đề tài nghiên cứu khoa học tập trung vào giải quyết bài toán chống báo động giả (False Alarm Reduction - FAR) trong giám sát camera thời gian thực theo Mục tiêu MT3 & MT5 của Đề cương NCKH:
- **MT3:** Giảm $\ge 50\%$ số lần báo động giả trên mỗi giờ (FAR) so với cảnh báo từng khung đơn lẻ; Event Recall giảm $\le 5$ điểm phần trăm; Độ trễ phát hiện trung bình $\le 10$ giây.
- **MT5:** Đảm bảo duy trì khả năng giám sát lưới đa camera (tối đa 9 camera 3×3) với tốc độ $\ge 15$ FPS/luồng trên GPU và $\ge 5$ FPS trên CPU, không gây giật lag luồng đọc camera (`capture_thread`).

Hệ thống đã chính thức tích hợp thuật toán **ByteTrack** (thông qua `ultralytics.trackers.byte_tracker.BYTETracker`) kết hợp cùng cơ chế cửa sổ trượt $k/N$, lọc vùng loại trừ ROI, phân tách đặc trưng thời gian (Temporal Features), và giãn cách cảnh báo (Cooldown).

---

## 2. So Sánh Kiến Trúc Hệ Thống (Trước vs Sau Tích Hợp)

### 2.1. Kiến Trúc Cũ (Trước Tuần 4)
```
Camera Frame (RTSP / Video / Webcam)
    ↓
SharedFireSmokeDetector (YOLOv8 / YOLO11)
    ↓
Raw Detection Boxes [x1, y1, x2, y2, conf, label]
    ↓
TemporalVerifier (IoU Tracking thủ công đơn giản)
    - Ghép cặp IoU frame-to-frame trực tiếp
    - Bỏ qua các detection có confidence thấp (< conf_thresh)
    - Dễ đứt gãy track_id khi khói đổi hình dạng hoặc lửa bập bùng
    ↓
Sliding Window k/N + ROI Exclusion + Cooldown
    ↓
Alert Decision -> EventDatabase (SQLite) -> Web Dashboard
```

**Nhược điểm của kiến trúc cũ:**
1. *Đứt gãy track (ID Switch & Track Fragmentation):* Ngọn lửa và khói là vật thể phi cứng (non-rigid objects), diện tích và hình thái thay đổi liên tục khiến chỉ số IoU giữa hai khung hình liên tiếp có thể sụt giảm đột ngột, dẫn đến việc bộ ghép cặp IoU đơn giản gán ID mới liên tục.
2. *Mất đối tượng khi detector suy giảm confidence:* Khi có khói loãng hoặc lửa bị che khuất một phần, confidence tạm thời giảm xuống khiến detector bỏ sót khung hình. Bộ IoU tracking cũ xóa track ngay sau đó, làm gián đoạn chuỗi tích lũy $k/N$.

---

### 2.2. Kiến Trúc Mới Đã Tích Hợp (Sau Tuần 4)
```
Camera Frame (RTSP / Video / Webcam)
    ↓
CameraStreamWorker (Capture Thread - 25-30 FPS)
    ↓ (Non-blocking queue & thread decoupling)
CameraStreamWorker (Inference Thread)
    ↓
SharedFireSmokeDetector (best.pt - FP16 / CPU Fallback)
    ↓
Raw Detection Boxes
    ↓
ByteTrackAdapter (Chính thức từ Ultralytics BYTETracker)
    - association 2 giai đoạn (High-conf: 0.5, Low-conf: 0.1)
    - Khôi phục track qua Kalman Filter khi bị che khuất (track_buffer: 30)
    - Tách biệt tracker state độc lập theo từng CameraStreamWorker
    - Tách biệt tracking theo class (per_class=True) tránh nhầm lẫn giữa Lửa & Khói
    ↓
Tracked Detection Boxes (với stable track_id)
    ↓
TemporalFeatureExtractor (src/temporal_features.py)
    - Centroid speed & Area growth rate
    - Bounded ROI intensity variance & Flicker score
    - Lọc bỏ các nguồn sáng tĩnh (stationary lights, reflections)
    ↓
TemporalVerifier (Cửa sổ k/N + ROI Exclusion + Cooldown)
    - Tiếp nhận track_id ổn định từ ByteTrack (tracks_assigned=True)
    - Phân cấp cảnh báo: SAFE -> SUSPECT (k_suspect) -> FIRE (k_confirm)
    - Ngăn báo lặp bằng alert_cooldown_sec (12s)
    ↓
Alert Decision -> EventDatabase (SQLite + Snapshot JPEG) -> Web Dashboard (9 Camera Grid)
```

---

## 3. Chi Tiết Các Module Thay Đổi

### 3.1. `src/config.py`
- Bổ sung dataclass `ByteTrackSettings`:
  ```python
  @dataclass
  class ByteTrackSettings:
      track_high_thresh: float = 0.5
      track_low_thresh: float = 0.1
      new_track_thresh: float = 0.6
      track_buffer: int = 30
      match_thresh: float = 0.8
      fuse_score: bool = False
  ```
- Thêm trường `bytetrack: ByteTrackSettings = field(default_factory=ByteTrackSettings)` vào `SystemSettings` để tập trung toàn bộ cấu hình, không hardcode tham số rải rác.

### 3.2. `src/bytetrack_adapter.py` (Mới)
- Class `ByteTrackAdapter` bọc trực tiếp `ultralytics.trackers.byte_tracker.BYTETracker`.
- Lớp nội bộ `_DetectionBatch` thực hiện giao diện duck-typing với các thuộc tính `xywh`, `conf`, `cls` và phương thức `__getitem__` hỗ trợ boolean indexing.
- Hỗ trợ `per_class=True` với bộ sinh ID liên tục (`itertools.count(1)`), đảm bảo ngọn lửa và làn khói ở cùng tọa độ không bao giờ bị ghép cặp chéo với nhau.
- Xử lý mượt mà các trường hợp biên: khung hình không có detection, khung hình `None` hoặc lỗi định dạng, loại trừ detection dưới ngưỡng khởi tạo track mới (`new_track_thresh`).

### 3.3. `src/temporal_verifier.py`
- Bổ sung tham số `tracks_assigned: bool = False` trong `update(detections, tracks_assigned=False)`.
- Khi `tracks_assigned=True` (hoặc các detection đã có `track_id > 0` từ ByteTrack), phương thức `_record_external_tracks()` cập nhật lịch sử diện tích, số lần xuất hiện và làm sạch track hết hạn mà **không ghi đè** `track_id`.
- Khi `tracks_assigned=False`, hệ thống tự động fallback về thuật toán ghép cặp IoU nội bộ cũ, đảm bảo tương thích ngược 100% với các test cũ.
- Giữ nguyên toàn bộ cơ chế cửa sổ trượt $k/N$, lọc vùng loại trừ ROI theo camera, tính toán nhãn chủ đạo (`dominant_label`), và cơ chế giãn cách `cooldown_sec`.

### 3.4. `src/camera_manager.py`
- Tích hợp `ByteTrackAdapter` và `TemporalFeatureExtractor` riêng biệt vào từng thực thể `CameraStreamWorker`.
- Đảm bảo tính cô lập hoàn toàn giữa 9 camera: mỗi camera quản lý Kalman filter và ID counter riêng, không gây race condition và không chia sẻ bộ nhớ tracker.
- Khi camera ngắt kết nối, dừng (`stop()`), khởi chạy (`start()`), hoặc tự kết nối lại (`reconnect()`), toàn bộ trạng thái tracking của camera đó được reset sạch sẽ, tránh lưu lại track cũ của video trước.
- Luồng suy luận `_infer_loop` thực hiện quy trình tuần tự:
  `YOLO Detector -> ByteTrack -> Temporal Features -> TemporalVerifier -> Alert DB & Dashboard`.

---

## 4. Hướng Dẫn Cài Đặt & Chạy Kiểm Thử

### 4.1. Cài Đặt Môi Trường
```powershell
# Kích hoạt môi trường ảo Python 3.11
.\.venv\Scripts\Activate.ps1

# Cài đặt các gói phụ thuộc (bao gồm ultralytics và lap)
pip install -r requirements.txt
```

### 4.2. Chạy Bộ Kiểm Thử Tự Động (Unit & Regression Tests)
```powershell
# Chạy toàn bộ 26 unit tests trong tests/
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"

# Chạy kiểm thử toàn diện hệ thống Web Dashboard 9 camera
.\.venv\Scripts\python.exe tests/test_dashboard_system.py
```

### 4.3. Chạy Đánh Giá Benchmark Đối Sánh (Week 4)
```powershell
.\.venv\Scripts\python.exe scripts/benchmark_week4_bytetrack.py
```

### 4.4. Khởi Động Web Dashboard 9 Camera
```powershell
streamlit run web_app.py
```
