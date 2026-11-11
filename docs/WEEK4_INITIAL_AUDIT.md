# BÁO CÁO KIỂM TRA REPOSITORY VÀ TRẠNG THÁI BAN ĐẦU (GIAI ĐOẠN 0)
**Dự án:** Computer Vision-based Fire and Smoke Detection and Early Warning from Surveillance Camera Footage  
**Ngày thực hiện:** 11/11/2026  
**Chủ biên / Kiểm tra:** Mai Quốc Việt & Lê Minh Thiên  

---

## 1. Kiểm tra Git & Phiên bản
- **Branch hiện tại:** `main`
- **Commit mới nhất:** `256065e7` (`feat: cap nhat 192 video, toan bo anh va nhan 2 tap Viet Nam moi`)
- **Remote tracking:** `origin/main` (đồng bộ hoàn toàn, 0 commit lệch)
- **Trạng thái working tree:** Sạch hoàn toàn, không có uncommitted changes gây mất mát dữ liệu
- **Quy tắc bảo vệ:** Không reset, không checkout đè dữ liệu người dùng, bảo toàn toàn bộ kết quả benchmark đã thực hiện.

---

## 2. Cấu trúc Dự án Thực tế
- `src/`: Gói module lõi gồm:
  - `src/detector.py`: `SharedFireSmokeDetector` hỗ trợ YOLOv8/YOLO11, tự động kích hoạt CUDA FP16 (`NVIDIA GeForce RTX 4050 Laptop GPU`) hoặc CPU đa luồng, warmup, fuse conv+bn.
  - `src/bytetrack_adapter.py`: `ByteTrackAdapter` bọc Ultralytics `BYTETracker` 2 giai đoạn association, tách biệt tracker per-class (FIRE / SMOKE), quản lý ID độc lập và hỗ trợ `reset()`.
  - `src/temporal_verifier.py`: Bộ xác nhận theo thời gian `TemporalVerifier` theo dõi độc lập theo `track_id`, cửa sổ trượt $k/N$, kiểm tra vùng ROI loại trừ, cổng biến động `feature_threshold`, Cooldown nhân quả và 2 mức cảnh báo `SUSPECT` / `FIRE`.
  - `src/temporal_features.py`: `TemporalFeatureExtractor` trích xuất độ chớp cường độ ROI (flicker), độ biến thiên diện tích và tốc độ lan tỏa khói.
  - `src/event_eval.py` & `src/week4_event_eval.py`: Công cụ chuẩn hóa đánh giá cấp sự kiện và quét lưới tham số Grid Search.
  - `src/camera_manager.py`: Quản trị đa luồng tối đa 9 camera với vòng đời máy trạng thái chuẩn.
- `scripts/`:
  - `scripts/benchmark_week4_bytetrack.py`: Benchmark thực nghiệm đối sánh 4 Pipeline (A, B1, B2, C) và Grid Search 432 tổ hợp tham số.
  - `scripts/run_week4_audit.py`: Kiểm toán dữ liệu thực tế trên ổ đĩa.
  - `scripts/generate_week4_docs.py`: Sinh hồ sơ Ground Truth và báo cáo nghiệm thu.
- `datasets/`:
  - `datasets/vn_dataset/`: 3.121 ảnh và 3.121 file nhãn YOLO (.txt) Việt Nam, chia 60/20/20 theo video (Train: 1.868, Val: 623, Test: 630).
  - `datasets/dfire/`: 21.527 ảnh và 21.527 nhãn D-Fire (Train: 17.221, Test: 4.306).
  - `negative_videos/`: 143 video vật lý (80 POS, 63 NEG) bổ sung từ các đợt thu thập thực tế tại Việt Nam.
  - `datasets/test_videos/`: 24 video dương tính gốc (`posVideo1..24`) và 16 video âm tính gốc (`negVideo1..16`).
- `runs/benchmarks/week4_final/`: Toàn bộ artifact thực nghiệm Tuần 4 (`benchmark_report.md`, `best_operating_point.json`, `comparison.csv`, `metrics.json`, `grid_search.json`, `frame_level_events.json`, `temporal_confirmed_events.json`).

---

## 3. Kiểm tra Pipeline và Cấu hình Thực tế
- **Model Checkpoint:** `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt` (hoặc `best.pt`), kích thước 6.0MB, 73 layers, 3.006.038 tham số, mAP@0.5 = 76.78% trên D-Fire Test.
- **Thứ tự lớp:** `0: SMOKE`, `1: FIRE` (khớp 100% giữa YAML, Model và Code).
- **Độ phân giải đầu vào:** `imgsz = 416` (tối ưu tốc độ thời gian thực ~32.6 FPS FP16 GPU, ~55.2 FPS CPU).
- **Ngưỡng Confidence phát hiện:** `0.25` (sàng lọc sơ bộ cho Tracker) và `0.35` (ngưỡng đánh giá xác nhận cảnh báo).
- **Ngưỡng Temporal Confirmation:** Cửa sổ trượt $N = 8$ frame, số frame xác nhận $K = 3$ frame ($k/N = 3/8$), mức nghi ngờ $k = 2$.
- **Cổng biến động đặc trưng (Feature Gate):** `feature_threshold = 0.02` (yêu cầu độ chớp/biến thiên hình thái $\ge 0.02$ để loại trừ vật thể tĩnh có màu giống lửa).
- **Thời gian Cooldown:** `5.0 s` (hoạt động độc lập giữa FIRE và SMOKE, chống spam cảnh báo).
- **Gom nhóm sự kiện (Event Merging):** Dung sai thời gian `max_event_gap_sec = 2.0 s`.
- **Dung sai khớp sự kiện với Ground Truth:** `temporal_tolerance_sec = 10.0 s`.

---

## 4. Danh sách vấn đề phát hiện và Biện pháp xử lý
1. **Đường dẫn video trong catalog lệch vị trí vật lý:** File `video_dataset_list.csv` trỏ tới các đường dẫn tương đối giả định (`videovietnam/`, `VN_NEG_05/`), trong khi người dùng đã lưu trữ tập trung tại thư mục `negative_videos/`.  
   *Biện pháp:* Đã chạy script chuẩn hóa tự động cập nhật đúng 143 đường dẫn vật lý thực tế trên ổ đĩa.
2. **UTF-8 BOM trong file CSV:** Gây lỗi `KeyError: 'video_id'` khi mở bằng `encoding="utf-8"`.  
   *Biện pháp:* Đã đổi sang `encoding="utf-8-sig"` trong toàn bộ code đọc catalog.
3. **Bộ đếm ID toàn cục của ByteTrack:** Khi reset hoặc tạo tracker mới, `BaseTrack._count` gây lệch chỉ số track đầu tiên (bị thành 2 thay vì 1).  
   *Biện pháp:* Đã cô lập `_ids = itertools.count(1)` per tracker instance trong `ByteTrackAdapter`.
4. **Kết quả kiểm thử unit test:** Toàn bộ 62 unit tests trong `tests/` hiện đã PASS 100%.

