---
trigger: always_on
---

# Đề cương NCKH – Phát hiện & cảnh báo cháy từ camera giám sát

Tài liệu gốc: `DeCuong_NCKH_PhatHienChay_Camera.docx` (thư mục gốc dự án). Mọi thay đổi code phải bám theo đề cương này. Nếu yêu cầu của người dùng mâu thuẫn với đề cương, hãy nêu rõ mâu thuẫn trước khi làm.

## Thông tin
- Đề tài: Nghiên cứu ứng dụng computer vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát
- Thời gian: 8 tuần, 19/10/2026 – 13/12/2026. GVHD: ThS. Nguyễn Hữu Trung
- Lê Minh Thiên (chủ nhiệm): mô hình phát hiện, baseline, thích nghi miền, chủ biên báo cáo/bài báo
- Mai Quốc Việt (người dùng repo này): dữ liệu video & gán nhãn, xác nhận theo thời gian, hệ thống đa luồng camera, gửi cảnh báo, dashboard, đo hiệu năng triển khai

## Mục tiêu có chỉ tiêu
- MT1 Dữ liệu: D-Fire là tập huấn luyện chính; ≥ 20 video có cháy/khói; ≥ 5 giờ video không cháy có cảnh nhiễu; ≥ 1.000 khung hình Việt Nam gán nhãn tay
- MT2 Bộ phát hiện: ≥ 4 mô hình (YOLOv8n/s, YOLO11n/s, RT-DETR-R18) + 2 baseline (B1 luật màu HSV/YCbCr + trừ nền; B2 MobileNetV3); mAP@0.5 ≥ 0,75 trên D-Fire
- MT3 Giảm báo giả: giảm ≥ 50% báo động giả/giờ so với cảnh báo từng khung; recall sự kiện giảm ≤ 5 điểm %; độ trễ TB ≤ 10 s
- MT4 Thích nghi miền: 3 chiến lược (toàn bộ / đóng băng backbone / đóng băng + replay 20% D-Fire); +≥ 5 điểm mAP@0.5 tập Việt Nam, mAP D-Fire giảm ≤ 2 điểm
- MT5 Hệ thống: ≥ 4 luồng RTSP; ≥ 15 FPS/luồng GPU, ≥ 5 FPS CPU (ONNX/OpenVINO); cảnh báo kèm ảnh tới điện thoại ≤ 5 s

## Quy trình 6 bước
1. Dữ liệu: chuẩn hóa D-Fire; cắt video; gán nhãn bằng CVAT/Roboflow; ghi mốc giây bắt đầu cháy mỗi video
2. Baseline B1, B2
3. Huấn luyện bộ phát hiện: imgsz 640, cùng số epoch, có tăng cường dữ liệu; 3 seed, báo cáo trung bình ± độ lệch chuẩn
4. Xác nhận theo thời gian (`temporal_verifier.py`): ByteTrack; cảnh báo khi ≥ k/N khung + biến động đặc trưng; vùng loại trừ theo camera; 2 mức "nghi ngờ" / "cháy"; quét (k, N) và ngưỡng trên tập kiểm định. Đánh giá bằng `event_eval.py` (recall sự kiện, độ trễ, báo giả/giờ)
5. Thích nghi miền: đo mAP cả tập Việt Nam và D-Fire để tính mức quên
6. Triển khai: ONNX/OpenVINO, TensorRT FP16; đa luồng RTSP (giả lập bằng MediaMTX); Telegram Bot/email kèm ảnh có khung bao; dashboard có trạng thái camera, lịch sử, nút xác nhận/bỏ qua; SQLite; Docker

## Tiến độ
- Tuần 1 (19–25/10): khảo sát tài liệu, phân tích D-Fire, danh sách nguồn video, repo
- Tuần 2 (26/10–01/11): bộ video kiểm thử + mốc thời gian, gán nhãn khung hình Việt Nam, B1
- Tuần 3 (02–08/11): huấn luyện các mô hình + B2 — mốc kiểm tra 1
- Tuần 4 (09–15/11): temporal_verifier.py, event_eval.py
- Tuần 5 (16–22/11): thích nghi miền
- Tuần 6 (23–29/11): ablation, xuất mô hình, đo FPS, phân tích ≥ 50 lỗi — mốc kiểm tra 2
- Tuần 7 (30/11–06/12): hệ thống ≥ 4 luồng, Telegram/email, dashboard, Docker, chạy thử ≥ 2 giờ
- Tuần 8 (07–13/12): báo cáo, bài báo IEEE 6–8 trang, README, slide, video demo — nghiệm thu

## Ràng buộc bắt buộc
- Không bao giờ đề xuất tự đốt lửa để quay dữ liệu; lửa thật chỉ quay trong diễn tập PCCC có thẩm quyền
- Làm mờ khuôn mặt và biển số trước khi công bố dữ liệu
- Chia khung hình Việt Nam 60/20/20 theo video (không để khung của cùng video ở hai tập)
- Nhãn trong `FireSmokeLabeler/dataset` do chính `best.pt` tự sinh: không dùng làm ground truth để đánh giá hay báo cáo
- Ngoài phạm vi: camera nhiệt, kết nối tủ báo cháy trung tâm, điều khiển chữa cháy
- Bài báo Nested Learning chỉ là cơ sở lý thuyết cho thích nghi miền; không cài kiến trúc Hope/CMS
- Báo cáo số liệu trung thực, kể cả khi không đạt chỉ tiêu

## Hiện trạng code (cập nhật 10/11/2026 – sửa mục này khi có thay đổi)
- Đã có: YOLO `best.pt` (smoke=0, fire=1); Desktop GUI 1 luồng trong `app.py`, `detector.py`; Web Dashboard **FIREWATCH – INTELLIGENT MONITORING STATION (V2)** Dark Industrial PCCC trong `web_app.py` (cơ chế kết nối thủ công `MANUAL START` khởi động mặc định ở trạng thái `STOPPED – Chưa gán nguồn` không tự động gắn file video mẫu vào camera (`auto_assign_videos=False`), tự động dò quét camera phần cứng đang hoạt động (`detect_available_cameras`) để chọn gán vào ô camera, tính năng `🎬 Test Cam File` cho phép tải lên hoặc chọn file video để kiểm thử trên từng camera cụ thể, nút `Bắt đầu giám sát` / `Dừng` & áp dụng cấu hình động không gián đoạn camera đang chạy, hỗ trợ chọn 1/2/4/6/9 camera & Tùy chọn, 3 chế độ bố cục Lưới/Ưu tiên 1 cam/Danh sách, Phóng to & ROI, Trung tâm cảnh báo 3 tab + Modal phóng to + phân trang + tắt âm báo còi hú ngay khi xác nhận/bỏ qua (`acknowledge_event`), Thống kê tổng quan + Lịch sử sự kiện + Biểu đồ sự kiện theo giờ, Quản lý camera Thêm/Sửa/Xóa/Gỡ nguồn có xác nhận & ROI Editor, 3 chế độ hiệu năng Low Resource/Balanced/High Performance) cùng gói module `src/` (`src/config.py` cấu hình ByteTrack & `get_available_test_videos`, `src/detector.py` tối ưu fuse/warmup/FP16, `src/bytetrack_adapter.py` tích hợp ByteTrack chính thức từ Ultralytics 2 giai đoạn association + per-class separation + cô lập ID per-adapter, `src/temporal_verifier.py` với cửa sổ trượt k/N độc lập theo từng track_id và phân tách lớp FIRE/SMOKE + hỗ trợ `tracks_assigned=True` tiếp nhận ByteTrack ID + cổng biến động `feature_threshold` + vùng loại trừ ROI + 2 mức cảnh báo SUSPECT/FIRE + Cooldown nhân quả chống báo lặp mà không chặn sự kiện mới sau Cooldown, `src/temporal_features.py` tính độ chớp ROI/variance/tốc độ lan tỏa diện tích, `src/event_eval.py` công cụ đánh giá cấp sự kiện chuẩn hóa & quét lưới tham số Grid Search cùng wrapper tương thích `src/week4_event_eval.py`, `src/camera_manager.py` đa luồng 9 camera với máy trạng thái vòng đời `STOPPED`/`CONNECTING`/`RUNNING`/`RECONNECTING`/`ERROR`/`PAUSED` + `start_monitoring`/`stop_monitoring`/`remove_camera`/`detect_available_cameras`/`assign_active_hardware_cameras`/`assign_test_video_to_camera`/`detach_camera_source` + tách biệt 100% tracker state và feature extractor theo từng camera + dừng thật sự worker của camera không được chọn + truyền `feature_map` và `timestamp_sec` vào `verifier` + tự reset khi reconnect + nén JPEG C++ chống lag + 4 chế độ `VIDEO_TEST`/`WEBCAM`/`IP_CAMERA`/`MULTI_CAMERA` + kiểm tra kết nối xem trước + ẩn mật khẩu RTSP + cô lập lỗi, `src/database.py` & `alert_manager.py` SQLite + snapshot + lọc đa chiều + xuất CSV + tắt còi khi xác nhận); toàn bộ 106 tests (unit + system) pass 100%; khảo sát tài liệu IEEE `literature_survey.md`; chuẩn hóa & chia phân tầng tập D-Fire 3 tập độc lập Train 17.221 / Val 2.153 / Test 2.153; danh mục 192 video kiểm định (`data/video_dataset_list.csv`: 104 video dương tính, 88 video âm tính; trong đó có 79 video âm tính thực tế 7.20 giờ tại `datasets/negative_videos` và `datasets/test_videos/neg`); 3.121 khung hình Việt Nam gán nhãn chia đúng 60/20/20 theo video độc lập có làm mờ mặt/biển số (`datasets/vn_dataset/`); Baseline B1, B2; Pipeline huấn luyện MT2; **Hoàn tất 100% nghiệm thu khoa học Tuần 4 (`docs/WEEK4_FINAL_ACCEPTANCE_REPORT.md`)**: Full inference 79 video âm tính (7.20h, 929.939 frames) đo lường FAR giảm 57.39% từ 137.17/h (Pipeline A) xuống 58.45/h (Pipeline B2), 95% Poisson CI [53.00, 61.92], độ trễ 2.13s, recall sự kiện duy trì 95.83% (suy giảm 0%), chính thức khóa Operating Point OFFICIAL (conf=0.35, K=3, N=8, feat=0.02, cooldown=5.0s, gap=2.0s).
- Chưa có: Thích nghi miền (Tuần 5), Telegram/email, ONNX/OpenVINO, Docker, MediaMTX
- Chưa có: Thích nghi miền (Tuần 5), Telegram/email, ONNX/OpenVINO, Docker, MediaMTX
