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

## Hiện trạng code (cập nhật 29/10/2026 – sửa mục này khi có thay đổi)
- Đã có: YOLO `best.pt` (smoke=0, fire=1); Desktop GUI 1 luồng trong `app.py`, `detector.py`; Web Dashboard 9 camera (3×3) Dark Mode PCCC trong `web_app.py` cùng gói module `src/` (`src/config.py`, `src/detector.py`, `src/temporal_verifier.py` với cửa sổ trượt k/N + IoU tracking + vùng loại trừ ROI + 2 mức cảnh báo SUSPECT/FIRE + Cooldown chống báo lặp, `src/camera_manager.py` đa luồng 9 camera hỗ trợ 4 chế độ `VIDEO_TEST`/`WEBCAM`/`IP_CAMERA`/`MULTI_CAMERA` + tự kết nối lại `RECONNECTING` + kiểm tra kết nối xem trước + ẩn mật khẩu RTSP + tách FPS đọc/FPS suy luận + cô lập lỗi, `src/database.py` & `alert_manager.py` SQLite + snapshot + lọc đa chiều + xuất CSV); kiểm thử tự động `tests/test_dashboard_system.py` & `test_integration.py`; khảo sát tài liệu IEEE `literature_survey.md`; chuẩn hóa & chia phân tầng tập D-Fire 3 tập độc lập Train 17.221 / Val 2.153 / Test 2.153 (`data_dfire.yaml`, `scripts/split_dfire_val_test.py`, kiểm định MD5 & làm sạch nhãn tại `scripts/verify_dfire_integrity.py`, `docs/dfire_verification_report.md`); danh mục 60 video kiểm định (`data/video_dataset_list.csv`, hiện có 1,32h video âm tính, cần bổ sung đủ ≥ 5h); 1.050 khung hình Việt Nam gán nhãn chia 60/20/20 theo video có làm mờ mặt/biển số (`datasets/vn_dataset/`, `data_vn.yaml`); Baseline B1 (`baselines/baseline_b1.py`, `docs/baseline_b1_report.md`); Baseline B2 (`baselines/baseline_b2_mobilenet.py`, `docs/baseline_b2_report.md`, đã đo FPS CPU, chưa lưu checkpoint `.pt`); Pipeline huấn luyện MT2 (`scripts/train_benchmarks.py`, đã xong 3/15 runs là trọn bộ 3 seeds `yolov8n_seed42/123/456` đạt Val mAP@0.5 = 76.62% ± 0.19%, Test mAP@0.5 = 76.74% ± 0.19%); Script đánh giá đối sánh MT2 theo từng lớp (`scripts/evaluate_benchmarks.py`, `docs/benchmark_detectors_dfire.md`, `docs/tuan_3_checklist.md`).
- Chưa có: Quét tham số (k, N) bằng `event_eval.py` trên toàn bộ 60 video kiểm định, thích nghi miền (Tuần 5), Telegram/email, ONNX/OpenVINO, Docker, MediaMTX
