# Báo Cáo & Thảo Luận Chi Tiết Tiến Độ Tuần 3 (Mốc Kiểm Tra 1)

**Đề tài NCKH:** Nghiên cứu ứng dụng computer vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
**Giảng viên hướng dẫn:** ThS. Nguyễn Hữu Trung  
**Nhóm sinh viên thực hiện:**  
- **Lê Minh Thiên** (Chủ nhiệm đề tài): Mô hình phát hiện, baseline, thích nghi miền, chủ biên báo cáo/bài báo IEEE  
- **Mai Quốc Việt** (Thành viên chính): Dữ liệu video & gán nhãn, xác nhận theo thời gian, hệ thống đa luồng, dashboard  
**Giai đoạn:** Tuần 3 (02/11/2026 – 08/11/2026) – **Mốc kiểm tra 1 với GVHD**  
**Ngày cập nhật:** 29/10/2026  

---

## 📌 1. YÊU CẦU ĐỀ CƯƠNG VỀ MỤC TIÊU MT2 (TUẦN 3)

Theo đề cương NCKH đã được phê duyệt, các nhiệm vụ trọng tâm của Tuần 3 bao gồm:
1. **Hoàn thành 2 mô hình Baseline đối chứng:**
   - **Baseline B1:** Phương pháp thị giác máy tính truyền thống (Luật màu không gian HSV/YCbCr kết hợp trừ nền thích ứng Gaussian Mixture MOG2).
   - **Baseline B2:** Mạng học sâu phân loại ảnh nhẹ cấp độ khung hình (*Frame-level multi-label classification*) sử dụng kiến trúc **MobileNetV3-Small**.
2. **Huấn luyện chuẩn mực >= 4 mô hình Object Detection:**
   - Các kiến trúc: **YOLOv8n**, **YOLOv8s**, **YOLO11n**, **YOLO11s** và **RT-DETR-R18**.
   - Chuẩn hóa siêu tham số: Kích thước ảnh  = 640$, cùng cấu hình dữ liệu tăng cường (Augmentations: mosaic, fliplr, hsv, scale).
   - Chạy trên **3 Random Seeds ngẫu nhiên (42, 123, 456)** để báo cáo kết quả trung bình và độ lệch chuẩn (mean +- std) theo chuẩn công bố bài báo IEEE.
   - Chỉ tiêu kỹ thuật MT2: @0.5 >= 0.75$ (75%) trên tập chuẩn D-Fire.

---

## 🔍 2. ĐỐI SOÁT CÁC TẬP DỮ LIỆU HUẤN LUYỆN (DATASETS AUDIT)

| Tập dữ liệu | Quy mô | Vai trò & Mục đích trong đề tài | Trạng thái ở Tuần 3 |
| :--- | :---: | :--- | :--- |
| **D-Fire Chuẩn hóa** (data_dfire.yaml) | **21.527 ảnh**<br>(Train: 17.221<br>Val: 2.153<br>Test: 2.153) | Tập dữ liệu công khai tiêu chuẩn quốc tế dùng để huấn luyện và đánh giá benchmark các mô hình phát hiện cháy/khói. | **Sẵn sàng 100%**.<br>Đây là **tập duy nhất** được phép dùng để train trong Tuần 3. |
| **CCTV Việt Nam** (data_vn.yaml) | **1.050 ảnh**<br>(Train 60% / Val 20% / Test 20%) | Tập dữ liệu camera giám sát thực tế tại Việt Nam đã gán nhãn Ground Truth thủ công, che mờ mặt & biển số xe. | **Giữ nguyên cho Tuần 5**.<br>Tuyệt đối **không** trộn vào tập train Tuần 3 để phục vụ bài toán Thích nghi miền (Domain Adaptation - MT4). |
| **Video Kiểm định** (ideo_dataset_list.csv) | **60 video**<br>(24 POS, 25 NEG, 11 VN) | Đo đạc độ trễ phát hiện (s), số lần báo giả/giờ và recall sự kiện. | **Dành riêng cho Tuần 4**<br>(Đánh giá mô-đun xác nhận theo thời gian 	emporal_verifier.py). |

---

## 📊 3. TRẠNG THÁI CHI TIẾT CÁC MÔ HÌNH VÀ PHIÊN HUẤN LUYỆN (BENCHMARK RUNS)

Theo thiết kế thực nghiệm MT2, cần thực hiện tổng cộng **15 phiên huấn luyện Object Detection** (5 mô hình x 3 seeds) cùng với mạng phân loại Baseline B2.

### Bảng Kiểm Tra Tiến Độ Từng Mô Hình & Từng Seed (Thực tế trong 
uns/):

| Mô hình | Trọng số ban đầu | Seed 42 | Seed 123 | Seed 456 | Trạng thái & Kết quả thực tế |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **YOLOv8n** | yolov8n.pt |  **Hoàn thành 50/50 epoch**<br>(@0.5 = 76.36\%$, -95 = 42.95\%$) | ⚠️ **Dở dang**<br>(Dừng ở epoch 2/50 do ngắt tiến trình) |  **Chưa chạy** | Đã có 1 bản đạt chỉ tiêu đề cương (@0.5 >= 0.75$). |
| **YOLOv8s** | yolov8s.pt |  **Chưa chạy** |  **Chưa chạy** |  **Chưa chạy** | Còn thiếu cả 3 seed. |
| **YOLO11n** | yolo11n.pt |  **Chưa chạy** |  **Chưa chạy** |  **Chưa chạy** | Còn thiếu cả 3 seed *(Mới qua bước Dry-run kiểm tra pipeline)*. |
| **YOLO11s** | yolo11s.pt |  **Chưa chạy** |  **Chưa chạy** |  **Chưa chạy** | Còn thiếu cả 3 seed. |
| **RT-DETR-L** | 
tdetr-l.pt |  **Chưa chạy** |  **Chưa chạy** |  **Chưa chạy** | Còn thiếu cả 3 seed. |
| **Baseline B2** | MobileNetV3-Small |  Đã có mã nguồn & đo đạc hiệu năng | — | — | Đã có báo cáo kiểm thử trên 200 ảnh test CPU (72.8 FPS, Latency 13.73 ms). |

> **Tổng kết:** Hiện tại mới hoàn tất **1/15 phiên huấn luyện** (yolov8n_seed42), 1 phiên bị ngắt quãng giữa chừng (yolov8n_seed123), còn **thiếu 13 phiên huấn luyện** của các kiến trúc còn lại.

---

## ⚖️ 4. BÀI HỌC KHOA HỌC TỪ HAI BASELINE (B1 & B2) ĐỐI VỚI BỘ PHÁT HIỆN

Trong Tuần 3, nhóm đã hoàn thành việc xây dựng và thực nghiệm đối chiếu 2 baseline:

1. **Baseline B1 (Luật màu HSV/YCbCr + Trừ nền MOG2 - aselines/baseline_b1.py):**
   - *Tốc độ:* Rất nhanh (~60 FPS trên CPU thông thường).
   - *Nhược điểm:* Quá nhạy cảm với nguồn sáng đỏ, xe cộ, ánh nắng, dẫn đến tỷ lệ báo giả khung hình rất cao (~75% FPR).
2. **Baseline B2 (MobileNetV3-Small - aselines/baseline_b2_mobilenet.py):**
   - *Tốc độ:* Đạt **72.8 FPS** trên CPU Intel, độ trễ trung bình **13.73 ms/ảnh**, kích thước nhỏ nhẹ (~2.54M tham số).
   - *Nhược điểm chí tử:* Là bài toán phân loại toàn khung hình (*Classification*), **hoàn toàn không cung cấp tọa độ Bounding Box** (, y, w, h$). Do đó, **không thể tích hợp thuật toán theo dõi ByteTrack** và không thể phân tích biến động diện tích/chu vi theo thời gian ở Tuần 4.
3. **Kết luận khoa học:**
   - Bắt buộc phải sử dụng **Bộ phát hiện đối tượng (Object Detectors)** dạng YOLO / RT-DETR để cung cấp tọa độ vùng cháy/khói chính xác, phục vụ trực tiếp cho bộ lọc thời gian chống báo giả ở Tuần 4.

---

## 🚀 5. KẾ HOẠCH HÀNH ĐỘNG TIẾP THEO

### Bước 1: Hoàn thành bộ 3 seed cho YOLOv8n (Ưu tiên số 1)
Chạy nốt seed 123 và seed 456 của yolov8n để có bộ dữ liệu hoàn chỉnh 3 seed tính giá trị trung bình +- độ lệch chuẩn:
`ash
# Huấn luyện hoàn thiện các seed còn lại của YOLOv8n
python scripts/train_benchmarks.py --models yolov8n --seeds 123 456 --epochs 50 --batch 16
`

### Bước 2: Huấn luyện đại diện thế hệ mới YOLO11n
Huấn luyện thử nghiệm yolo11n (seed 42) để có mô hình thế hệ mới nhất đối sánh trực tiếp với YOLOv8n:
`ash
# Huấn luyện YOLO11n seed 42
python scripts/train_benchmarks.py --models yolo11n --seeds 42 --epochs 50 --batch 16
`

### Bước 3: Đánh giá tự động và tổng hợp báo cáo MT2
Sau khi các phiên huấn luyện hoàn thành, chạy script trích xuất chỉ số tự động:
`ash
python scripts/evaluate_benchmarks.py
`
Script sẽ tự động đọc các file 
esults.csv, tính toán các chỉ số khoa học và tạo báo cáo chuẩn IEEE tại docs/benchmark_detectors_dfire.md.

### Bước 4: Sẵn sàng tiến sang Tuần 4 (09/11 – 15/11/2026)
Trọng số tốt nhất từ yolov8n_seed42 (@0.5 = 76.36\%$) đã đạt chỉ tiêu MT2 (>= 0.75), hoàn toàn đủ điều kiện làm đầu vào để nhóm triển khai ngay module **Xác nhận theo thời gian** (src/temporal_verifier.py) và **Đánh giá sự kiện** (src/event_eval.py) trên 60 video kiểm định.
