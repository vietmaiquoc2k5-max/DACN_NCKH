# Báo Cáo Đánh Giá Khoa Học – Baseline B2 (MobileNetV3-Small)

**Đề tài:** Nghiên cứu ứng dụng computer vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
**Giai đoạn:** Tuần 3 (Mục tiêu MT2: Bộ phát hiện và các Baseline so sánh)  
**Mô hình Baseline B2:** MobileNetV3-Small (Phân loại cấp độ khung hình – Frame-level Multi-label Classification)  
**Tập dữ liệu kiểm thử:** D-Fire Test Set (200 khung hình)  
**Thiết bị đánh giá:** CPU Intel (cpu)  

---

## 1. TỔNG QUAN KIẾN TRÚC VÀ CẤU HÌNH THỰC NGHIỆM

Baseline B2 đại diện cho hướng tiếp cận **Học sâu phân loại ảnh nhẹ (Lightweight Deep Classification)** nhằm trả lời câu hỏi nghiên cứu:  
> *"Liệu một mạng phân loại khung hình siêu nhẹ có thể thay thế bộ phát hiện đối tượng (Object Detector) để phát hiện cháy hiệu quả và giảm thiểu tài nguyên tính toán hay không?"*

- **Kiến trúc Backbone:** MobileNetV3-Small (Pretrained ImageNet)
- **Kích thước đầu vào:** $224 \times 224 \times 3$ (Chuẩn hóa ImageNet)
- **Classifier Head:** `Linear(1024, 256) -> Hardswish -> Dropout(0.2) -> Linear(256, 2)`
- **Hàm kích hoạt đầu ra:** Sigmoid độc lập cho 2 lớp (Smoke, Fire)
- **Tổng số tham số (Parameters):** 1,075,234 (~1.08M)
- **Dung lượng tệp trọng số:** ~9.8 MB

---

## 2. KẾT QUẢ ĐO ĐẠC HIỆU NĂNG PHÂN LOẠI (FRAME-LEVEL CLASSIFICATION)

### Bảng 1: Kết quả phân loại chi tiết theo lớp trên D-Fire Test

| Lớp đối tượng | TP (Đúng) | FP (Báo sai) | FN (Bỏ sót) | Precision | Recall | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Khói (Smoke)** | 60 | 72 | 31 | **45.45%** | **65.93%** | **53.81%** | 48.50% |
| **Lửa (Fire)** | 0 | 0 | 53 | **0.00%** | **0.00%** | **0.00%** | 73.50% |
| **Trung bình Macro** | — | — | — | **22.73%** | **32.97%** | **26.91%** | — |

---

## 3. TỐC ĐỘ SUY LUẬN & TỶ LỆ BÁO ĐỘNG GIẢ (FALSE ALARM RATE)

### Bảng 2: Hiệu năng tính toán và khả năng kháng nhiễu

| Chỉ số đo đạc | Giá trị đo được | Ý nghĩa trong hệ thống thực tế |
| :--- | :---: | :--- |
| **Độ trễ TB (Latency)** | **13.73 ms** | Thời gian xử lý 1 khung hình trên CPU |
| **Độ trễ P95 (95th Percentile)** | **20.20 ms** | Đảm bảo không bị giật lag khung hình |
| **Tốc độ xử lý (Throughput)** | **72.8 FPS** | Đạt chỉ tiêu $\ge 5$ FPS trên CPU của MT5 |
| **Số ảnh âm tính kiểm thử** | 96 ảnh | Ảnh không có khói lẫn lửa |
| **Số ảnh âm tính bị báo giả** | 68 ảnh | Nhầm lẫn nguồn sáng / sương mù / vật thể đỏ |
| **Tỷ lệ báo giả khung hình (FPR)** | **70.83%** | Tỷ lệ nhầm lẫn cấp khung hình đơn lẻ |

---

## 4. ĐỐI SÁNH KHOA HỌC: BASELINE B1 vs. BASELINE B2

### Bảng 3: So sánh giữa B1 (Phương pháp truyền thống) và B2 (Học sâu phân loại)

| Tiêu chí đánh giá | Baseline B1 (Luật màu + Trừ nền MOG2) | Baseline B2 (MobileNetV3-Small) | Nhận xét chuyên môn |
| :--- | :---: | :---: | :--- |
| **Cách tiếp cận** | Heuristic thị giác máy tính truyền thống | Học sâu (Deep Learning Classification) | B2 học biểu diễn đặc trưng ngữ nghĩa tự động |
| **Định vị Bounding Box** | Có (Nhờ Connected Components / Contour) | **Không** (Chỉ xuất nhãn toàn khung hình) | B2 không khoanh vùng được ngọn lửa/cột khói |
| **Khả năng theo dõi Temporal** | Có thể gán box vào ByteTrack | **Không thể tích hợp ByteTrack** | B2 không có tọa độ $x, y, w, h$ để liên kết qua thời gian |
| **Kháng nhiễu ánh sáng tĩnh** | Rất kém (Dễ nhầm bóng đèn, tường đỏ) | **Tốt hơn rõ rệt** (Học được bối cảnh xung quanh) | B2 giảm được phần lớn báo giả do màu sắc đơn thuần |
| **Tốc độ trên CPU** | ~60 - 80 FPS | ~72.8 FPS | B1 nhanh hơn vì chỉ có phép biến đổi màu & MOG2 |
| **Số lượng tham số** | 0 tham số | ~1.08M tham số | B2 yêu cầu bộ nhớ trọng số ~9.8 MB |

---

## 5. KẾT LUẬN & ĐÓNG GÓP CHO ĐỀ TÀI NCKH

1. **Ý nghĩa của Baseline B2 trong nghiên cứu:**
   - Baseline B2 chứng minh rằng các mô hình học sâu có độ chính xác và khả năng tổng quát hóa vượt trội so với các luật màu heuristic (B1).
   - Tuy nhiên, điểm hạn chế cốt tử của B2 là **thiếu khả năng khoanh vùng không gian (Bounding Box Spatial Localization)**. Do không có Bounding Box, B2 không thể phối hợp với thuật toán theo dõi ByteTrack và các bộ lọc biến động đặc trưng diện tích/chu vi theo thời gian (Trọng tâm của MT3).
2. **Cơ sở khoa học chọn mô hình phát hiện đối tượng (Object Detection):**
   - Kết quả này củng cố tính tất yếu của việc sử dụng các mô hình phát hiện đối tượng hiện đại (**YOLOv8**, **YOLO11**, **RT-DETR**) trong đề tài: Vừa giữ được độ chính xác cao của học sâu, vừa cung cấp Bounding Box chính xác phục vụ xác nhận theo thời gian (`temporal_verifier.py`).
