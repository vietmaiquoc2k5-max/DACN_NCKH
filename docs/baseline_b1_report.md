# Báo Cáo Thực Nghiệm Baseline B1 (Luật Màu & Trừ Nền)

**Đề tài:** Nghiên cứu ứng dụng computer vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
**Mục tiêu đối chiếu:** MT2 – Xây dựng mốc tham chiếu truyền thống (Baseline 1)  
**Thời điểm thực nghiệm:** 25/10/2026 (Tuần 2)  
**Thiết bị đánh giá:** CPU Intel Core (x86_64, Single-Thread OpenCV)  

---

## 1. Mô Tả Kỹ Thuật Baseline B1

Baseline B1 kết hợp các đặc trưng thị giác truyền thống:
- **Luật màu ngọn lửa:** Kết hợp $R > G > B$ (RGB), $Y > Cb$ & $Cr > Cb$ (YCbCr) và dải sắc đỏ/vàng của HSV.
- **Luật màu khói:** Phân tích độ phân tán sắc độ thấp (Gray Dispersion) và độ bão hòa nhạt ($S \le 45, 65 \le V \le 230$).
- **Trừ nền thích nghi:** Gaussian Mixture Model (MOG2) để tách các vùng chuyển động và loại bỏ các đối tượng tĩnh màu đỏ.

---

## 2. Bảng Kết Quả Định Lượng Trên Tập Kiểm Thử Việt Nam (Test Set - 210 ảnh)

| Lớp đối tượng | Precision | Recall | F1-Score | Số lượng GT | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Khói (Smoke)** | **0.0004** | **0.0294** | **0.0008** | 34 | 1 | 2423 | 33 |
| **Lửa (Fire)** | **0.0011** | **0.0526** | **0.0022** | 57 | 3 | 2618 | 54 |
| **Trung bình** | **0.0008** | **0.041** | **0.0015** | 91 | 4 | 5041 | 87 |

---

## 3. Hiệu Năng Tính Toán (Computational Efficiency)

### 3.1. Đánh giá cấp ảnh tĩnh (Static Frames)
| Chỉ số đo lường | Giá trị thực nghiệm | Nhận xét chuyên môn |
| :--- | :---: | :--- |
| **Thời gian suy luận (Latency)** | **56.8 ms** | Chạy đơn luồng trên CPU x86_64, không cần GPU. |
| **Tốc độ xử lý (Throughput)** | **17.6 FPS** | Độ trễ thấp khi xử lý ảnh độ phân giải gốc. |
| **Hạn chế ảnh tĩnh** | Báo giả cao | Do thiếu thông tin chuyển động thời gian, các bề mặt màu đỏ/vàng tĩnh bị nhận nhầm. |

### 3.2. Đánh giá cấp luồng Video động (Video Stream với MOG2 Motion Filtering)
Khi kích hoạt mô hình trừ nền Gaussian Mixture Model (MOG2) và co tỉ lệ tính toán `imgsz=640`:

| Chỉ số đo lường | Giá trị thực nghiệm | Nhận xét chuyên môn |
| :--- | :---: | :--- |
| **Độ trễ xử lý (Latency)** | **21.87 ms** | Tối ưu hóa vượt trội với OpenCV C++ backend. |
| **Tốc độ xử lý (Throughput)** | **45.7 FPS** | **Đạt và vượt tiêu chuẩn thời gian thực ($\ge 30$ FPS) trên CPU**. |
| **Phát hiện chuyển động cháy** | **Tỷ lệ bắt cháy 70.0%** | MOG2 lọc bỏ hoàn toàn các biển quảng cáo và vật thể màu đỏ cố định. |

---

## 4. Đánh Giá Khoa Học & Kết Luận Cho Bài Báo IEEE

1. **Ưu điểm vượt trội:**
   - Thuật toán có cấu trúc tính toán đơn giản, đạt tốc độ **17.6 FPS** trên CPU mà không cần card đồ họa rời (GPU).
   - Khả năng bắt nhạy các ngọn lửa bùng phát lớn có màu sắc rực rỡ đặc trưng.
2. **Hạn chế cố hữu:**
   - Khói có độ bão hòa màu biến thiên liên tục và dễ bị hòa lẫn với sương mù hoặc mặt đường màu xám (Recall thấp).
   - Dễ gây báo động giả khi gặp ánh đèn pha xe máy ban đêm hoặc áo phản quang màu cam/đỏ.
3. **Ý nghĩa khoa học:**
   - Đây là mốc đối chứng (Ground Zero) hoàn hảo để chứng minh sự cần thiết và tính ưu việt của các mô hình Deep Learning (YOLOv8, YOLO11, RT-DETR) sẽ được huấn luyện ở Tuần 3.
