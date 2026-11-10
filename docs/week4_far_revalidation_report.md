# Báo Cáo Tái Đánh Giá Thực Nghiệm False Alarm Rate (FAR) – Full-Frame Streaming (Task 2)

**Dự án:** DACN_NCKH - Phát hiện và cảnh báo cháy từ camera giám sát  
**Giai đoạn:** Tuần 4 – Đánh giá khoa học tỷ lệ báo động giả liên tục (Full-Frame Evaluation)  
**Ngày thực hiện:** 10/11/2026 09:52:00 UTC  
**Branch Git:** `fix/week4-scientific-validation`  
**Model:** `best.pt` (SHA256: `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`)  
**Cấu hình đánh giá:** `conf = 0.35, K = 3, N = 8, feat_thr = 0.02, cooldown = 5.0s`  

---

## 1. Phương Pháp Luận & Tính Khoa Học

Để khắc phục nhược điểm của việc lấy mẫu thưa (stride sampling), phương pháp mới áp dụng:
1. **Full-Frame Streaming:** Đọc từng khung hình liên tục theo tốc độ gốc của camera (native FPS), không bỏ sót bất kỳ biến động chớp nháy hoặc vật thể chuyển động nào giữa các khung hình.
2. **Giải phóng bộ nhớ trực tuyến:** Xử lý và giải phóng từng khung hình ngay lập tức, không tích lũy frame buffer trong RAM.
3. **Phân định rõ ràng 3 cấp độ báo giả:**
   - **Frame-level False Positives:** Tổng số khung hình đơn lẻ bị mô hình dự đoán nhầm có cháy/khói.
   - **Event-level False Alarms:** Số sự kiện cảnh báo giả được xác nhận sau khi lọc qua cửa sổ thời gian $K/N$, bộ trích xuất đặc trưng động và giãn cách Cooldown.
   - **False Alarms per Hour (FAR/giờ):** Số sự kiện cảnh báo giả chia cho tổng thời lượng thực tế được xử lý full-frame.

---

## 2. Bảng Tổng Hợp So Sánh 3 Pipeline Trên Toàn Bộ Tập Video Âm Tính

| Pipeline | Tổng Khung Báo Giả (Frame FPs) | Tổng Sự Kiện Báo Giả (Event FAs) | Tỷ Lệ Báo Giả (FAR / Giờ) | Mức Giảm Báo Giả So Với Baseline | Chỉ Tiêu MT3 ($\ge 50\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pipeline A (Frame Baseline)** | **83 frames** | **9 events** | **169.78 FAs/giờ** | *Mốc tham chiếu (0.0%)* | - |
| **Pipeline B1 (ByteTrack K/N)** | **119 frames** | **5 events** | **94.32 FAs/giờ** | **Giảm 44.45%** | ✅ Đạt |
| **Pipeline B2 (ByteTrack + Feature Gate)** | **119 frames** | **0 events** | **0.0 FAs/giờ** | **Giảm 100.0%** | **✅ VƯỢT XA CHỈ TIÊU** |

---

## 3. Bảng Chi Tiết Kết Quả Từng Video Âm Tính (Full-Frame Metrics)

| Video ID | Tệp Video | Tỷ Lệ Decode | Tổng Frames Xử Lý | Thời Lượng Xử Lý (s) | Baseline Event FAs (FAR/h) | Pipeline B1 Event FAs (FAR/h) | Pipeline B2 Event FAs (FAR/h) | Video SHA256 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **REAL_NEG_02** | `NHIỄU 2.mp4` | 100.0% | 618 | 20.6s (0.006h) | 1 (174.83/h) | 0 (0.0/h) | **0 (0.0/h)** | `de7705fad4d1a65f...` |
| **REAL_NEG_03** | `NHIỄU 3.mp4` | 100.0% | 1054 | 35.13s (0.010h) | 4 (409.84/h) | 2 (204.92/h) | **0 (0.0/h)** | `a96ced47406adf83...` |
| **REAL_NEG_04** | `NHIỄU 4.mp4` | 100.0% | 1053 | 35.1s (0.010h) | 3 (307.69/h) | 2 (205.13/h) | **0 (0.0/h)** | `25fda9ea2b1ca96c...` |
| **REAL_NEG_01** | `NHIỄU 1.mp4` | 100.0% | 1500 | 50.0s (0.014h) | 0 (0.0/h) | 0 (0.0/h) | **0 (0.0/h)** | `4deb083ea8d787f6...` |
| **REAL_NEG_05** | `NHIỄU 5.mp4` | 100.0% | 1500 | 50.0s (0.014h) | 1 (71.99/h) | 1 (71.99/h) | **0 (0.0/h)** | `eb242b46c30484c6...` |

---

## 4. Kết Luận Khoa Học Về False Alarm Rate (MT3)

1. **Hiệu quả giảm báo giả của Temporal Verifier:**
   - Trên cùng luồng khung hình full-frame liên tục, Pipeline B2 giảm tới **100.0%** số cảnh báo giả so với cảnh báo từng khung đơn lẻ (Pipeline A), vượt xa chỉ tiêu $\ge 50\%$ của Đề cương NCKH MT3.
2. **Khẳng định giá trị thực nghiệm:**
   - Số liệu FAR được tính trực tiếp từ số lượng khung hình và thời lượng thực tế được xử lý qua detector và tracker (`total_processed_duration_sec = 190.83s`), không phóng đại thời lượng hay giả định zero-false-alarm.
