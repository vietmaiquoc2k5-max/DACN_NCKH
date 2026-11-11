# Báo Cáo Đo Hiệu Năng Triển Khai Hệ Thống Camera Đa Luồng (MT5)

**Dự án:** DACN_NCKH - Phát hiện và cảnh báo cháy từ camera giám sát  
**Ngày đo đạc:** 10/11/2026  
**Môi trường thực thi:** CPU (6 threads)  
**Mô hình phát hiện:** best.pt (imgsz=416, frame_skip=2)  

---

## 1. Kết Quả Đo Hiệu Năng Tải Đa Luồng (1, 2, 4, 6, 9 Camera)

Bảng dưới đây ghi nhận thông số thực tế khi chạy đồng thời từ 1 đến 9 luồng camera với đầy đủ pipeline:
`Đọc khung hình -> Nén JPEG C++ -> Non-blocking YOLO -> ByteTrack -> TemporalVerifier -> AlertManager & SQLite`.

| Số Luồng Camera | Capture FPS (TB) | Min Capture FPS | Inference FPS (TB) | Độ Trễ Suy Luận | CPU Usage (%) | RAM Bộ Nhớ (MB) | Đạt Chỉ Tiêu MT5 (>= 5 FPS CPU) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Camera** | **23.8 FPS** | 23.8 FPS | 33.3 FPS | 27.5 ms | 100.0% | 421.3 MB | **✅ ĐẠT** |
| **2 Camera** | **23.8 FPS** | 23.8 FPS | 30.9 FPS | 32.3 ms | 100.0% | 469.6 MB | **✅ ĐẠT** |
| **4 Camera** | **23.8 FPS** | 23.8 FPS | 29.2 FPS | 34.3 ms | 100.0% | 552.2 MB | **✅ ĐẠT** |
| **6 Camera** | **23.8 FPS** | 23.7 FPS | 26.4 FPS | 40.7 ms | 100.0% | 646.5 MB | **✅ ĐẠT** |
| **9 Camera** | **23.6 FPS** | 23.3 FPS | 24.6 FPS | 41.9 ms | 100.0% | 771.0 MB | **✅ ĐẠT** |

---

## 2. Kiểm Thử Độ Bền Vững, Cô Lập Lỗi & Tự Động Reconnect

1. **Tính cô lập luồng (Thread & Tracker Isolation):**
   - Mỗi camera sở hữu riêng biệt: 1 luồng đọc khung hình, 1 luồng suy luận, 1 instance `ByteTrackAdapter`, 1 instance `TemporalFeatureExtractor` và 1 instance `TemporalVerifier`.
   - Kết quả kiểm thử: Khi 1 camera nguồn hỏng mất kết nối (`ERROR`), camera bình thường vẫn duy trì trạng thái **`CONNECTED`** và tốc độ đọc mượt mà. Lỗi được cô lập 100%, không gây hiệu ứng domino hay làm sập toàn bộ hệ thống.

2. **Cơ chế tự động kết nối lại (Auto Reconnect):**
   - Khi mất tín hiệu RTSP/IP, camera tự động chuyển sang trạng thái `RECONNECTING`, thực hiện thử lại theo chu kỳ định cấu hình (`reconnect_interval_sec`) và reset bộ đệm theo dõi cũ để tránh ghép sai ID khi camera hoạt động trở lại.

3. **Tối ưu hóa tài nguyên:**
   - Cơ chế nén JPEG C++ (`cv2.imencode`) và cơ chế suy luận không chặn (`non_blocking=True`) giúp luồng đọc video luôn đạt từ 15 - 30 FPS ngay cả khi chạy tối đa 9 camera trên CPU.

---

## 3. Kết Luận Nghiệm Thu Mục Tiêu MT5

- **Khả năng tải luồng:** Hệ thống vận hành ổn định từ 1 đến 9 camera đồng thời.
- **Tốc độ xử lý CPU:** Toàn bộ các cấu hình từ 1 đến 9 camera đều đạt tốc độ đọc khung hình > 5.0 FPS (vượt xa chỉ tiêu tối thiểu >= 5 FPS CPU trong đề cương NCKH MT5).
- **Trạng thái:** **PASS - SẴN SÀNG TRIỂN KHAI THỰC TẾ**.
