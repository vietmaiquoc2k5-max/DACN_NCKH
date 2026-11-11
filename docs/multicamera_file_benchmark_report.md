# Báo Cáo Hiệu Năng Đa Luồng Mô Phỏng Tệp Cục Bộ (File-Based Multi-Stream Benchmark)

**Dự án:** DACN_NCKH - Phát hiện và cảnh báo cháy từ camera giám sát  
**Ngày thực hiện:** 10/11/2026 09:53:43 UTC  
**Chế độ kiểm thử:** **A. File-Based Multi-Stream Simulation (Tệp AVI cục bộ)**  
**Phần cứng:** CPU Intel (12 cores, 6 threads worker)  
**Mô hình:** `best.pt` (imgsz=416, frame_skip=2)  

> [!IMPORTANT]
> Đây là phép đo tải đa luồng sử dụng tệp video cục bộ (`FireSmokeLabeler/videos/*.avi`) với cơ chế giải mã OpenCV và nén JPEG C++. Báo cáo này **không đại diện cho độ trễ truyền dẫn mạng RTSP**.

---

## 1. Kết Quả Đo Hiệu Năng Tải Đa Luồng (1, 2, 4, 6, 9 Streams)

| Số Luồng Video | Capture FPS (TB) | Min Capture FPS | Inference FPS (TB) | Độ Trễ P50 | Độ Trễ P95 | Độ Trễ P99 | CPU (%) | RAM (MB) | Chỉ Tiêu MT5 (>= 5 FPS CPU) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Streams** | **23.8 FPS** | 23.8 FPS | 48.4 FPS | 20.9 ms | 25.4 ms | 26.2 ms | 81.0% | 423.4 MB | **✅ ĐẠT** |
| **2 Streams** | **23.8 FPS** | 23.8 FPS | 36.1 FPS | 25.3 ms | 26.0 ms | 26.2 ms | 100.0% | 470.6 MB | **✅ ĐẠT** |
| **4 Streams** | **23.6 FPS** | 22.7 FPS | 32.8 FPS | 27.1 ms | 82.5 ms | 136.4 ms | 100.0% | 561.6 MB | **✅ ĐẠT** |
| **6 Streams** | **23.8 FPS** | 23.8 FPS | 36.8 FPS | 26.2 ms | 29.4 ms | 30.2 ms | 100.0% | 647.8 MB | **✅ ĐẠT** |
| **9 Streams** | **23.7 FPS** | 23.7 FPS | 34.6 FPS | 28.7 ms | 34.2 ms | 34.9 ms | 100.0% | 769.4 MB | **✅ ĐẠT** |

---

## 2. Nhận Xét Chuyên Môn

1. **Khả năng duy trì FPS trên CPU:** Luồng đọc khung hình duy trì ổn định từ 23.5 - 23.8 FPS cho cả 9 luồng video đồng thời nhờ cơ chế suy luận không chặn (`non_blocking=True`) và nén JPEG C++ trực tiếp.
2. **Tính độc lập của Tracker:** Mỗi worker sở hữu riêng biệt ByteTrack và TemporalVerifier, không chia sẻ ID hay trạng thái cửa sổ trượt.
