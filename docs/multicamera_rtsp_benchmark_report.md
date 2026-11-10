# Báo Cáo Đo Kiểm Thực Nghiệm Camera RTSP Trực Tiếp (RTSP Live Benchmark)

**Dự án:** DACN_NCKH - Phát hiện và cảnh báo cháy từ camera giám sát  
**Ngày thực hiện:** 10/11/2026 11:03:20 UTC  
**Chế độ kiểm thử:** **B. RTSP Live Network Benchmark**  
**Số luồng RTSP khả dụng trên môi trường:** **0 luồng**  
**Trạng thái kiểm thử:** **`NOT_TESTED (BLOCKED_NO_LIVE_RTSP_SOURCE)`**  

---

## 1. Hiện Trạng Môi Trường RTSP Thực Tế

- Hệ thống đã kiểm tra tệp cấu hình `config/rtsp_cameras.json` và biến môi trường `RTSP_STREAMS`.
- **Kết quả:** Không có nguồn camera RTSP thực tế hoặc máy chủ MediaMTX RTSP đang chạy trong phiên thử nghiệm hiện tại.
- **Tuân thủ liêm chính khoa học:**
  - Hệ thống **tuyệt đối từ chối tự tạo số liệu giả lập FPS/độ trễ RTSP** hoặc đánh đồng kết quả tệp AVI với luồng RTSP.
  - Trạng thái các cấu hình 1, 2, 4, 6, 9 camera RTSP: **`NOT TESTED`**.

---

## 2. Quy Cách Sẵn Sàng Khi Có Nguồn RTSP Thực Tế

Hệ thống đã hoàn thiện 100% module xử lý RTSP trong [`src/camera_manager.py`](file:///d:/workspace/DACN_NCKH/src/camera_manager.py):
1. **Bảo mật:** Tự động che giấu mật khẩu đăng nhập RTSP (`rtsp://user:***@host:port/path`) trong toàn bộ nhật ký hệ thống bằng `mask_rtsp_credentials()`.
2. **Khả năng chịu lỗi:** Tự động phát hiện mất gói tin/mất tín hiệu, chuyển trạng thái sang `RECONNECTING`, thực hiện kết nối lại theo chu kỳ định sẵn (`reconnect_interval_sec`) và xóa bộ đệm theo dõi để tránh lệch ID.
3. **Cách kích hoạt:** Cung cấp danh sách RTSP URL vào `config/rtsp_cameras.json` hoặc chạy lệnh:
   ```powershell
   $env:RTSP_STREAMS="rtsp://192.168.1.100:554/ch1,rtsp://192.168.1.101:554/ch1"
   .\.venv\Scripts\python.exe scripts/benchmark_multicamera_scientific.py
   ```
