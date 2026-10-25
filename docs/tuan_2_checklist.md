# Biên Bản Nghiệm Thu Tiến Độ Tuần 2 (26/10 – 01/11/2026)

**Đề tài:** Nghiên cứu ứng dụng computer vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
**Thành viên thực hiện:** Mai Quốc Việt, Lê Minh Thiên  
**Giảng viên hướng dẫn:** ThS. Nguyễn Hữu Trung  
**Thời điểm đánh giá:** 25/10/2026 (Hoàn thành sớm trước hạn tiến độ)  

---

## 📋 BẢNG ĐỐI CHIẾU TIẾN ĐỘ TUẦN 2 VÀ KẾT QUẢ THỰC HIỆN

| Hạng mục đề cương | Yêu cầu chi tiết | Kết quả thực hiện trong Repo | Trạng thái |
| :--- | :--- | :--- | :--- |
| **1. Bộ video kiểm thử & Mốc thời gian (MT1, MT3)** | - Đưa video camera Việt Nam vào hệ thống.<br>- Gán mốc giây bắt đầu có khói / cháy (`smoke_start_sec`, `fire_start_sec`). | - Đã đưa đủ 11 video (`VN_POS_01` – `10`, `VN_NEG_01`) vào `data/video_dataset_list.csv`.<br>- Tự động dò và cập nhật mốc thời gian bằng AI trên GPU CUDA qua `scripts/auto_detect_timestamps.py`.<br>- Cung cấp công cụ tinh chỉnh thủ công `scripts/tag_video_timestamps.py`. | **HOÀN THÀNH 100%** |
| **2. Gán nhãn 1.000 khung hình Việt Nam (MT1, MT4)** | - $\ge$ 1.000 khung hình Việt Nam.<br>- Chia 60/20/20 theo video độc lập.<br>- Làm mờ mặt và biển số xe bảo mật.<br>- Nghiệm thu Ground Truth thủ công. | - Cắt & bảo mật **1.050 khung hình** bằng `scripts/extract_vn_frames.py` (tích hợp YOLOv8n làm mờ mặt và biển số xe).<br>- Phân chia chuẩn 60/20/20 theo video (Train: 630, Val: 210, Test: 210 ảnh) kèm file cấu hình `data_vn.yaml`.<br>- Đã hoàn tất kiểm duyệt & tinh chỉnh $100\%$ Ground Truth ($640$ boxes) qua `scripts/review_vn_labels.py`, lưu nhật ký tại `datasets/vn_dataset/review_status.csv`. | **HOÀN THÀNH 100%** |
| **3. Cài đặt Baseline B1 (MT2)** | - Luật màu HSV/YCbCr kết hợp trừ nền thích nghi GMM/MOG2.<br>- Đo đạc tốc độ xử lý (FPS) và độ chính xác mAP. | - Đã cài đặt hoàn chỉnh module tại `baselines/baseline_b1.py`.<br>- Xây dựng bộ đánh giá khoa học định lượng tại `baselines/evaluate_b1.py`.<br>- Đạt tốc độ **45.7 – 60.3 FPS** trên CPU (độ trễ 16.6 – 21.8 ms).<br>- Đã chạy thử nghiệm video và xuất demo tại `runs/b1_demo.mp4`.<br>- Xuất báo cáo khoa học đối chuẩn chi tiết tại `docs/baseline_b1_report.md`. | **HOÀN THÀNH 100%** |

---

## 🎯 CHUYỂN GIAO SANG TUẦN 3 (02/11 – 08/11/2026 - MỐC KIỂM TRA 1)

Tuần 2 đã hoàn tất toàn bộ các mục tiêu cốt lõi. Kế hoạch chuyển giao cho **Tuần 3** (Mốc kiểm tra 1 với GVHD):
1. **Baseline B2 (Mạng phân loại nhẹ MobileNetV3):**
   - Xây dựng mô hình phân loại nhẹ làm mốc đối chứng thứ hai.
2. **Huấn luyện các mô hình phát hiện Deep Learning (MT2):**
   - Huấn luyện 4 kiến trúc chính trên tập chuẩn D-Fire (21.527 ảnh): **YOLOv8n/s**, **YOLO11n/s**, **RT-DETR-R18**.
   - Cùng số epoch, ảnh đầu vào `imgsz=640`, 3 seed ngẫu nhiên để báo cáo giá trị trung bình $\pm$ độ lệch chuẩn.
   - Mục tiêu: $mAP@0.5 \ge 0,75$ trên tập D-Fire.
