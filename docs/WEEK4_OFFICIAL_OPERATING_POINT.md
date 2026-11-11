# BÁO CÁO XÁC NHẬN OPERATING POINT WEEK 4 (GIAI ĐOẠN 5)
**Dự án:** Computer Vision-based Fire and Smoke Detection and Early Warning from Surveillance Camera Footage  
**Thời gian thực hiện:** 11/11/2026  
**File kết quả JSON:** [`best_operating_point.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/best_operating_point.json)  

---

## 1. Trạng Thái Điểm Vận Hành: `PROVISIONAL` (TẠM THỜI)
Căn cứ theo mục 5.3 của Master Prompt và nguyên tắc trung thực khoa học của Đề cương NCKH:
- Trạng thái xác nhận: **`PROVISIONAL`** (Tuyệt đối KHÔNG ghi `OFFICIAL`).
- **Lý do khoa học:** Điểm vận hành đã được tối ưu hóa toàn diện trên tập Validation dương tính và kiểm chứng độc lập trên tập Test dương tính, nhưng chỉ số **Tỷ lệ Báo Động Giả (FAR/hour) trên tập dữ liệu âm tính thực tế** chưa được đo kiểm đầy đủ tại thời điểm chạy benchmark tự động. Theo quy chuẩn Đề cương (MT3: giảm $\ge 50\%$ FAR/giờ), điểm vận hành chỉ được chuyển sang trạng thái `OFFICIAL` khi có số liệu đo lường FAR cụ thể trên tập video âm tính $\ge 5,0$ giờ.

---

## 2. Thông Số Cấu Hình Điểm Vận Hành Đang Áp Dụng

### Cấu hình 1: Cấu hình Khuyến nghị Triển khai (Pipeline B2)
Áp dụng trên hệ thống giám sát và đối sánh toàn bộ 24 video dương tính:
- **Ngưỡng tin cậy (Confidence Threshold):** `0.35`
- **Cửa sổ trượt thời gian ($N$):** `8 frame`
- **Số frame xác nhận ($K$):** `3 frame` ($k/N = 3/8$)
- **Cổng biến động đặc trưng (Feature Threshold):** `0.02` (yêu cầu độ chớp cường độ hoặc biến thiên diện tích $\ge 0.02$)
- **Thời gian Cooldown:** `5.0 s` (áp dụng độc lập cho từng lớp FIRE và SMOKE)
- **Hiệu năng thực tế trên 24 video dương tính:**
  - Giảm cảnh báo spam: **-95,64%** (từ 2.938 frame alarms xuống 128 sự kiện DB)
  - Video-level Recall: **95,83%** (23 / 24 video, 0% suy giảm so với baseline)
  - Event Precision: **80,65%** (+27,82 điểm % so với 52,83% của Baseline)
  - Event Recall: **71,43%**
  - Event F1-Score: **75,76%** (+12,12 điểm % so với 63,64% của Baseline)
  - Độ trễ phát hiện trung bình: **2,13 giây** (đạt chỉ tiêu đề cương $\le 10\text{ s}$)
  - Báo động trước khi cháy nổ: **0** (giảm từ 3 ở baseline)

### Cấu hình 2: Cấu hình Tối ưu Quét Lưới (Grid Search Locked)
Tối ưu trên 12 video Validation và kiểm định trên 12 video Test độc lập:
- **Confidence Threshold:** `0.25`
- **Cửa sổ trượt ($k/N$):** `5 / 12`
- **Feature Threshold:** `0.0`
- **Cooldown:** `0.0 s`
- **Kết quả trên tập Test độc lập (12 video):**
  - Video Recall: **100,0%** (12 / 12 video)
  - Event Precision: **72,22%**
  - Event Recall: **72,22%**
  - Event F1-Score: **72,22%**
  - Độ trễ trung bình: **3,21 giây**

---

## 3. Tính Toàn Vẹn Thực Nghiệm & Tái Lập
- **Git Commit:** `256065e7`
- **Model Checkpoint:** `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt`
- **Model Checkpoint SHA-256:** `4d7e9b08f886f3b46944e05b5fbaec6866fbe537aa2db2dc9d3118cf6e9e42e1`
- **Phần cứng thực thi:** CPU Intel Core i7 / GPU NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.x, FP16)
- **Môi trường:** Python 3.10.11, PyTorch 2.x, Torchvision, Ultralytics 8.3.x, OpenCV 4.10.0

