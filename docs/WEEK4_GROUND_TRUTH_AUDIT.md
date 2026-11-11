# BÁO CÁO KIỂM TOÁN GROUND TRUTH WEEK 4 (GIAI ĐOẠN 3)
**Dự án:** Computer Vision-based Fire and Smoke Detection and Early Warning from Surveillance Camera Footage  
**Thời gian thực hiện:** 11/11/2026  
**File Ground Truth Master:** [`ground_truth_master.csv`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/ground_truth/ground_truth_master.csv)  
**File Ground Truth Validation:** [`ground_truth_validation.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/ground_truth/ground_truth_validation.json)  

---

## 1. Nguyên Tắc Chuẩn Hóa và Phân Loại Ground Truth
Để đảm bảo tính nghiêm cẩn khoa học, Ground Truth được phân cấp thành các mức xác thực rõ ràng:
- **`VERIFIED` (Đã kiểm tra và xác nhận):**
  - Gồm 24 video dương tính chuẩn (`POS_01` – `POS_24`) và 20 video dương tính Việt Nam (`VN_POS_01` – `VN_POS_20`).
  - Toàn bộ 79 video âm tính thực tế (`NEG_01..40`, `VN_NEG_01..65`).
  - Đã được con người kiểm tra trực tiếp qua video player, đối chiếu khung hình và ghi nhận chính xác mốc giây bắt đầu bốc khói (`smoke_start_sec`) và bùng lửa (`fire_start_sec`).
- **`EXPLORATORY` (Dữ liệu thăm dò / Gán nhãn sơ bộ):**
  - Gồm 60 video mới (`VN_POS_21` – `VN_POS_80`).
  - Đã được trích xuất khung hình và sinh nhãn sơ bộ bằng mô hình kết hợp kiểm duyệt tự động; cần thêm quy trình thẩm định chéo độc lập trước khi đưa vào các báo cáo công bố quốc tế.
  - Tuyệt đối không dùng dữ liệu thăm dò làm Ground Truth cho bảng đối sánh điểm vận hành chính thức của Tuần 4.

---

## 2. Thống Kê Sự Kiện Ground Truth Trong Tập Đánh Giá Tuần 4
Trên tập 24 video kiểm thử dương tính chuẩn hóa (`POS_01` – `POS_24`):
- **Tổng số khoảng sự kiện Ground Truth:** **35 sự kiện**
  - **FIRE (Ngọn lửa):** **16 sự kiện**
  - **SMOKE (Khói):** **19 sự kiện**
- Các video có cả hai hiện tượng lửa và khói (ví dụ `POS_01`, `POS_08`, `POS_14`, `POS_15`) được tách thành 2 sự kiện Ground Truth độc lập với mốc thời gian bắt đầu riêng biệt.

---

## 3. Thuật Toán và Quy Tắc Khớp Sự Kiện (Event Matching Algorithm)
Triển khai tại `src/event_eval.py` (`evaluate_events_strict`):
1. **Phân tách theo lớp (Strict Class Matching):**
   - Dự đoán `FIRE` chỉ được khớp với Ground Truth `FIRE`.
   - Dự đoán `SMOKE` chỉ được khớp với Ground Truth `SMOKE`.
2. **Khớp 1-1 (One-to-One Matching):**
   - Mỗi khoảng Ground Truth chỉ được ghép với tối đa một sự kiện dự đoán (và ngược lại) để ngăn chặn việc đếm lặp làm tăng số lượng True Positives giả tạo.
3. **Dung sai thời gian (Temporal Tolerance):**
   - Dung sai cho phép: $\Delta t \le 10,0\text{ s}$ kể từ mốc bắt đầu Ground Truth (`start_sec - 2.0s \le t_{pred} \le end_sec + 2.0s`).
4. **Định nghĩa Độ trễ phát hiện (Detection Delay):**
   $$\text{Delay} = \max(0.0, t_{\text{confirmed}} - t_{\text{GT\_start}})$$
   Trong đó $t_{\text{confirmed}}$ là thời điểm hệ thống (Temporal Verifier) thực sự phát ra cảnh báo đến cơ sở dữ liệu/người dùng.

