# BÁO CÁO KIỂM TOÁN DỮ LIỆU THỰC TẾ WEEK 4 (GIAI ĐOẠN 1)
**Dự án:** Computer Vision-based Fire and Smoke Detection and Early Warning from Surveillance Camera Footage  
**Thời gian thực hiện:** 11/11/2026  
**Công cụ kiểm toán:** `scripts/run_week4_audit.py` (sử dụng OpenCV, SHA-256 chunking, FFmpeg codec inspection)  
**File kết quả chi tiết:**  
- JSON: [`data_audit.json`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/data_audit.json)  
- CSV: [`data_audit.csv`](file:///d:/DACN_NCKH/runs/benchmarks/week4_final/data_audit.csv)  

---

## 1. Tổng quan Kiểm toán Kho Dữ liệu Video

| Chỉ tiêu kiểm toán | Giá trị ghi nhận thực tế | Yêu cầu Đề cương (MT1) | Đánh giá |
| :--- | :---: | :---: | :---: |
| **Tổng số video trong danh mục (`video_dataset_list.csv`)** | **192 video** | - | Đã lập chỉ mục đầy đủ |
| **Số video thực tế tồn tại trên ổ đĩa và giải mã thành công (VALID)** | **183 video** | - | **ĐẠT (95.3%)** |
| **Số video bị thiếu (MISSING)** | **9 video** (`NEG_41..49`) | 0 | Không ảnh hưởng do đã có 79 NEG thay thế |
| **Số video bị lỗi giải mã (CORRUPTED)** | **0 video** | 0 | **HOÀN TOÀN HỢP LỆ** |
| **Tổng số video Dương tính hợp lệ (Positive)** | **104 video** | $\ge 20$ video | **VƯỢT CHỈ TIÊU (104 / 20)** |
| **Tổng thời lượng video Dương tính hợp lệ** | **18.627,1 giây (5,17 giờ)** | - | Đủ cho cả Val, Test và mở rộng |
| **Tổng số video Âm tính hợp lệ (Negative)** | **79 video** | - | Đủ điều kiện exposure |
| **Tổng thời lượng video Âm tính hợp lệ** | **25.929,5 giây (7,20 giờ)** | $\ge 5,0$ giờ | **VƯỢT CHỈ TIÊU (7,20h / 5,0h)** |

---

## 2. Kiểm toán Chi tiết Tập Video Dương Tính (Positive Videos)
Tập 104 video dương tính được phân cấp rõ ràng theo mức độ xác minh:
1. **Nhóm chuẩn hóa Tuần 4 (`POS_01` – `POS_24`):**
   - Số lượng: 24 video gốc chuẩn.
   - Thời lượng: 863,98 giây (0,24 giờ).
   - Kiểm tra giải mã OpenCV: 100% video mở thành công, trích xuất khung hình mượt mà tại 5.0 FPS.
   - Ground Truth: Đã xác minh từng giây xuất hiện ngọn lửa (`fire_start_sec`) và khói (`smoke_start_sec`).
2. **Nhóm giám sát bối cảnh Việt Nam (`VN_POS_01` – `VN_POS_20`):**
   - Số lượng: 20 video ghi nhận từ camera an ninh, nhà xưởng, đường phố Việt Nam.
   - Thời lượng: 7.422,3 giây (2,06 giờ).
   - Đã được làm mờ khuôn mặt và biển số xe bảo vệ quyền riêng tư.
3. **Nhóm mở rộng thực tế mới (`VN_POS_21` – `VN_POS_80`):**
   - Số lượng: 60 video bổ sung từ đợt khảo sát mới.
   - Thời lượng: 10.340,8 giây (2,87 giờ).
   - Đã giải mã kiểm tra tính hợp lệ bằng OpenCV, sẵn sàng cho công tác gán nhãn chéo độc lập.

---

## 3. Kiểm toán Chi tiết Tập Video Âm Tính & Cảnh Gây Nhiễu (Negative Videos)
Đề cương yêu cầu tối thiểu 5,0 giờ video âm tính chứa các bối cảnh dễ gây báo động giả:
- **Thời lượng thực tế đạt:** **7,20 giờ (25.929,5 giây)**, vượt 44% so với chỉ tiêu tối thiểu 5,0 giờ.
- **Phân bố các cảnh gây nhiễu đặc thù:**
  1. *Ánh sáng vàng/đỏ, đèn xe, đèn pha giao thông:* `VN_NEG_01`, `VN_NEG_06`, `VN_NEG_14`, `VN_NEG_21`, `VN_NEG_31` (giao thông ban đêm, đèn hậu xe máy, ô tô).
  2. *Biển hiệu LED, neon, đèn quảng cáo nhấp nháy:* `VN_NEG_10`, `VN_NEG_15`, `VN_NEG_20`, `VN_NEG_40` (quán ăn, mặt tiền phố đêm).
  3. *Bụi đường, hơi nước, sương mù, khói sinh hoạt không cháy:* `VN_NEG_04`, `VN_NEG_08`, `VN_NEG_18`, `VN_NEG_25`, `VN_NEG_36`.
  4. *Cảnh bếp ăn, nấu nướng sinh hoạt:* `VN_NEG_09`, `VN_NEG_27`, `VN_NEG_44`.
  5. *Ánh hoàng hôn, phản chiếu mặt trời trên mái tôn/cửa kính:* `VN_NEG_11`, `VN_NEG_22`, `VN_NEG_50`, `VN_NEG_62`.
- **9 video âm tính thiếu (`NEG_41..49`):** Đây là các file tổng hợp giả lập cũ (`testneg01..09.avi`, tổng 0,08h). Do đã có 79 video thực tế chất lượng cao (7,20h), việc thiếu 9 file này không ảnh hưởng đến độ tin cậy của tập dữ liệu âm tính.

---

## 4. Kiểm toán Tập Dữ liệu Hình ảnh (Image Dataset Audit)

### 4.1. Tập Dữ liệu Việt Nam (`datasets/vn_dataset/`)
- **Tổng số ảnh:** **3.121 ảnh** (định dạng `.jpg` / `.png`).
- **Tổng số nhãn:** **3.121 file nhãn** YOLO `.txt` tương ứng 1-1 (100% khớp, không có file mồ côi).
- **Phân chia tập (Dataset Splitting):**
  - `train`: **1.868 ảnh** (59,9%)
  - `val`: **623 ảnh** (20,0%)
  - `test`: **630 ảnh** (20,2%)
- **Kiểm định Chống Rò rỉ Dữ liệu (Data Leakage Verification):**
  - Nguyên tắc phân chia: Phân tách tuyệt đối 100% theo ID video (các khung hình trích từ cùng một video chỉ xuất hiện trong duy nhất một tập).
  - Giao tập Train $\cap$ Val: $\emptyset$ (0 video trùng lặp).
  - Giao tập Train $\cap$ Test: $\emptyset$ (0 video trùng lặp).
  - Giao tập Val $\cap$ Test: $\emptyset$ (0 video trùng lặp).
  - **Kết luận:** **`is_leakage_free: True`** (Đạt chuẩn khoa học cao nhất).

### 4.2. Tập Huấn Luyện D-Fire (`datasets/dfire/`)
- **Tổng số ảnh:** **21.527 ảnh**.
- **Tổng số nhãn:** **21.527 nhãn** định dạng YOLO (0: smoke, 1: fire).
- **Phân bổ:**
  - `train`: 17.221 ảnh (80,0%)
  - `test`: 4.306 ảnh (20,0%, trong đó chia phân tầng độc lập thành Val 2.153 và Test 2.153).

