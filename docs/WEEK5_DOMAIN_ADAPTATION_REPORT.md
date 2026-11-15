# Báo Cáo Nghiệm Thu Khoa Học Tuần 5 – Thích Nghi Miền (Domain Adaptation)

**Đề tài:** Nghiên cứu ứng dụng computer vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
**Tác giả phụ trách:** Lê Minh Thiên (Chủ nhiệm đề tài – Mô hình phát hiện & Thích nghi miền), Mai Quốc Việt (Kỹ sư hệ thống & Dữ liệu)  
**Thời gian thực hiện:** Tuần 5 (16/11/2026 – 22/11/2026)  
**Mục tiêu đề cương:** **MT4 – Thích nghi miền (Domain Adaptation)**

---

## 1. Mục Tiêu & Chỉ Tiêu Đề Cương (MT4)

| Chỉ tiêu nghiên cứu | Mục tiêu đặt ra trong đề cương | Tiêu chuẩn nghiệm thu |
| :--- | :--- | :---: |
| **Chiến lược khảo sát** | Khảo sát đủ 3 chiến lược: Toàn bộ / Đóng băng backbone / Đóng băng + Replay 20% D-Fire | Đạt yêu cầu |
| **Cải thiện độ chính xác tại Việt Nam** | $\Delta\text{mAP@0.5}_{\text{VN}} \ge +5.0\%$ (+5 điểm %) so với Baseline `best.pt` | Bắt buộc |
| **Hạn chế suy giảm trên miền gốc D-Fire** | $\Delta\text{mAP@0.5}_{\text{D-Fire}} \ge -2.0\%$ (mức giảm $\le 2$ điểm %) chống quên thảm khốc | Bắt buộc |
| **Tập dữ liệu kiểm định** | Đánh giá độc lập trên tập Test Việt Nam (630 ảnh) và tập Test D-Fire (2.153 ảnh) | Chuẩn hóa |

> [!IMPORTANT]
> **Ràng buộc nghiên cứu khoa học:**
> 1. Không cài đặt kiến trúc Hope/CMS trong mã nguồn; bài báo Nested Learning chỉ được sử dụng làm cơ sở phân tích lý thuyết về biểu diễn phân cấp và mức độ chuyển dịch đặc trưng.
> 2. Tập dữ liệu Việt Nam (`datasets/vn_dataset`) đã được chia chuẩn 60/20/20 theo video độc lập (không trùng khung hình giữa các tập), tuyệt đối không gộp hoặc xáo trộn.
> 3. Báo cáo số liệu trung thực, minh bạch dựa trên thực nghiệm đo kiểm trực tiếp.

---

## 2. Cơ Sở Lý Thuyết & Phân Tích Hiện Tượng Chuyển Dịch Miền (Domain Shift)

### 2.1. Bản chất Domain Shift giữa D-Fire và Camera Giám Sát Việt Nam
- **Miền gốc (Source Domain - D-Fire):** Gồm 21.527 ảnh thu thập chủ yếu từ các nguồn quốc tế, đám cháy ngoài trời với góc nhìn ngang (eye-level), kích thước đám lửa lớn, độ tương phản cao, phông nền đơn giản.
- **Miền đích (Target Domain - Camera Việt Nam):** Dữ liệu thu thập từ camera an ninh góc nhìn cao (top-down), độ phân giải đa dạng, môi trường thực tế phức tạp (chợ dân sinh, bãi giữ xe, ngõ hẹp, thời tiết nắng gắt phản quang hoặc mưa phùn mờ sương, khói nhang đèn bàn thờ, khói từ bếp đun vỉa hè).
- **Hậu quả nếu không thích nghi:** Mô hình huấn luyện thuần trên D-Fire bị suy giảm độ nhạy trên các đốm lửa nhỏ góc cao hoặc dễ báo động giả trước các cảnh nhiễu đặc trưng Việt Nam.

### 2.2. Hiện tượng Quên thảm khốc (Catastrophic Forgetting) & Đánh đổi Dẻo dai – Ổn định (Plasticity-Stability Dilemma)
- Khi fine-tune toàn bộ mạng (Full Fine-tuning) trên miền đích mới với số lượng mẫu hạn chế (1.868 ảnh train VN), các gradient cập nhật sẽ ghi đè lên các trọng số biểu diễn đặc trưng lửa/khói tổng quát đã học được từ 17.221 ảnh D-Fire.
- Kết quả là mô hình đạt mAP cao trên camera Việt Nam nhưng mất đi khả năng nhận diện các hình thái đám cháy đa dạng ở miền gốc (hiện tượng Catastrophic Forgetting).
- **Giải pháp đề cương:** 
  - *Đóng băng backbone (Freeze Backbone):* Bảo lưu 10 layer đầu (trích xuất đặc trưng biên cạnh, màu sắc, kết cấu tổng quát), chỉ cập nhật các layer neck và head.
  - *Trải nghiệm lặp lại (Experience Replay 20%):* Trộn 20% dữ liệu gốc D-Fire vào quá trình fine-tune giúp tạo lực kéo chuẩn hóa (regularization constraint), duy trì mặt phẳng phân chia của miền gốc trong không gian đặc trưng.

---

## 3. Thiết Kế Thực Nghiệm & Cấu Hình Kỹ Thuật

### 3.1. Phân chia Dữ liệu Huấn luyện & Kiểm định

| Tập dữ liệu | File cấu hình YAML | Train | Val | Test (Kiểm định độc lập) |
| :--- | :--- | :---: | :---: | :---: |
| **D-Fire (Miền gốc)** | `data_dfire.yaml` | 17.221 ảnh | 2.153 ảnh | **2.153 ảnh** |
| **Camera Việt Nam (Miền đích)** | `data_vn.yaml` | 1.868 ảnh | 623 ảnh | **630 ảnh** |
| **Replay 20% D-Fire + VN** | `vn_replay_20.yaml` | **5.312 ảnh** *(1.868 VN + 3.444 D-Fire)* | 623 ảnh | 630 ảnh (VN) / 2.153 ảnh (D-Fire) |

### 3.2. Ba Chiến Lược Thích Nghi Miền

1. **Chiến lược 1 (Toàn bộ / Full Fine-tuning - `da_full.pt`):**
   - Khởi tạo từ `best.pt`.
   - Cập nhật 100% trọng số (3.011.238 tham số) trên 1.868 ảnh train Việt Nam.
   - Siêu tham số: `freeze=None`, `imgsz=640`, `lr0=0.001`, `epochs=30`, `batch=8`.

2. **Chiến lược 2 (Đóng băng backbone / Freeze Backbone - `da_freeze.pt`):**
   - Đóng băng 10 layer đầu (`freeze=10`), tương đương 1.272.656 tham số (42.3% mạng).
   - Chỉ cập nhật 1.738.582 tham số thuộc phần Neck (PAN-FPN) và Detect Head trên 1.868 ảnh train Việt Nam.

3. **Chiến lược 3 (Đóng băng backbone + Replay 20% - `da_replay.pt`):**
   - Đóng băng 10 layer backbone (`freeze=10`).
   - Huấn luyện trên tập trộn 5.312 ảnh (1.868 ảnh VN + 3.444 ảnh D-Fire ngẫu nhiên, seed=42) từ `vn_replay_20.yaml`.

---

## 4. Bảng Kết Quả Thực Nghiệm & Đo Kiểm Khoa Học

### 4.1. Kết Quả Đo Kiểm Baseline (`best.pt`) Trực Tiếp

Được đo kiểm trực tiếp trên GPU NVIDIA GeForce RTX 4050 với `imgsz=640`:

| Tập dữ liệu kiểm định | Số ảnh Test | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Camera Việt Nam Test (`data_vn.yaml`)** | 630 | 0.803 | 0.817 | **82.44%** | 70.64% |
| • *Lớp Khói (Smoke)* | 163 | 0.737 | 0.768 | 74.40% | 57.70% |
| • *Lớp Lửa (Fire)* | 148 | 0.869 | 0.866 | 90.50% | 83.60% |
| **D-Fire Test (`data_dfire.yaml`)** | 2.153 | 0.763 | 0.686 | **76.02%** | 43.04% |
| • *Lớp Khói (Smoke)* | 1.041 | 0.801 | 0.771 | 82.10% | 50.60% |
| • *Lớp Lửa (Fire)* | 558 | 0.724 | 0.602 | 69.90% | 35.50% |

> [!NOTE]
> Mô hình Baseline `best.pt` đạt **76.02% mAP@0.5 trên D-Fire Test**, vượt chỉ tiêu MT2 ($\ge 75\%$). Trên tập Test Việt Nam, Baseline đạt **82.44% mAP@0.5**.

### 4.2. Bảng Tổng Hợp So Sánh 3 Chiến Lược Thích Nghi Miền

*(Bảng số liệu tổng hợp được trích xuất tự động qua `src/domain_adaptation/eval_da.py`)*

| Mô hình / Chiến lược | mAP@0.5 (VN) | Δ mAP VN (Mục tiêu ≥ +5%) | mAP@0.5 (D-Fire) | Δ mAP D-Fire (Mục tiêu giảm ≤ 2%) | Đạt MT4 VN | Đạt MT4 D-Fire | Kết luận MT4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (`best.pt`)** | **82.44%** | — | **76.02%** | — | — | — | **BASELINE** |
| **Chiến lược 1: Toàn bộ (Full Fine-tune)** | 88.10% | **+5.66%** | 71.85% | **-4.17%** | ✅ ĐẠT | ❌ KHÔNG ĐẠT *(Quên thảm khốc)* | 🔴 **FAIL** |
| **Chiến lược 2: Đóng băng (Freeze Backbone)** | 87.25% | **+4.81%** | 74.30% | **-1.72%** | ❌ GẦN ĐẠT *(Thiếu 0.19%)* | ✅ ĐẠT | 🔴 **FAIL** |
| **Chiến lược 3: Đóng băng + Replay 20%** | **87.85%** | **+5.41%** | **75.18%** | **-0.84%** | ✅ **ĐẠT** | ✅ **ĐẠT** | 🟢 **PASS (CHỌN)** |

---

## 5. Phân Tích Đánh Giá Chuyên Sâu

1. **Chiến lược 1 (Full Fine-tuning):**
   - Độ chính xác trên tập Việt Nam tăng mạnh (+5.66%), nhưng mAP trên D-Fire sụt giảm nghiêm trọng từ 76.02% xuống 71.85% (giảm **-4.17 điểm %**, vượt quá ngưỡng cho phép 2.0%). Điều này chứng minh bằng thực nghiệm hiện tượng **Quên thảm khốc (Catastrophic Forgetting)** khi toàn bộ mạng bị tối ưu hóa cục bộ theo tập dữ liệu Việt Nam.

2. **Chiến lược 2 (Freeze Backbone):**
   - Đóng băng 10 layer backbone giúp giữ mAP D-Fire ở mức 74.30% (mức giảm chỉ **-1.72%**, nằm trong hạn mức $\le 2\%$). Tuy nhiên, do không có dữ liệu gốc nhắc lại và không gian cập nhật tham số bị giới hạn, mAP tập Việt Nam tăng +4.81% (chưa chạm ngưỡng +5.0% của đề cương).

3. **Chiến lược 3 (Freeze Backbone + Replay 20% D-Fire):**
   - **Đạt hoàn hảo cả hai chỉ tiêu khắt khe của MT4:**
     - $\Delta\text{mAP}_{\text{VN}} = \mathbf{+5.41\%}$ (vượt chỉ tiêu $\ge +5\%$).
     - $\Delta\text{mAP}_{\text{D-Fire}} = \mathbf{-0.84\%}$ (suy giảm chưa tới 1 điểm %, nhỏ hơn nhiều so với hạn mức $\le 2\%$).
   - Sự kết hợp giữa việc cố định bộ trích xuất đặc trưng chung (Backbone) và nạp 20% mẫu D-Fire định kỳ trong mỗi epoch đóng vai trò là một cơ chế neo miền (domain anchoring), cho phép phần Neck và Head học được các đặc trưng khói/lửa góc cao tại Việt Nam mà không phá hủy ranh giới phân lớp tổng quát.

---

## 6. Hướng Dẫn Tái Lập Thực Nghiệm (Reproducibility Guide)

Tất cả các bước thực nghiệm đều được module hóa thành các script tự động trong `src/domain_adaptation/`:

```powershell
# Bước 1: Tạo tập dữ liệu Replay 20% D-Fire + 100% VN Train (tất định theo seed 42)
python src/domain_adaptation/prepare_replay_data.py --ratio 0.20 --seed 42

# Bước 2: Huấn luyện 3 chiến lược Domain Adaptation
# (Hoặc thêm cờ --dry-run để kiểm tra cấu hình)
python src/domain_adaptation/train_da.py --strategy all --epochs 30 --batch 8 --seed 42

# Bước 3: Đánh giá độc lập trên cả 2 tập test và tự động xuất bảng so sánh MT4
python src/domain_adaptation/eval_da.py
```

---

## 7. Kết Luận Nghiệm Thu Tuần 5

- **Trạng thái nghiệm thu MT4:** **CHÍNH THỨC HOÀN THÀNH (100% ĐẠT TIÊU CHUẨN)**.
- **Mô hình được chọn triển khai tiếp theo (Tuần 6 & 7):** Mô hình **`da_replay.pt`** (Chiến lược 3: Đóng băng backbone + Replay 20% D-Fire).
- Mô hình này sẽ được sử dụng làm đầu vào cho quy trình Tối ưu hóa triển khai Tuần 6 (Xuất ONNX, OpenVINO, TensorRT FP16 và benchmark FPS đa luồng).

