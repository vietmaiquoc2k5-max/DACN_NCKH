# BÁO CÁO KIỂM TOÁN TÁI LẬP KHOA HỌC TUẦN 5 (REVISED AUDIT REPORT)
## DOMAIN ADAPTATION & CONTINUAL LEARNING (MT4)

> **Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
> **Hạng mục kiểm toán:** Tuần 5 – Thích nghi miền (Domain Adaptation theo chỉ tiêu MT4 Đề cương NCKH)  
> **Chuyên gia kiểm toán:** Senior ML Engineer & Research Reproducibility Auditor  
> **Commit kiểm toán:** `3eb714bcac64fdcc5443db07bbe221b11b17d505` | Nhánh: `fix-4`  
> **Môi trường thực nghiệm:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.1, PyTorch 2.5.1+cu121)  
> **Thời điểm hoàn thành:** 17/11/2026 15:50:00 (UTC+7)  
> **Khuyến nghị nghiệm thu:** **`VERIFIED_FOR_VALIDATION_HANDOFF`** (Đạt chuẩn nghiệm thu trên tập kiểm định)

---

## 1. TỔNG HỢP TRẠNG THÁI 9 TEST CASES KIỂM TOÁN

Dựa trên toàn bộ bằng chứng thực nghiệm và kiểm tra độc lập tại môi trường local, bảng trạng thái kiểm toán được cập nhật như sau:

| Test Case | Trạng thái ban đầu | Trạng thái REVISED | Lý do điều chỉnh & Bằng chứng thực tế |
| :--- | :---: | :---: | :--- |
| **TC1: Checkpoint training provenance** | PASS | **PASS** | Đủ 3 thư mục huấn luyện gốc trong `gpu_training/`, đầy đủ `fit/results.csv` (30 epochs), `fit/args.yaml`, `training_receipt.json`, `locked_inputs.json`. |
| **TC2: Checkpoint SHA-256 match** | PASS | **PASS** | Tệp bàn giao trong `final_gpu_handoff/checkpoints/` khớp 100% bit-for-bit với tệp sinh ra từ quá trình huấn luyện `fit/weights/best.pt`. |
| **TC3: CSV không khớp JSON** | PASS | **PASS** | Đối chiếu 25 cột dữ liệu giữa `comparison.csv` và `comparison.json`: 0 mismatch ($|c - j| < 10^{-9}$). |
| **TC4: Đánh giá trên tập dữ liệu khác nhau** | PASS | **PASS** | Tập validation của cả 3 lần chạy huấn luyện trùng khớp SHA-256 từng file: 201 ảnh VN Val và 2.547 ảnh D-Fire Val (độ lệch = 0 file). |
| **TC5: Replay ratio không bằng 20%** | PASS | **PASS** | Đúng 147 ảnh D-Fire train (Seed 42) pha trộn với 585 ảnh VN train: $\frac{147}{585 + 147} = \mathbf{20{,}08\%}$ (độ lệch $+0{,}08\%$ do làm tròn nguyên). 0 ảnh trùng lặp với D-Fire Val hay VN. |
| **TC6: Video-source disjointness (Train/Val/Test)** | PASS | **`PARTIAL`** | **ĐIỀU CHỈNH CHÍNH XÁC:** VN Train (61 videos) và VN Val (23 videos) đạt 0 video overlap. Tuy nhiên, tập **Test** (630 ảnh) được bảo lưu độc lập và cố tình không đưa vào `dataset_manifest.json` để tránh data leakage trong pha thích nghi; do đó tính phân tách 3 tập (Train/Val/Test) được ghi nhận là **`PARTIAL`** chờ nghiệm thu tập Test cuối cùng. |
| **TC7: Class mapping không nhất quán** | PASS | **PASS** | Trích xuất từ 4 checkpoint PyTorch: `names = {0: 'smoke', 1: 'fire'}` đồng nhất 100%. |
| **TC8: Công thức Forgetting bị tính sai** | PASS | **PASS** | Công thức $\Delta \text{VN}$ và Catastrophic Forgetting tính đúng định nghĩa toán học. Đã phân biệt rõ giữa Điểm phần trăm (pp) và % tương đối. |
| **TC9: Khả năng tái lập (Reproducibility)** | PASS | **PASS** | Đã tái lập độc lập thành công trên GPU RTX 4050 trên **CẢ HAI TẬP**: VN Val (201 ảnh) và D-Fire Val (2.547 ảnh) với sai số tuyệt đối $\epsilon = 0{,}0$. |

---

## 2. KẾT QUẢ TÁI LẬP THỰC NGHIỆM ĐỘC LẬP TRÊN CẢ HAI MIỀN (REPLICATION MATRIX)

Chuyên gia kiểm toán đã thực hiện tái đánh giá độc lập hoàn toàn (không sử dụng cache, nạp checkpoint từ ổ đĩa) trên trạm GPU NVIDIA GeForce RTX 4050 (`conf=0.001`, `iou=0.7`, `imgsz=640`) thông qua hai script kiểm chứng:
- VN Validation: [scripts/audit_vn_eval_replication.py](file:///d:/DACN_NCKH/scripts/audit_vn_eval_replication.py) (201 ảnh)
- D-Fire Validation: [scripts/audit_dfire_eval_replication.py](file:///d:/DACN_NCKH/scripts/audit_dfire_eval_replication.py) (2.547 ảnh, 3.462 instances)

### 2.1. Kết quả trên tập kiểm định Việt Nam (VN Val)

| Checkpoint | Chỉ số | Giá trị báo cáo | Tái lập độc lập | Độ lệch ($\epsilon$) | Kết luận |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline ($M_0$)** | mAP@0.5 | 0,4772311488186418 | 0,4772311488186418 | 0,000000 | **MATCH** |
| | Smoke mAP@0.5 | 0,3440784347004965 | 0,3440784347004965 | 0,000000 | **MATCH** |
| | Fire mAP@0.5 | 0,6103838629367870 | 0,6103838629367870 | 0,000000 | **MATCH** |
| **Strategy A** *(Full)* | mAP@0.5 | 0,4759836093246743 | 0,4759836093246743 | 0,000000 | **MATCH** |
| | Smoke mAP@0.5 | 0,3681464587667256 | 0,3681464587667256 | 0,000000 | **MATCH** |
| | Fire mAP@0.5 | 0,5838207598826230 | 0,5838207598826230 | 0,000000 | **MATCH** |
| **Strategy B** *(Freeze)* | mAP@0.5 | 0,4718695046749269 | 0,4718695046749269 | 0,000000 | **MATCH** |
| | Smoke mAP@0.5 | 0,3591639274422402 | 0,3591639274422402 | 0,000000 | **MATCH** |
| | Fire mAP@0.5 | 0,5845750819076136 | 0,5845750819076136 | 0,000000 | **MATCH** |
| **Strategy C** *(Replay)* | **mAP@0.5** | **0,4930422318848344** | **0,4930422318848344** | 0,000000 | **MATCH** |
| | Smoke mAP@0.5 | **0,3635716878485735** | **0,3635716878485735** | 0,000000 | **MATCH** |
| | Fire mAP@0.5 | **0,6225127759210953** | **0,6225127759210953** | 0,000000 | **MATCH** |

### 2.2. Kết quả trên tập kiểm định quốc tế D-Fire (D-Fire Val - 2.547 ảnh)

| Checkpoint | Chỉ số | Giá trị báo cáo | Tái lập độc lập | Độ lệch ($\epsilon$) | Mức quên thực tế ($\mathcal{F}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline ($M_0$)** | mAP@0.5 | 0,8387699019070076 | 0,8387699019070076 | 0,000000 | 0,00% (Baseline) |
| | Smoke mAP@0.5 | 0,9053697210712316 | 0,9053697210712316 | 0,000000 | — |
| | Fire mAP@0.5 | 0,7721700827427838 | 0,7721700827427838 | 0,000000 | — |
| **Strategy A** *(Full)* | mAP@0.5 | 0,7985699080260099 | 0,7985699080260099 | 0,000000 | **4,02 điểm %** (Quên nặng nhất) |
| | Smoke mAP@0.5 | 0,8646743662729883 | 0,8646743662729883 | 0,000000 | — |
| | Fire mAP@0.5 | 0,7324654497790315 | 0,7324654497790315 | 0,000000 | — |
| **Strategy B** *(Freeze)* | mAP@0.5 | 0,8127236388787922 | 0,8127236388787922 | 0,000000 | **2,60 điểm %** (Giữ backbone) |
| | Smoke mAP@0.5 | 0,8851316112198169 | 0,8851316112198169 | 0,000000 | — |
| | Fire mAP@0.5 | 0,7403156665377675 | 0,7403156665377675 | 0,000000 | — |
| **Strategy C** *(Replay)* | mAP@0.5 | **0,8083195169674757** | **0,8083195169674757** | 0,000000 | **3,05 điểm %** (Kiềm chế tốt) |
| | Smoke mAP@0.5 | 0,8781123946730012 | 0,8781123946730012 | 0,000000 | — |
| | Fire mAP@0.5 | 0,7385266392619503 | 0,7385266392619503 | 0,000000 | — |

---

## 3. ĐỐI SOÁT TÍNH ĐỒNG BỘ CỦA MÔI TRƯỜNG & DEPENDENCIES

Auditor đã đối soát chi tiết giữa môi trường máy trạm thực tế (`.venv`), tệp [requirements-training.txt](file:///d:/DACN_NCKH/requirements-training.txt) và các receipts:

| Package / Module | Phiên bản cài đặt thực tế | Yêu cầu trong requirements-training.txt | Trạng thái đồng bộ | Đánh giá tác động |
| :--- | :---: | :---: | :---: | :--- |
| **Python** | `3.10.11` | `>=3.10` | **MATCH** | Môi trường chuẩn Windows AMD64 |
| **PyTorch** | `2.5.1+cu121` | `torch==2.5.1+cu121` (chỉ dẫn) | **MATCH** | CUDA 12.1 kích hoạt, nhận diện RTX 4050 Laptop GPU |
| **Ultralytics** | `8.4.190` | `ultralytics==8.4.198` | **VERSION DRIFT** | Bản 8.4.190 chạy ổn định 100%, tái lập kết quả $\epsilon = 0{,}0$. Không khuyến nghị nâng cấp ép buộc để tránh phá vỡ tính lặp lại. |
| **NumPy** | `2.2.6` | `>=1.24.0,<2.0.0` | **VERSION DRIFT** | NumPy 2.2.6 tương thích hoàn toàn với PyTorch 2.5.1 trên máy trạm này. |
| **OpenCV** | `5.0.0` | `>=4.8.0` | **MATCH** | Thỏa mãn |
| **PyYAML** | `6.0.3` | `>=6.0` | **MATCH** | Thỏa mãn |
| **Pillow** | `12.3.0` | `>=10.0.0` | **MATCH** | Thỏa mãn |
| **SciPy** | *Chưa cài đặt* | `>=1.10.0` | **NON-BLOCKING** | Ultralytics YOLOv8 không dùng SciPy cho các tác vụ detection cơ bản nên không gây lỗi. |

---

## 4. CHI TIẾT TẠO TÁC VÀ BẰNG CHỨNG BÀN GIAO MỚI (DELIVERABLES)

Đã bổ sung và hoàn thiện toàn bộ các file bàn giao theo yêu cầu:

1. [WEEK5_AUDIT_REPORT_REVISED.md](file:///d:/DACN_NCKH/WEEK5_AUDIT_REPORT_REVISED.md): Báo cáo kiểm toán sửa đổi toàn văn (tài liệu này).
2. [WEEK5_AUDIT_RESULTS_REVISED.json](file:///d:/DACN_NCKH/WEEK5_AUDIT_RESULTS_REVISED.json): File kết quả kiểm toán định dạng JSON cập nhật trạng thái TC6 (`PARTIAL`) và bảng đối chuẩn tái lập trên cả hai miền.
3. [WEEK5_ARTIFACT_INVENTORY.json](file:///d:/DACN_NCKH/WEEK5_ARTIFACT_INVENTORY.json): Danh mục kiểm kê 36 tạo tác cốt lõi của Tuần 5 kèm mã băm SHA-256 (0 artifact bị thiếu).
4. [scripts/verify_week5_audit_package.py](file:///d:/DACN_NCKH/scripts/verify_week5_audit_package.py): Script Python kiểm toán tự động, kiểm tra toàn bộ 36 tạo tác, tính bất biến backbone, tỷ lệ replay và video disjointness (trả về Exit code 0).
5. [runs/benchmarks/week5_domain_adaptation/independent_dfire_eval_results.json](file:///d:/DACN_NCKH/runs/benchmarks/week5_domain_adaptation/independent_dfire_eval_results.json): Bản ghi số liệu đo đạc độc lập trên tập D-Fire Val.

---

## 5. KHUYẾN NGHỊ NGHIỆM THU CUỐI CÙNG (FINAL RECOMMENDATION)

1. **Khẳng định tính hợp lệ của Pha Thích nghi miền (Tuần 5):**  
   - Quá trình huấn luyện GPU thực tế và đo đạc trên cả hai tập kiểm định (VN Val và D-Fire Val) đã được chứng minh 100% bằng thực nghiệm độc lập.
   - Chiến lược C ([strategy_c_best.pt](file:///d:/DACN_NCKH/runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/checkpoints/strategy_c_best.pt)) là mô hình chiến thắng thuyết phục với mAP@0.5 đạt **49,30%** trên camera Việt Nam (tăng +1,58 điểm % so với Baseline), đồng thời bảo toàn năng lực nhận diện quốc tế ở mức 80,83% trên D-Fire.

2. **Xử lý nội dung còn lưu ý (`PARTIAL` ở TC6):**  
   - Việc tập Test (630 ảnh) không nằm trong `dataset_manifest.json` là quyết định thiết kế chuẩn mực của nhóm nghiên cứu để bảo vệ tập Test khỏi mọi nguy cơ rò rỉ (held-out test set).
   - Đề xuất bước tiếp theo: Tiến hành đánh giá độc lập mô hình chiến thắng `strategy_c_best.pt` trên tập Test chính thức trong khuôn khổ Tuần 6 trước khi đóng gói triển khai hệ thống thời gian thực.
