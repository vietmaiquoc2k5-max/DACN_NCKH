# BÁO CÁO KIỂM TOÁN ĐỘC LẬP KHOA HỌC: DOMAIN ADAPTATION TUẦN 5 (MT4)

> **Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
> **Hạng mục kiểm toán:** Tuần 5 – Thích nghi miền (Domain Adaptation & Continual Learning)  
> **Chuyên gia kiểm toán (Auditor):** Senior ML Engineer & Research Auditor (YOLO, PyTorch, Domain Adaptation & Continual Learning)  
> **Nhánh thực thi / Commit:** `fix-4` | Môi trường: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.1, PyTorch 2.5.1+cu121)  
> **Thời điểm kiểm toán:** 17/11/2026 15:20:00 (UTC+7)  
> **Quyết định nghiệm thu:** **VERIFIED (XÁC NHẬN TOÀN DIỆN TÍNH TOÀN VẸN & KHẢ NĂNG TÁI LẬP)**

---

## 1. TỔNG QUAN VÀ PHẠM VI KIỂM TOÁN (EXECUTIVE SUMMARY)

Kiểm toán độc lập toàn bộ các kết quả thực nghiệm, dữ liệu đầu vào, cấu hình huấn luyện, nhật ký huấn luyện (training logs), checkpoint trọng số mô hình và báo cáo khoa học của Tuần 5 (Thích nghi miền - Domain Adaptation theo chỉ tiêu MT4 của Đề cương NCKH).

### 1.1. Các tạo tác (Artifacts) được kiểm toán
1. **Checkpoints trọng số:**
   - Baseline ($M_0$): `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt`
   - Strategy A: `runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/checkpoints/strategy_a_best.pt`
   - Strategy B: `runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/checkpoints/strategy_b_best.pt`
   - Strategy C: `runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/checkpoints/strategy_c_best.pt`
2. **Dữ liệu đối chuẩn & Nhật ký:**
   - Bảng tổng hợp đối chuẩn: `runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/comparison.csv`
   - Dữ liệu JSON đối chuẩn: `runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/comparison.json`
   - Thư mục huấn luyện GPU: `runs/benchmarks/week5_domain_adaptation/gpu_training/` (`run_full_finetune_seed42`, `run_freeze_backbone_seed42`, `run_replay_20_percent_seed42`)
   - Báo cáo khoa học gốc: `docs/WEEK5_DOMAIN_ADAPTATION_REPORT.md`

### 1.2. Kết luận tổng quan (Classification of Objectives)
Toàn bộ 8 mục tiêu kiểm toán và 9 ca kiểm thử (Test Cases) đã được rà soát với bằng chứng thực nghiệm rõ ràng.

| Hạng mục kiểm toán | Phân loại | Bằng chứng tóm tắt |
| :--- | :---: | :--- |
| **1. Provenance của checkpoint** | **PASS** | 100% checkpoint có đầy đủ fit logs, receipts, khớp SHA-256 tuyệt đối giữa thư mục fit và bàn giao |
| **2. Cấu hình & Log 3 chiến lược** | **PASS** | 30 epochs thực tế trên GPU, SGD optimizer, lr0=0.01; chứng minh bất biến backbone (0/162 mutated tensors) |
| **3. Phân chia dữ liệu & Nhãn người** | **PASS** | 786 ảnh VN được duyệt tay bởi REVIEWER01; chia 60/20 theo video ID (0 video trùng lặp giữa Train và Val) |
| **4. Tỷ lệ 20% D-Fire Replay** | **PASS** | 147 ảnh D-Fire train pha trộn với 585 ảnh VN train đạt tỷ lệ 20,08%; 0 ảnh trùng với D-Fire Val/Test |
| **5. Tái lập đánh giá độc lập** | **PASS** | Tự chạy eval độc lập trên GPU đạt cùng kết quả với độ lệch tuyệt đối $\epsilon = 0{,}0$ |
| **6. Công thức $\Delta$mAP & Forgetting** | **PASS** | Công thức toán học chuẩn xác; phân định rõ giữa Điểm phần trăm (pp) và Phần trăm tương đối (%) |
| **7. Đối chiếu CSV, JSON, Markdown** | **PASS** | Khớp chính xác 100% từng con số trên mọi bảng biểu và định dạng |
| **8. Ngăn ngừa Data Leakage & Thiên vị** | **PASS** | Không sử dụng test set để chọn mô hình hay chỉnh ngưỡng; class mapping `{0: smoke, 1: fire}` đồng nhất |

---

## 2. KẾT QUẢ ĐÁNH GIÁ 9 TEST CASES BẮT BUỘC

### Test Case 1: Checkpoint không có training provenance
- **Phân loại:** **PASS**
- **Bằng chứng:**
  Mỗi checkpoint đều có cây thư mục huấn luyện độc lập hoàn chỉnh tại `runs/benchmarks/week5_domain_adaptation/gpu_training/`:
  - `run_full_finetune_seed42/`: chứa `fit/args.yaml`, `fit/results.csv` (đủ 30 epochs), `fit/weights/best.pt`, `fit/weights/last.pt`, `training_receipt.json`, `locked_inputs.json`.
  - `run_freeze_backbone_seed42/`: chứa đầy đủ cấu hình đóng băng 10 layers, log biến động tham số, `training_receipt.json`.
  - `run_replay_20_percent_seed42/`: chứa `replay_selection.json`, bảng theo dõi 732 lượt trình bày ảnh/epoch (585 VN + 147 D-Fire).
- **Kết luận:** Đầy đủ bằng chứng xuất xứ huấn luyện gốc, không có checkpoint nào được đưa vào tùy tiện.

### Test Case 2: Checkpoint không khớp SHA-256
- **Phân loại:** **PASS**
- **Bằng chứng:**
  So sánh trực tiếp hàm băm SHA-256 giữa tệp sinh ra từ quá trình huấn luyện (`fit/weights/best.pt`) và tệp xuất xưởng (`final_gpu_handoff/checkpoints/`):
  ```
  Baseline (M0):
    Path: runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt
    SHA-256: f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6 (6.247.332 bytes)
  
  Strategy A (Full Finetune):
    Fit path:     gpu_training/run_full_finetune_seed42/fit/weights/best.pt
    Handoff path: final_gpu_handoff/checkpoints/strategy_a_best.pt
    SHA-256:      11f460c7bc1408c3da9bd99fb9299da3d7b78c80ac3a36161b39a9652339050b (6.223.523 bytes)
    Trạng thái:   TRÙNG KHỚP 100% BIT-FOR-BIT

  Strategy B (Freeze Backbone):
    Fit path:     gpu_training/run_freeze_backbone_seed42/fit/weights/best.pt
    Handoff path: final_gpu_handoff/checkpoints/strategy_b_best.pt
    SHA-256:      0fa658c2f041175b36bf33f2fcf59c055b1c19a12e057baf3d079ce4c70546cc (6.223.523 bytes)
    Trạng thái:   TRÙNG KHỚP 100% BIT-FOR-BIT

  Strategy C (Freeze Backbone + 20% Replay):
    Fit path:     gpu_training/run_replay_20_percent_seed42/fit/weights/best.pt
    Handoff path: final_gpu_handoff/checkpoints/strategy_c_best.pt
    SHA-256:      356a288f045dcc3dfe4f9984e86dffc8b854627327370198edc015672265e5df (6.223.523 bytes)
    Trạng thái:   TRÙNG KHỚP 100% BIT-FOR-BIT
  ```
- **Kết luận:** 0 mismatch.

### Test Case 3: CSV không khớp JSON
- **Phân loại:** **PASS**
- **Bằng chứng:**
  Thực hiện đối soát tự động toàn bộ 25 trường thông tin trên 4 dòng dữ liệu giữa [comparison.csv](file:///d:/DACN_NCKH/runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/comparison.csv) và [comparison.json](file:///d:/DACN_NCKH/runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/comparison.json):
  - Số dòng: 4 dòng (Baseline, freeze_backbone, full_finetune, replay_20_percent).
  - Số trường: 25 cột (`vn_map50`, `delta_vn_map50`, `vn_smoke_map50`, `vn_fire_map50`, `dfire_map50`, `forgetting_map50`, `seed`, v.v.).
  - Số lượng sai lệch phát hiện: **0 (Mismatches: 0)** với độ chính xác dấu phẩy động tuyệt đối ($|c - j| < 10^{-9}$).

### Test Case 4: Mô hình được đánh giá trên các tập dữ liệu khác nhau
- **Phân loại:** **PASS**
- **Bằng chứng:**
  Quét toàn bộ hàm băm SHA-256 của từng file ảnh và label trong thư mục validation của 3 lần chạy huấn luyện:
  - Tập VN Val: Mỗi run đều chứa đúng 201 ảnh và 201 nhãn; độ lệch giữa run 0, run 1 và run 2 là **0 file** (trùng khớp SHA-256 100%).
  - Tập D-Fire Val: Mỗi run đều chứa đúng 2.547 ảnh và 2.547 nhãn; độ lệch là **0 file** (trùng khớp SHA-256 100%).
- **Kết luận:** Cả 4 mô hình (Baseline, A, B, C) được đánh giá trên cùng một tập kiểm định cố định, không có sự sai lệch tập dữ liệu.

### Test Case 5: Replay ratio không thực sự bằng 20%
- **Phân loại:** **PASS**
- **Bằng chứng:**
  - File biên bản trích xuất: `runs/benchmarks/week5_domain_adaptation/gpu_training/run_replay_20_percent_seed42/replay_selection.json`.
  - Số ảnh VN Train: $N_{\text{VN}} = 585$ ảnh.
  - Số ảnh D-Fire Replay trích xuất: $N_{\text{replay}} = \lceil 585 / 4 \rceil = 147$ ảnh (Seed = 42, từ tập train của D-Fire).
  - Tỷ lệ pha trộn trong batch/epoch:
    $$\text{Tỷ lệ thực tế} = \frac{147}{585 + 147} = \frac{147}{732} = 20{,}08197\% \approx 20{,}08\%$$
  - Sai số làm tròn so với mốc $20{,}0\%$: $+0{,}08197\%$ (thỏa mãn quy chuẩn lấy mẫu số nguyên).
  - Kiểm tra rò rỉ: 147 ảnh replay có giao tập với D-Fire Val = **0 ảnh**, giao tập với VN Dataset = **0 ảnh**.

### Test Case 6: Validation/test bị trùng video nguồn (Data Leakage by Video)
- **Phân loại:** **PASS**
- **Bằng chứng:**
  Trích xuất ID và hàm băm SHA-256 của toàn bộ video nguồn sinh ra các khung hình trong `handoff/week5/dataset_manifest.json`:
  - Số lượng video nguồn sinh ra 585 ảnh VN Train: **61 video**.
  - Số lượng video nguồn sinh ra 201 ảnh VN Val: **23 video**.
  - Giao tập video nguồn: $\text{Videos}_{\text{Train}} \cap \text{Videos}_{\text{Val}} = \emptyset$ (**0 video trùng lặp**).
  - Giao tập nhóm bối cảnh cảnh quay (`scene_group_id`): **0 nhóm trùng lặp**.
- **Kết luận:** Tuân thủ 100% nguyên tắc phân chia Group-K-Fold theo video độc lập của Đề cương NCKH.

### Test Case 7: Class mapping không nhất quán
- **Phân loại:** **PASS**
- **Bằng chứng:**
  Trích xuất thuộc tính `names` trực tiếp từ tệp checkpoint PyTorch bằng Ultralytics:
  - `baseline_m0`: `{0: 'smoke', 1: 'fire'}`
  - `strategy_a_best.pt`: `{0: 'smoke', 1: 'fire'}`
  - `strategy_b_best.pt`: `{0: 'smoke', 1: 'fire'}`
  - `strategy_c_best.pt`: `{0: 'smoke', 1: 'fire'}`
  - Staged YAML configs (`vn.yaml`, `dfire_eval.yaml`): `{0: 'smoke', 1: 'fire'}`
- **Kết luận:** Thứ tự phân lớp hoàn toàn nhất quán trên toàn bộ pipeline.

### Test Case 8: Công thức Forgetting bị tính sai
- **Phân loại:** **PASS**
- **Bằng chứng:**
  - Định nghĩa Catastrophic Forgetting trong lý thuyết Continual Learning:
    $$\mathcal{F} = \text{mAP}_{\text{source}}(M_0) - \text{mAP}_{\text{source}}(M_{\text{adapted}})$$
  - Đối chiếu số học:
    + Baseline D-Fire mAP50 = $0{,}83876990$
    + Strategy A D-Fire mAP50 = $0{,}79856991 \implies \mathcal{F} = 0{,}83876990 - 0{,}79856991 = 0{,}04019999$ (**4,02 điểm %**)
    + Strategy B D-Fire mAP50 = $0{,}81272364 \implies \mathcal{F} = 0{,}83876990 - 0{,}81272364 = 0{,}02604626$ (**2,60 điểm %**)
    + Strategy C D-Fire mAP50 = $0{,}80831952 \implies \mathcal{F} = 0{,}83876990 - 0{,}80831952 = 0{,}03045038$ (**3,05 điểm %**)
  - Định nghĩa $\Delta \text{VN}$:
    $$\Delta \text{VN} = \text{mAP}_{\text{VN}}(M_{\text{adapted}}) - \text{mAP}_{\text{VN}}(M_0)$$
    + Strategy A: $0{,}47598361 - 0{,}47723115 = -0{,}00124754$ (**-0,12 điểm %**)
    + Strategy B: $0{,}47186950 - 0{,}47723115 = -0{,}00536164$ (**-0,54 điểm %**)
    + Strategy C: $0{,}49304223 - 0{,}47723115 = +0{,}01581108$ (**+1,58 điểm %**)
- **Kết luận:** Mọi phép tính khớp chính xác định nghĩa toán học.

### Test Case 9: Kết quả báo cáo không thể tái lập
- **Phân loại:** **PASS**
- **Bằng chứng:**
  Chuyên gia kiểm toán đã chạy một script đánh giá độc lập hoàn toàn (không dùng cache, nạp lại checkpoint từ file) trên GPU NVIDIA RTX 4050 với cùng giao thức kiểm định (`imgsz=640`, `conf=0.001`, `iou=0.7`, `device='cuda:0'`).
  Kết quả thu được khớp tuyệt đối đến 16 chữ số thập phân (xem Bảng 3.1).

---

## 3. BẢNG ĐỐI CHIẾU TÁI LẬP ĐỘC LẬP (REPLICATION MATRIX)

### 3.1. Đối chiếu kết quả trên tập kiểm định Việt Nam (VN Val - 201 ảnh)

| Mô hình | Chỉ số | Báo cáo trong Artifacts | Tái lập độc lập (Auditor) | Độ lệch ($\epsilon$) | Trạng thái |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline ($M_0$)** | mAP@0.5 | 0,4772311488186418 | 0,4772311488186418 | 0,0000000000000000 | **MATCH** |
| | mAP@0.5:0.95 | 0,3411986854548823 | 0,3411986854548823 | 0,0000000000000000 | **MATCH** |
| | Smoke mAP@0.5 | 0,3440784347004965 | 0,3440784347004965 | 0,0000000000000000 | **MATCH** |
| | Fire mAP@0.5 | 0,6103838629367870 | 0,6103838629367870 | 0,0000000000000000 | **MATCH** |
| **Strategy A** *(Full)* | mAP@0.5 | 0,4759836093246743 | 0,4759836093246743 | 0,0000000000000000 | **MATCH** |
| | mAP@0.5:0.95 | 0,3356247601517232 | 0,3356247601517232 | 0,0000000000000000 | **MATCH** |
| | Smoke mAP@0.5 | 0,3681464587667256 | 0,3681464587667256 | 0,0000000000000000 | **MATCH** |
| | Fire mAP@0.5 | 0,5838207598826230 | 0,5838207598826230 | 0,0000000000000000 | **MATCH** |
| **Strategy B** *(Freeze)* | mAP@0.5 | 0,4718695046749269 | 0,4718695046749269 | 0,0000000000000000 | **MATCH** |
| | mAP@0.5:0.95 | 0,3328816130774981 | 0,3328816130774981 | 0,0000000000000000 | **MATCH** |
| | Smoke mAP@0.5 | 0,3591639274422402 | 0,3591639274422402 | 0,0000000000000000 | **MATCH** |
| | Fire mAP@0.5 | 0,5845750819076136 | 0,5845750819076136 | 0,0000000000000000 | **MATCH** |
| **Strategy C** *(Replay)* | mAP@0.5 | **0,4930422318848344** | **0,4930422318848344** | 0,0000000000000000 | **MATCH** |
| | mAP@0.5:0.95 | 0,3407927382053683 | 0,3407927382053683 | 0,0000000000000000 | **MATCH** |
| | Smoke mAP@0.5 | **0,3635716878485735** | **0,3635716878485735** | 0,0000000000000000 | **MATCH** |
| | Fire mAP@0.5 | **0,6225127759210953** | **0,6225127759210953** | 0,0000000000000000 | **MATCH** |

---

## 4. KIỂM TOÁN TÍNH BẤT BIẾN CỦA BACKBONE (LAYER-LEVEL INVARIANCE AUDIT)

Để đảm bảo việc đóng băng backbone trong Strategy B và Strategy C không bị trôi do tính toán EMA (Exponential Moving Average) hay sai lệch cấu hình optimizer, chuyên gia kiểm toán đã thực hiện phép trừ tensor trực tiếp giữa mô hình Baseline $M_0$ và các checkpoint thích nghi đối với 10 stages đầu của backbone (`model.0.` đến `model.9.`):

- **Số tensor trọng số trong Backbone (layers 0-9):** 162 tensors.
- **Số tensor trọng số trong Head & Neck (layers 10-22):** 193 tensors.

| Mô hình kiểm toán | Số tensor Backbone bị biến đổi | Max chênh lệch trọng số Backbone | Số tensor Head/Neck bị biến đổi | Max chênh lệch trọng số Head/Neck | Kết luận kiểm toán |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Strategy A** *(Full Finetune)* | **135 / 162** | 14,6875 | 162 / 193 | 4,1250 | Mở toàn bộ mạng đúng thiết kế |
| **Strategy B** *(Freeze Backbone)* | **0 / 162** | **0,0000** | 162 / 193 | 7,5312 | **Backbone bảo toàn 100% tuyệt đối** |
| **Strategy C** *(Freeze + Replay)* | **0 / 162** | **0,0000** | 162 / 193 | 7,1875 | **Backbone bảo toàn 100% tuyệt đối** |

Bằng chứng tensor này khẳng định cơ chế bảo vệ backbone (`freeze_backbone_statistics` và `state_digest`) hoạt động hoàn hảo trong suốt 30 epochs.

---

## 5. ĐỐI CHIẾU CHỈ TIÊU ĐỀ CƯƠNG NCKH (COMPLIANCE & HONESTY REVIEW)

Đề cương NCKH quy định tại mục Chỉ tiêu MT4:
- $\Delta \text{VN} \ge +5{,}0 \text{ điểm \%}$
- $\text{Mức quên} \le 2{,}0 \text{ điểm \%}$

### 5.1. So sánh Điểm phần trăm (Percentage Points) vs Phần trăm tương đối (Relative %)

| Chỉ số khoa học | Baseline ($M_0$) | Strategy C (Thắng cuộc) | Thay đổi theo Điểm phần trăm (pp) | Thay đổi tương đối (%) |
| :--- | :---: | :---: | :---: | :---: |
| **mAP@0.5 trên VN** | 47,72% | 49,30% | **+1,58 điểm %** | **+3,31%** |
| - Lớp Khói (Smoke) | 34,41% | 36,36% | +1,95 điểm % | +5,67% |
| - Lớp Lửa (Fire) | 61,04% | 62,25% | +1,21 điểm % | +1,98% |
| **mAP@0.5 trên D-Fire** | 83,88% | 80,83% | **-3,05 điểm %** (Mức quên) | **-3,63%** (Mức quên tương đối) |

### 5.2. Đánh giá tính trung thực (Đạo đức NCKH)
- Kết quả thực nghiệm của Strategy C đạt mức tăng **+1,58 điểm %** (dương tính trên cả 2 phân lớp Khói và Lửa) và mức quên **3,05 điểm %**.
- Con số này chưa đạt trọn vẹn mục tiêu danh nghĩa tối đa của Đề cương (+5,0 điểm % và $\le 2{,}0$ điểm %).
- Báo cáo gốc `docs/WEEK5_DOMAIN_ADAPTATION_REPORT.md` đã phân tích trung thực nguyên nhân kỹ thuật (quy mô tập mẫu duyệt tay 585 ảnh và giới hạn an toàn 30 epochs để tránh overfitting), không thực hiện bất kỳ hành vi "làm đẹp" số liệu hay sinh nhãn giả nào. Điều này hoàn toàn tuân thủ điều khoản bắt buộc của Đề cương: *"Báo cáo số liệu trung thực, kể cả khi không đạt chỉ tiêu"*.

---

## 6. DANH MỤC TẠO TÁC BÀN GIAO KIỂM TOÁN (AUDIT DELIVERABLES)

Các tệp được tạo lập và lưu trữ trong thư mục gốc của repository:
1. `WEEK5_AUDIT_REPORT.md`: Báo cáo kiểm toán toàn văn chi tiết này.
2. `WEEK5_AUDIT_RESULTS.json`: Toàn bộ kết quả đối chiếu máy đọc (machine-readable) định dạng JSON.
3. `WEEK5_CHECKPOINT_HASHES.txt`: Bảng kê khai chi tiết mã SHA-256, dung lượng và xuất xứ của tất cả các checkpoint.

---

## 7. QUYẾT ĐỊNH NGHIỆM THU (FINAL VERDICT)

Căn cứ trên các bằng chứng thực nghiệm độc lập:
1. **Provenance:** Đầy đủ và minh bạch 100%.
2. **Reproducibility:** Tái lập chính xác tuyệt đối với $\epsilon = 0{,}0$.
3. **Integrity:** Không có rò rỉ dữ liệu (0 frame/video overlap), không thiên vị test set, không nhãn giả mạo.
4. **Consistency:** Bảng biểu CSV, JSON, Markdown và checkpoint đồng nhất hoàn toàn.

**KẾT LUẬN:** **CHẤP THUẬN NGHIỆM THU KHOA HỌC TUẦN 5 (STATUS: VERIFIED).**  
Mô hình `strategy_c_best.pt` chính thức được công nhận là mô hình tối ưu nhất của giai đoạn Thích nghi miền và sẵn sàng chuyển giao cho Tuần 6 (Ablation, Triển khai ONNX/OpenVINO và Đánh giá thời gian thực).
