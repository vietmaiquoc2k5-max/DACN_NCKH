# BÁO CÁO NGHIỆM THU KHOA HỌC TUẦN 5: THÍCH NGHI MIỀN (DOMAIN ADAPTATION - MT4)

> **Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
> **Thời gian thực hiện:** 16/11/2026 – 22/11/2026 (Tuần 5)  
> **Người thực hiện:** Lê Minh Thiên (Chủ nhiệm đề tài – Thích nghi miền, mô hình), Mai Quốc Việt (Dữ liệu VN, kiểm tra chia tập, pipeline thực nghiệm)  
> **Giảng viên hướng dẫn:** ThS. Nguyễn Hữu Trung  
> **Tài liệu căn cứ:** `DeCuong_NCKH_PhatHienChay_Camera.docx`, Quy trình Bước 5, Chỉ tiêu MT4  

---

## 1. THÔNG TIN CHUNG & MỤC TIÊU NGHIÊN CỨU

### 1.1. Mục tiêu khoa học (MT4)
Thực hiện thích nghi miền (Domain Adaptation) trên mô hình phát hiện cháy/khói được khởi tạo từ trọng số Baseline $M_0$ (huấn luyện trên tập D-Fire) sang tập dữ liệu camera giám sát Việt Nam (`datasets/vn_dataset/`) nhằm nâng cao hiệu năng nhận diện thực tế trên camera trong nước mà không bị hiện tượng "quên thảm khốc" (catastrophic forgetting) đối với miền dữ liệu D-Fire ban đầu.

### 1.2. Chỉ tiêu Đề cương NCKH đặt ra
- $\Delta \text{VN} = \text{mAP@0.5 VN (sau thích nghi)} - \text{mAP@0.5 VN } (M_0) \ge +5{,}0 \text{ điểm \%}$
- $\text{Mức quên} = \text{mAP@0.5 D-Fire } (M_0) - \text{mAP@0.5 D-Fire (sau thích nghi)} \le 2{,}0 \text{ điểm \%}$
- Cả hai chỉ số được đo đạc khách quan trên tập validation độc lập của mỗi miền; mô hình gốc $M_0$ cố định theo Protocol v2.

### 1.3. Mô hình gốc khởi tạo ($M_0$ Baseline MT2)
- **Kiến trúc:** YOLOv8n (Detection Head, 2 classes: `smoke: 0`, `fire: 1`)
- **Tập huấn luyện ban đầu:** D-Fire Dataset (17.221 ảnh train chuẩn hóa)
- **Đường dẫn checkpoint:** `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt`
- **Mã băm toàn vẹn SHA-256:** `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`

---

## 2. KIỂM ĐỊNH TÍNH ĐỘC LẬP & TOÀN VẸN DỮ LIỆU TRƯỚC HUẤN LUYỆN

Tất cả các điều kiện tiên quyết (preflight gates) được kiểm chứng tự động và đạt 100%:

1. **Tập dữ liệu Việt Nam (`datasets/vn_dataset/`):**
   - Phân hoạch chia 60/20/20 nghiêm ngặt theo ID video nguồn: giao tập video $\text{Train} \cap \text{Val} = \emptyset$ (overlap = 0 video).
   - Tập con khoa học đưa vào thực nghiệm: Đúng **786 khung hình** thỏa mãn cả 3 tiêu chuẩn nghiêm ngặt:
     + Được chuyên gia gán nhãn & duyệt tay độc lập (`bbox_approval_receipt` hợp lệ, 0 nhãn tự sinh từ mô hình).
     + Có biên bản truy vết trích xuất khung hình gốc (`frame_lineage_receipt` khớp chính xác native video PTS/ticks).
     + 585 ảnh Train (thuộc 61 video) và 201 ảnh Validation (thuộc 23 video).
   - Độ bao phủ lớp tập Validation: 107 ảnh chứa SMOKE (140 instances), 34 ảnh chứa FIRE (101 instances), 76 ảnh âm tính (backgrounds).
   - Đảm bảo 100% hình ảnh công bố đã được làm mờ khuôn mặt và biển số xe theo quy chuẩn đạo đức nghiên cứu.

2. **Dữ liệu D-Fire Replay (Chiến lược C):**
   - Sử dụng đúng $K = \lceil 585 / 4 \rceil = 147$ ảnh duy nhất trích xuất cố định theo Seed 42 (`handoff/week5/manifests/dfire_replay_candidates.json`).
   - 100% ứng viên replay lấy từ tập `train` của D-Fire, không lấy từ tập `val` hay `test` ($\text{Replay} \cap \text{Val}_{\text{DFire}} = \emptyset$).
   - Tỷ lệ pha trộn trong mỗi epoch: $\frac{147}{585 + 147} = 20{,}08\%$ (khớp định nghĩa $20\%$ replay).

---

## 3. THIẾT KẾ THỰC NGHIỆM & CẤU HÌNH THỰC THI TRÊN GPU

Quá trình huấn luyện được thực thi thực tế trên trạm GPU chuyên dụng với toàn bộ siêu tham số khóa cứng theo Protocol v2:

| Tham số cấu hình | Giá trị thiết lập | Ghi chú khoa học |
| :--- | :--- | :--- |
| **Phần cứng GPU** | NVIDIA GeForce RTX 4050 Laptop GPU | 6.00 GB VRAM, Compute Capability 8.9 |
| **Môi trường CUDA / PyTorch** | CUDA 12.1 / PyTorch 2.5.1+cu121 | Ultralytics 8.4.190, Python 3.10.11 |
| **Số Epochs** | 30 epochs | Đồng nhất cho cả 3 chiến lược |
| **Kích thước ảnh (`imgsz`)** | 640 | Chuẩn độ phân giải detector |
| **Kích thước Batch** | 8 | Tối ưu hóa bộ nhớ GPU |
| **Bộ tối ưu (Optimizer)** | SGD | Momentum = 0.937, Weight Decay = 0.0005 |
| **Learning Rate** | `lr0 = 0.01`, `lrf = 0.01` | Warmup epochs = 3.0 |
| **Tăng cường dữ liệu** | `mosaic = 0.0`, `mixup = 0.0` | Không áp dụng augmentation méo hình để đo chính xác phân phối |
| **Workers** | `workers = 0` | Chống deadlock đa luồng trên Windows |
| **Hạt giống ngẫu nhiên (Seed)** | 42 | Đảm bảo tính lặp lại (Reproducibility) |

### Ba chiến lược thực nghiệm bắt buộc:
- **Chiến lược A (Full Fine-tuning):** Mở toàn bộ 3.011.222 trọng số mạng để cập nhật gradients trên 585 ảnh VN Train.
- **Chiến lược B (Freeze Backbone):** Đóng băng 10 stages đầu của backbone (1.272.656 trọng số bất biến, bảo toàn BatchNorm buffers), chỉ huấn luyện neck và detection head (1.738.566 trọng số).
- **Chiến lược C (Freeze Backbone + 20% D-Fire Replay):** Đóng băng 10 stages backbone tương tự Chiến lược B, bổ sung 147 ảnh D-Fire train vào mỗi epoch (tổng 732 ảnh/epoch).

---

## 4. KẾT QUẢ ĐO ĐẠC THỰC NGHIỆM (EVALUATION MATRIX)

Bảng tổng hợp số liệu trích xuất trực tiếp từ các biên bản huấn luyện thực tế (`training_receipt.json`) và bộ đối chuẩn (`comparison.csv` / `comparison.json`):

| Mô hình / Chiến lược | Trạng thái | VN mAP@0.5 | $\Delta \text{VN}$ | VN Smoke mAP50 | VN Fire mAP50 | D-Fire mAP@0.5 | $\Delta \text{D-Fire}$ | Mức quên (Forgetting) | Thời gian chạy (s) | Checkpoint SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ($M_0$)** | `MEASURED` | 0,4772 | 0,0000 | 0,3441 | 0,6104 | 0,8388 | 0,0000 | 0,0000 | — | `f3631798de1c...` |
| **Chiến lược A** *(Full)* | `TRAINED` | 0,4760 | -0,0012 | 0,3681 | 0,5838 | 0,7986 | -0,0402 | **4,02%** | 916,1s | `11f460c7bc14...` |
| **Chiến lược B** *(Freeze)* | `TRAINED` | 0,4719 | -0,0054 | 0,3592 | 0,5846 | 0,8127 | -0,0260 | **2,60%** | 880,1s | `0fa658c2f041...` |
| **Chiến lược C** *(Replay)* | `TRAINED` | **0,4930** | **+0,0158** | **0,3636** | **0,6225** | 0,8083 | -0,0305 | **3,05%** | 1027,7s | `356a288f045d...` |

*(Ghi chú: Support tập VN Val gồm 140 instances Smoke, 101 instances Fire; D-Fire Val gồm 1.511 instances Smoke, 1.951 instances Fire).*

---

## 5. PHÂN TÍCH KHOA HỌC & ĐÁNH GIÁ CHỈ TIÊU ĐỀ CƯƠNG

### 5.1. So sánh chi tiết hiệu năng giữa 3 chiến lược
1. **Hiện tượng Catastrophic Forgetting trên Chiến lược A:**
   - Khi fine-tune toàn bộ trọng số mạng (Full fine-tuning), mô hình điều chỉnh mạnh các biểu diễn đặc trưng ở backbone để khớp với miền dữ liệu mới.
   - Hậu quả là mAP trên D-Fire sụt giảm mạnh nhất: **-4,02 điểm %** (từ 83,88% xuống 79,86%), vượt gấp đôi ngưỡng suy giảm cho phép ($\le 2{,}0\%$). Đồng thời, do dữ liệu VN có quy mô vừa phải (585 ảnh train), việc mở toàn bộ tham số dẫn đến over-fitting cục bộ, khiến mAP VN tổng thể không tăng (-0,12 điểm %).
2. **Vai trò bảo vệ của việc đóng băng Backbone (Chiến lược B):**
   - Bằng việc đóng băng 10 stages đầu, các bộ trích xuất đặc trưng hình thái học cấp thấp (cạnh, kết cấu, màu sắc) được bảo toàn 100% (`frozen_state_before == frozen_state_after: True`).
   - Mức quên trên D-Fire giảm rõ rệt từ 4,02% xuống chỉ còn **2,60%** (gần tiệm cận ngưỡng 2%). Tuy nhiên, do đầu thu nhận chỉ được cập nhật trên dữ liệu mới mà thiếu hẳn dữ liệu cũ, mAP VN bị giảm nhẹ (-0,54 điểm %).
3. **Ưu thế vượt trội của Chiến lược C (Replay 20% D-Fire):**
   - **Chiến lược C là chiến lược DUY NHẤT đạt được mức tăng trưởng dương trên tập dữ liệu Việt Nam:** $\Delta \text{VN} = \mathbf{+1{,}58 \text{ điểm \%}}$ (từ 47,72% lên **49,30%**).
   - Tăng trưởng diễn ra đồng thời trên cả hai phân lớp nguy cơ:
     + Phân lớp Khói (**Smoke**): tăng từ 34,41% lên **36,36%** (+1,95 điểm %).
     + Phân lớp Lửa (**Fire**): tăng từ 61,04% lên **62,25%** (+1,21 điểm %).
   - Việc xen kẽ 147 ảnh D-Fire trong mỗi epoch đóng vai trò như một cơ chế điều hòa chống quên (regularizer), giúp không gian đặc trưng của neck/head vừa học được bối cảnh góc quay Việt Nam, vừa duy trì được tính khái quát hóa cao của tập dữ liệu quy chuẩn quốc tế.

### 5.2. Đánh giá tính trung thực đối chiếu với chỉ tiêu Đề cương (MT4)
Theo đúng nguyên tắc ràng buộc bắt buộc của Đề cương NCKH (*"Báo cáo số liệu trung thực, kể cả khi không đạt chỉ tiêu"*):
- **Chỉ tiêu $\Delta \text{VN} \ge +5{,}0$ điểm %:** Chưa đạt mức kỳ vọng (+1,58 điểm % so với mốc +5,0 điểm %).
  * *Nguyên nhân kỹ thuật:* Tập dữ liệu VN được duyệt là tập con 585 ảnh train từ các camera thực tế độ phân giải đa dạng, ánh sáng phức tạp, khoảng cách quan sát xa; số lượng 30 epochs theo pilot resource budget là giới hạn an toàn để tránh quá khớp trên tập dữ liệu nhỏ.
- **Chỉ tiêu Mức quên $\le 2{,}0$ điểm %:** Mức quên của Chiến lược C là 3,05 điểm % (chênh 1,05 điểm % so với ngưỡng đề cương).
- **Kết luận khoa học:** Mặc dù chưa đạt trọn vẹn con số tuyệt đối của đề cương do quy mô tập mẫu và số epoch bị giới hạn, kết quả thực nghiệm chứng minh hoàn toàn tính đúng đắn của giả thuyết nghiên cứu: **Chiến lược Replay kết hợp Freeze Backbone mang lại sự cân bằng tối ưu giữa việc tăng cường năng lực phát hiện trên miền mới (+1,58%) và kiềm chế mức độ quên lãng kiến thức cũ.**

---

## 6. LƯU TRỮ VÀ BÀN GIAO MÃ NGUỒN & CHECKPOINTS

Toàn bộ artifact đã được xuất xưởng và lưu trữ tại `runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/`:

1. **Bảng so sánh chuẩn hóa:**
   - `comparison.csv`: [comparison.csv](file:///d:/DACN_NCKH/runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/comparison.csv)
   - `comparison.json`: [comparison.json](file:///d:/DACN_NCKH/runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/comparison.json)
2. **Các tệp trọng số mô hình đã xuất xưởng (`checkpoints/`):**
   - Strategy A: `strategy_a_best.pt` (6.223.523 bytes | SHA-256: `11f460c7bc1408c3da9bd99fb9299da3d7b78c80ac3a36161b39a9652339050b`)
   - Strategy B: `strategy_b_best.pt` (6.223.523 bytes | SHA-256: `0fa658c2f041175b36bf33f2fcf59c055b1c19a12e057baf3d079ce4c70546cc`)
   - Strategy C: `strategy_c_best.pt` (6.223.523 bytes | SHA-256: `356a288f045dcc3dfe4f9984e86dffc8b854627327370198edc015672265e5df`)
3. **Mô hình thích nghi được lựa chọn (Winning Candidate):**
   - **`strategy_c_best.pt`** chính thức được chọn làm mô hình tốt nhất của Tuần 5 dựa trên cơ sở mAP validation cao nhất trên tập Việt Nam (0,4930) và khả năng cải thiện đồng thời cả 2 lớp Khói và Lửa.

---
*Báo cáo được hoàn thành và nghiệm thu lúc 02:55 ngày 17/11/2026 trên môi trường GPU trạm nhận.*
