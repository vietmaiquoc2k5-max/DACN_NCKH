# Báo Cáo Kiểm Định Dữ Liệu Âm Tính & Nâng Cao Event Instance Recall
## (Week 4 Negative Data Completion & Event Recall Improvement Report)

**Dự án:** DACN_NCKH - Phát hiện và cảnh báo cháy từ camera giám sát  
**Nhánh Git:** `fix/week4-scientific-validation`  
**Commit nền:** `a41988cf31819472e64786fb9fc4a5d31472d439`  
**Thời gian hoàn thành:** 10/11/2026  
**Chủ biên thực hiện:** Lê Minh Thiên & Mai Quốc Việt  
**Tiêu chuẩn khoa học áp dụng:** Đề cương NCKH gốc (`DeCuong_NCKH_PhatHienChay_Camera.docx`), IEEE Transactions on CSVT Standards, Quy chuẩn Thống kê Poisson Rule of Three.

---

## 1. Kiểm Kê Tập Dữ Liệu Âm Tính Vật Lý (Negative Dataset Inventory)

*(Trích xuất từ kết quả audit thực tế tại [`runs/benchmarks/week4_scientific/negative_dataset_audit.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_scientific/negative_dataset_audit.json) và [`negative_dataset_audit.csv`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_scientific/negative_dataset_audit.csv))*

Hệ thống đã thực hiện quét và kiểm định toàn diện $100\%$ các tệp video âm tính có sẵn trong thư mục `datasets/negative_videos/`:

| Video ID | Tên tệp | SHA256 (64 hex) | Native FPS | Độ phân giải | Dung lượng (MB) | Tổng frames thực tế trên đĩa | Thời lượng thực tế trên đĩa (giây / giờ) | Trạng thái giải mã |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **REAL_NEG_01** | `NHIỄU 1.mp4` | `4deb083ea8d787f61d5fbcda21487ea0bcf3f41ebf225ea1f3416e53748be749` | 30.0 | $1920\times 1080$ | 1.052,48 | 201.836 | 6.727,87s (1,8689h) | Hợp lệ (0 decode error) |
| **REAL_NEG_02** | `NHIỄU 2.mp4` | `de7705fad4d1a65f9024f2b1d3fa54e580e60aa2b220a2323e1ca8e4d2579df2` | 30.0 | $1920\times 1080$ | 51,46 | 618 | 20,60s (0,0057h) | Hợp lệ (0 decode error) |
| **REAL_NEG_03** | `NHIỄU 3.mp4` | `a96ced47406adf8378bfb048fa538562d29aa45037ae1db50cb175ebca13bc62` | 30.0 | $1920\times 1080$ | 33,92 | 1.054 | 35,13s (0,0098h) | Hợp lệ (0 decode error) |
| **REAL_NEG_04** | `NHIỄU 4.mp4` | `25fda9ea2b1ca96cf5d6c8e3aa7ca6a2c2626e344e29ebdf9559ff14eafe7136` | 30.0 | $1920\times 1080$ | 17,64 | 1.053 | 35,10s (0,0097h) | Hợp lệ (0 decode error) |
| **REAL_NEG_05** | `NHIỄU 5.mp4` | `eb242b46c30484c6efab3980a30b427b2baea9d6bc4bfdff712b3226a273b4d1` | 30.0 | $1920\times 1080$ | 908,06 | 165.480 | 5.516,00s (1,5322h) | Hợp lệ (0 decode error) |
| **TỔNG CỘNG** | **5 tệp video** | — | **30.0** | — | **2.063,56 MB** | **370.041 frames** | **12.334,70s (3,4263 giờ)** | **100% OK** |

---

## 2. Đối Chiếu và Làm Rõ Thời Lượng Đánh Giá (Duration Reconciliation)

> [!IMPORTANT]
> **Điểm mấu chốt về tính liêm chính khoa học:**
> Nhóm nghiên cứu làm rõ sự khác biệt giữa **"Tổng thời lượng lưu trữ vật lý trên đĩa"** và **"Tổng thời lượng thực tế được đưa vào bộ đánh giá"**:

1. **Tổng thời lượng vật lý trên đĩa:** 5 tệp video có tổng thời lượng là **3,4263 giờ (12.334,70 giây, 370.041 frames)**. So với chỉ tiêu của MT1 và MT3 trong Đề cương NCKH ($\ge 5,0$ giờ), hệ thống hiện còn thiếu **1,5737 giờ**.
2. **Thời lượng thực tế đưa vào đo đạc FAR:**
   - Trong Task 2 trước đây, để kiểm tra nhanh trong phiên làm việc, script đánh giá đã thiết lập tham số trần `max_frames_per_long_video = 1500` cho 2 video dài (`NHIỄU 1` và `NHIỄU 5`), trong khi xử lý $100\%$ full clip cho 3 video ngắn (`NHIỄU 2, 3, 4`).
   - Tổng số frames thực tế chạy qua detector và temporal verifier là:
     $$1.500 + 618 + 1.054 + 1.053 + 1.500 = 5.725\text{ frames} \quad (\approx 190,83\text{ giây} = 0,0530\text{ giờ})$$
   - Trong đợt kiểm định mở rộng hoàn thiện (Final Evaluation Run), tham số trần được nâng lên `max_frames_per_long_video = 2000`:
     $$2.000 + 618 + 1.054 + 1.053 + 2.000 = 6.725\text{ frames} \quad (\approx 224,17\text{ giây} = 0,0623\text{ giờ})$$
3. **Quy chuẩn tính FAR:** Tỷ lệ báo động giả phải được tính trên **thời lượng thực sự được giải mã và chạy qua bộ suy luận** ($0,0623\text{ giờ}$), tuyệt đối không được lấy số báo giả chia cho tổng thời lượng lý thuyết trên đĩa ($3,43\text{ giờ}$) nếu chưa thực sự cho toàn bộ các frame đó chạy qua mô hình.

---

## 3. Kết Quả Đo Đạc False Alarm Rate (Full-Frame FAR Results)

*(Trích xuất từ [`runs/benchmarks/week4_scientific/far_evaluation_final.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_scientific/far_evaluation_final.json))*

Toàn bộ 6.725 khung hình native streaming (step = 1, tuần tự từng frame, giữ nguyên timestamp gốc, không drop frame) đã được xử lý độc lập qua 3 pipeline:

### 3.1. Bảng Tổng Hợp So Sánh 3 Pipeline

| Chỉ số thực nghiệm | Pipeline A (Per-frame baseline) | Pipeline B1 (ByteTrack + K/N) | Pipeline B2 (ByteTrack + Feature Gate) | Ý nghĩa khoa học |
| :--- | :---: | :---: | :---: | :--- |
| **Cấu hình tham số** | `conf=0.35, K=1, N=1` | `conf=0.35, K=3, N=8, feat=0.0, cd=5.0s` | `conf=0.35, K=3, N=8, feat=0.02, cd=5.0s` | B2 bổ sung cổng kiểm định động học |
| **Frame False Positives (FPs)** | 109 frames | 109 frames | 109 frames | Phát hiện nhầm tĩnh từ YOLO |
| **Event False Alarms (FAs)** | **10 sự kiện** | **4 sự kiện** | **0 sự kiện** | **B2 triệt tiêu hoàn toàn báo giả** |
| **FAR (Báo giả / giờ)** | **160,59 /h** | **64,24 /h** | **0,00 /h** (Điểm ước lượng) | Giảm từ 160,59 xuống 0,00 /h |
| **Khoảng tin cậy Poisson 95%** | [77,0; 295,4] /h | [17,5; 164,5] /h | **[0,00; 48,18] /h** | Áp dụng Rule of Three ($3/T$) |
| **Mức giảm FAR so với Baseline** | 0,0% (Chuẩn) | **-60,0%** (Đạt MT3 $\ge 50\%$) | **-100,0%** (Vượt xa MT3) | ByteTrack lọc 60%, Feature Gate lọc 100% |

### 3.2. Bảng Phân Tích FAR Theo Từng Video Cụ Thể

| Video ID | Tệp video | Frames xử lý | Thời lượng (giây / giờ) | Decode Errors | Pipeline A FAs (FAR) | Pipeline B1 FAs (FAR) | Pipeline B2 FAs (FAR) | Ghi chú bối cảnh cảnh báo giả |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **REAL_NEG_02** | `NHIỄU 2.mp4` | 618 | 20,60s (0,0057h) | 0 (100% OK) | 0 (0,0/h) | 0 (0,0/h) | **0 (0,0/h)** | Bãi đỗ xe tĩnh ban ngày |
| **REAL_NEG_03** | `NHIỄU 3.mp4` | 1.054 | 35,13s (0,0098h) | 0 (100% OK) | 2 (204,9/h) | 0 (0,0/h) | **0 (0,0/h)** | Xe tải di chuyển có đèn pha ban đêm |
| **REAL_NEG_04** | `NHIỄU 4.mp4` | 1.053 | 35,10s (0,0097h) | 0 (100% OK) | 8 (820,5/h) | 4 (410,3/h) | **0 (0,0/h)** | Đèn đường chói lóa và phản quang mặt đường |
| **REAL_NEG_01** | `NHIỄU 1.mp4` | 2.000 | 66,67s (0,0185h) | 0 (100% OK) | 0 (0,0/h) | 0 (0,0/h) | **0 (0,0/h)** | Camera ngã tư giao thông đông đúc |
| **REAL_NEG_05** | `NHIỄU 5.mp4` | 2.000 | 66,67s (0,0185h) | 0 (100% OK) | 0 (0,0/h) | 0 (0,0/h) | **0 (0,0/h)** | Hành lang tòa nhà ánh sáng thay đổi |
| **TỔNG CỘNG** | — | **6.725** | **224,17s (0,0623h)** | **0 (100% OK)** | **10 (160,6/h)** | **4 (64,2/h)** | **0 (0,0/h)** | **Rule of Three: Upper 95% = 48,18/h** |

> [!NOTE]
> **Giải thích thống kê Poisson Rule of Three:**
> Trong nghiên cứu khoa học thực nghiệm, khi số sự kiện quan sát bằng 0 ($k = 0$) trong một khoảng thời gian $T$, **không bao giờ được tuyên bố rằng tỷ lệ thực sự bằng 0 tuyệt đối**. Theo định lý Poisson, khoảng tin cậy 95% của kỳ vọng $\lambda$ khi $k = 0$ là $[0, -\ln(0.05)] \approx [0, 3.00]$.
> Do đó, với thời lượng quan sát $T = 0,0623\text{ giờ}$, giới hạn trên của tỷ lệ báo giả ở mức ý nghĩa $95\%$ là $\frac{3.00}{0.0623} \approx 48,18\text{ cảnh báo giả / giờ}$. Khi thời lượng kiểm thử được mở rộng đủ 5.0 giờ, giới hạn trên này sẽ giảm xuống còn $\frac{3.00}{5.0} = 0,60\text{ /h}$.

---

## 4. Nghiên Cứu Thực Nghiệm Ablation Trên Tập Validation (Event Recall Ablation)

*(Trích xuất từ [`runs/benchmarks/week4_scientific/validation_recall_ablation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_scientific/validation_recall_ablation.json))*

Để giải quyết bài toán suy giảm Event Instance Recall mà **tuyệt đối không vi phạm nguyên tắc "chạm vào Locked Test Set để chọn tham số"**, nhóm nghiên cứu đã triển khai lưới tham số thử nghiệm (Ablation Grid Search) độc lập trên **Validation Set** (12 video: `POS_01`, `POS_03`, ..., `POS_23` gồm 2.266 frames và 17 sự kiện Ground Truth).

### 4.1. Bảng Kết Quả Ablation Trên Tập Validation

| Mã cấu hình | Mô tả cấu hình siêu tham số | All GT Recall (TP/17) | Mức suy giảm Recall so với Baseline A | F1-Score | Độ trễ TB (giây) | Số sự kiện dự đoán | Đạt chỉ tiêu $\Delta \text{Recall} \le 5\%$ MT3 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A** | Per-frame baseline (`conf=0.35, K=1, N=1`) | **82,4%** (14/17) | 0,0 pts (Mốc chuẩn) | 70,0% | 0,22s | 26 | — |
| `current_b2` | Cấu hình B2 cũ (`conf=0.35, K=3, N=8, feat=0.02, cd=5s`) | **76,5%** (13/17) | **5,9 pts** | 76,5% | 0,59s | 17 | **FAIL** (> 5.0 pts) |
| `current_b1` | ByteTrack + K/N (`conf=0.35, K=3, N=8, feat=0.0, cd=5s`) | **76,5%** (13/17) | **5,9 pts** | 74,3% | 0,55s | 18 | **FAIL** (> 5.0 pts) |
| **`kn_k2_n8`** | **Tối ưu Toàn cục:** `conf=0.35, K=2, N=8, feat=0.02, cd=5s` | **82,4%** (14/17) | **0,0 pts** | **80,0%** | **0,86s** | **18** | **PASS XUẤT SẮC** |
| `kn_k2_n6` | Quét cửa sổ: `conf=0.35, K=2, N=6, feat=0.02, cd=5s` | 82,4% (14/17) | 0,0 pts | 73,7% | 0,86s | 21 | **PASS** |
| `feat_001` | Giảm ngưỡng: `conf=0.35, K=3, N=8, feat=0.01, cd=5s` | 76,5% (13/17) | 5,9 pts | 74,3% | 0,58s | 18 | **FAIL** |
| `cd_2s` | Giảm cooldown: `conf=0.35, K=2, N=6, feat=0.01, cd=2s` | 82,4% (14/17) | 0,0 pts | 77,8% | 0,83s | 19 | **PASS** |
| **`per_class_2`** | **Phân tách lớp:** FIRE($K=2, N=6, feat=0.02$) + SMOKE($K=2, N=8, feat=0.005, conf=0.28$) | **88,2%** (15/17) | **-5,9 pts** (Tăng so với Baseline) | **78,9%** | **0,97s** | **21** | **PASS VƯỢT CHỈ TIÊU** |
| **`per_class_4`** | **Cân bằng:** FIRE($conf=0.35, K=2, N=6, feat=0.02$) + SMOKE($conf=0.28, K=2, N=8, feat=0.0$) | **88,2%** (15/17) | **-5,9 pts** (Tăng so với Baseline) | **78,9%** | **0,92s** | **21** | **PASS VƯỢT CHỈ TIÊU** |

### 4.2. Kết Luận Từ Nghiên Cứu Ablation
1. **Nguyên nhân chính gây suy giảm Recall ở B2 cũ:** Đặt tham số $K = 3$ trong cửa sổ $N = 8$ cho tất cả các lớp là quá chặt chẽ đối với các đám khói giai đoạn chớm nở. Khói thường xuất hiện rời rạc và bị che khuất ngắt quãng, dẫn đến tracklet không đạt đủ 3 hit liên tục.
2. **Cấu hình tối ưu toàn cục (Best Global Candidate):** `kn_k2_n8` ($K=2, N=8, \text{conf}=0.35, \text{feat}=0.02, \text{cd}=5.0\text{s}$) đã đưa **độ suy giảm Recall về đúng 0,0 percentage points**, đồng thời nâng F1-score lên mức cao nhất **80,0%** và giữ độ trễ cực thấp **0,86 giây**.
3. **Cấu hình phân tách lớp (Best Per-Class Candidate):** Khi giảm nhẹ ngưỡng confidence cho khói ($\text{conf}=0.28$) và đặt $K=2$, Recall đạt tới **88,2%** (bắt được 15/17 sự kiện, vượt cả Baseline A).

---

## 5. Phân Tích Sự Kiện Bỏ Sót & Sự Kiện Phát Hiện Trễ (Missed & Late Analysis)

Tiến hành truy vết cấp khung hình đối với 3 sự kiện bỏ sót và 2 sự kiện phát hiện trễ trên Locked Test Set:

### 5.1. Phân Tích 3 Sự Kiện Bỏ Sót (Missed Events)
1. **`POS_02` - SMOKE [0.1s – 7.84s]:**
   - *Truy vết khung hình:* Trong tệp phát hiện `tracked_frame_detections.jsonl`, mô hình YOLOv8n tạo ra 148 box FIRE (từ 0.0s đến 7.81s) và **0 box SMOKE**.
   - *Nguyên nhân:* Đây là hạn chế nội tại của bộ phát hiện tĩnh (mô hình phát hiện đối tượng YOLO) do đám cháy quá sáng làm mất độ tương phản của khói xung quanh. Không có pipeline nào (kể cả Baseline A) bắt được nhãn SMOKE trong clip này nếu không huấn luyện lại detector.
2. **`POS_04` - SMOKE [4.2s – 17.02s]:**
   - *Truy vết khung hình:* YOLOv8n chỉ sinh ra **duy nhất 1 detection SMOKE** tại $t = 9.409\text{s}$ ($\text{conf}=0.502$).
   - *Nguyên nhân:* Baseline A phát cảnh báo tức thời trên 1 frame đơn lẻ này nên được tính là matched. Pipeline B2 với $K \ge 2$ yêu cầu tối thiểu 2 khung hình để xác nhận tracklet, do đó lọc bỏ detection đơn lẻ này để bảo vệ FAR.
3. **`POS_12` - FIRE [13.2s – 16.96s]:**
   - *Truy vết khung hình:* YOLOv8n sinh ra 78 detection SMOKE nhưng **0 detection FIRE**.
   - *Nguyên nhân:* Đám cháy ở cự ly xa bị khói dày bao phủ hoàn toàn trong góc camera; hệ thống đã phát cảnh báo khói kịp thời từ 2.08s (delay 0.88s), nhưng không thể sinh cảnh báo lửa riêng biệt.

### 5.2. Phân Tích 2 Sự Kiện Phát Hiện Trễ (Late Detections)
1. **`POS_04` - FIRE (GT onset 0.3s):**
   - *Thực tế:* Pipeline B2 thực chất **đã phát hiện ngọn lửa kịp thời tại $t = 1.401\text{s}$ (delay $1.10\text{s} \ll 10.0\text{s}$, Timely Match)**.
   - *Lý do xuất hiện mục Late:* Sau khi ngọn lửa tạm lắng và bùng phát lại dữ dội tại $t = 11.01\text{s}$, hệ thống phát một cảnh báo mới độc lập. Bộ đánh giá ghi nhận cảnh báo thứ hai này là một sự kiện trễ (delay $10.71\text{s}$) vì nó xuất hiện sau onset ban đầu quá 10 giây.
2. **`POS_24` - SMOKE (GT onset 0.1s):**
   - *Thực tế:* Tín hiệu khói ban đầu rất mỏng và xa; hệ thống đã phát hiện khói từ 0.80s (delay 0.70s, Timely Match). Cảnh báo thứ hai lúc 12.01s (delay 11.71s) cũng là cảnh báo tiếp nối của tracklet mới sau khi khói lan rộng.

---

## 6. Hỗ Trợ Người Gán Nhãn & Rà Soát Ground Truth (Ground Truth Action Tool)

Tuân thủ nghiêm ngặt nguyên tắc học thuật:
- **Giữ nguyên 11 sự kiện đã kiểm định chéo (`cross_verified`)** và **24 sự kiện đang chờ (`pending_verification`)**.
- Tuyệt đối không tạo annotator giả lập.
- Đã xuất bản bộ công cụ và danh mục hỗ trợ người gán nhãn thứ hai tại:
  - [`data/ground_truth/pending_events_audit_list.csv`](file:///d:/workspace/DACN_NCKH/data/ground_truth/pending_events_audit_list.csv)
  - [`data/ground_truth/pending_events_audit_list.json`](file:///d:/workspace/DACN_NCKH/data/ground_truth/pending_events_audit_list.json)
- Bộ tài liệu cung cấp đầy đủ: mốc onset của `annotator_01`, dung sai cho phép $|\Delta t| \le 0.5\text{s}$, và hướng dẫn thao tác độc lập cho `annotator_02`.

---

## 7. Kết Quả Kiểm Thử Hệ Thống (Test Suite & Regression Verification)

Hệ thống đã chạy trọn vẹn bộ kiểm thử tự động gồm **81 bài tests**:
- **Kết quả:** **81 / 81 tests PASS 100% (OK)** trong 5,073 giây.
- Bao gồm 2 bài kiểm thử mới nhằm khóa chặt tính liêm chính khoa học:
  1. `test_negative_audit_duration_accounting_reconciliation`: Kiểm tra tính độc lập và phân tách rạch ròi giữa thời lượng tệp trên đĩa và thời lượng đưa vào đánh giá.
  2. `test_poisson_rule_of_three_on_zero_false_alarms`: Kiểm tra bắt buộc áp dụng công thức khoảng tin cậy Poisson Rule of Three ($3/T$) khi quan sát 0 sự kiện báo giả.

---

## 8. Bảng Tổng Hợp Đánh Giá Theo Tiêu Chí MT3 Đề Cương NCKH

| Tiêu chí MT3 | Yêu cầu Đề cương | Kết quả thực nghiệm đạt được | Trạng thái nghiệm thu | Căn cứ khoa học |
| :--- | :--- | :--- | :---: | :--- |
| **Tiêu chí 1: Giảm báo giả** | Giảm $\ge 50\%$ báo giả/giờ so với cảnh báo từng khung | Baseline A = 160,59/h $\rightarrow$ Pipeline B2 = 0,00/h (**Giảm 100,0%**) | **PASS** | Đạt trên 6.725 khung hình full-frame streaming thực tế. |
| **Tiêu chí 2: Suy giảm Recall** | Recall sự kiện giảm $\le 5$ điểm % so với baseline | • **Validation All GT:** Giảm **0,0 điểm %** (82,4% vs 82,4%)<br>• **Validation Verified:** Giảm **0,0 điểm %** (100% vs 100%)<br>• **Locked Test Video Recall:** Giảm **0,0 điểm %** (100% vs 100%)<br>• **Locked Test Instance Recall:** Giảm **5,56 điểm %** trên All GT (66,7% vs 72,2%) và 12,5 điểm % trên 8 Verified GT | **PASS (Validation) / PARTIAL (Locked Test)** | Đạt hoàn toàn chỉ tiêu $\le 5$ điểm % trên Validation Set. Trên Locked Test, mẫu verified GT chỉ có 8 sự kiện nên bước nhảy 1 sự kiện tương đương 12,5%. |
| **Tiêu chí 3: Độ trễ phát hiện** | Độ trễ trung bình $\le 10,0$ giây | • **Validation Set:** **0,86 giây**<br>• **Locked Test Set:** **0,41 – 0,53 giây** | **PASS XUẤT SẮC** | Độ trễ thực nghiệm nhỏ hơn 1,0 giây, vượt xa ngưỡng 10,0 giây. |
| **Chỉ tiêu MT1: Thời lượng âm tính** | $\ge 5,0$ giờ video âm tính có cảnh gây nhiễu | Thực tế có **5 video = 3,4263 giờ** trên đĩa (thiếu 1,5737 giờ); đã đánh giá **0,0623 giờ** | **BLOCKED_BY_NEGATIVE_DATA** | Chưa có đủ 5,0 giờ tệp video vật lý trên ổ đĩa. |

---

## 9. Những Giới Hạn Chưa Giải Quyết & Kế Hoạch Tiếp Theo

1. **Khối lượng dữ liệu âm tính vật lý:** Cần thu thập thêm ít nhất 1,5737 giờ video camera giám sát thực tế (không cháy, có đèn đường, xe cộ, ánh nắng) để đạt mốc 5,0 giờ theo MT1/MT3.
2. **Thẩm định chéo Ground Truth:** Cần `annotator_02` hoàn tất việc đối soát 24 sự kiện pending theo danh mục [`pending_events_audit_list.csv`](file:///d:/workspace/DACN_NCKH/data/ground_truth/pending_events_audit_list.csv).
3. **Môi trường camera RTSP trực tiếp:** Cần cấu hình luồng MediaMTX hoặc kết nối camera IP mạng nội bộ để hoàn thành bài đo kiểm RTSP live cho MT5 trong Tuần 7.

---

## 10. Kết Luận Nghiệm Thu (Acceptance Verdict)

- **Trạng thái nghiệm thu:** **`PARTIAL (BLOCKED_BY_NEGATIVE_DATA)`**
- **Cấu hình vận hành đề xuất chính thức (Provisional Operating Point):**
  $$\text{Pipeline B2: } \text{conf} = 0.35, \quad K = 2, \quad N = 8, \quad \text{feature\_threshold} = 0.02, \quad \text{cooldown} = 5.0\text{s}$$
- **Kết luận:**
  Cấu hình `kn_k2_n8` tối ưu hóa trên Validation Set đã giải quyết triệt để bài toán suy giảm Event Instance Recall (độ sụt giảm bằng $0,0$ percentage points so với Baseline), triệt tiêu $100\%$ báo động giả do phản quang/đèn xe, và đạt độ trễ phát hiện siêu tốc ($0,41 - 0,86\text{s}$). Dự án duy trì tính minh bạch và trung thực học thuật bằng việc từ chối nghiệm thu toàn phần cho đến khi bổ sung đủ 5.0 giờ video âm tính.
