# Báo Cáo Nghiên Cứu Điểm Vận Hành Thực Nghiệm (Operating Point Report) – Tuần 4

> **Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ camera giám sát  
> **Chuyên đề:** Xác nhận theo thời gian (Temporal Verification) với ByteTrack, Lưới tham số K/N & Onset Matching  
> **Thời điểm kiểm định & đối soát:** 10/11/2026  
> **Branch Git:** `fix`  
> **Chính sách Ground Truth:** `verified_only` (100% bản ghi 35/35 events được đối soát thực tế trên video vật lý)  
> **Trạng thái quyết định:** `PARTIAL / BLOCKED_BY_NEGATIVE_DATA`  
> *(Thuật toán đạt toàn bộ chỉ tiêu kỹ thuật MT3; Trạng thái bảo lưu do cần bổ sung thêm 1.57h video âm tính để đạt đủ mốc >= 5.0h theo Đề cương NCKH)*  
> **Cấu hình vận hành khuyến nghị (Recommended Operating Point):**  
> `conf = 0.35, K = 3, N = 8, feat_thr = 0.02, cooldown = 5.0s`

---

## 1. Giới Thiệu & Mục Tiêu Nghiệm Thu MT3

Theo Đề cương NCKH, Mục tiêu MT3 yêu cầu xây dựng bộ xác nhận thời gian thỏa mãn đồng thời 4 tiêu chuẩn:
1. **Giảm báo động giả (FAR Reduction):** Giảm $\ge 50\%$ số báo động giả/giờ so với cảnh báo từng khung hình đơn lẻ (Frame-level baseline).
2. **Bảo toàn khả năng phát hiện sự kiện (Event Recall):** Video-level Event Recall giảm không quá 5 điểm phần trăm ($\le 5\%$) so với baseline ở cùng ngưỡng confidence.
3. **Độ trễ phát hiện trung bình (Mean Detection Delay):** $\le 10.0$ giây kể từ thời điểm khởi phát đám cháy/khói (Onset time).
4. **Quy mô tập dữ liệu âm tính:** Tối thiểu $\ge 5.0$ giờ video không cháy có chứa cảnh nhiễu thực tế (đèn xe, ánh nắng, sương mù, chuyển động).

Hệ thống tuân thủ các nguyên tắc khoa học nghiêm ngặt:
- **Tách biệt phân vùng 100% (Disjoint Partitions):** Tập Validation (12 video dương tính + 3 video âm tính = 1.88h) và Locked Test Set (12 video dương tính + 2 video âm tính = 1.54h) độc lập tuyệt đối.
- **Toàn vẹn Ground Truth:** Toàn bộ 35 sự kiện trên 24 video dương tính đã được đối soát trực tiếp trên file video vật lý (`FireSmokeLabeler/videos/*.avi`), đóng băng snapshot [`data/ground_truth/events_v1.0_snapshot.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events_v1.0_snapshot.jsonl) và mã băm SHA256 [`data/ground_truth/checksums.sha256`](file:///d:/workspace/DACN_NCKH/data/ground_truth/checksums.sha256).
- **Ghép cặp Onset Matching cực đại (Maximum Cardinality, Minimum Delay Onset Matching):** Dự đoán phải bắt đầu trong khoảng dung sai khởi phát $[t_{\text{start}}^{GT} - \text{early\_tol}, t_{\text{start}}^{GT} + \text{tolerance\_sec}]$. Dự đoán sau $t_{\text{start}}^{GT} + \text{tolerance\_sec}$ bị coi là **Late Detection**.

---

## 2. Thống Kê Dữ Liệu Đánh Giá Thực Nghiệm

1. **Dữ liệu Dương tính (24 video vật lý):**
   - 35 sự kiện Ground Truth (16 FIRE, 19 SMOKE | 100% `verified`).
   - Split Validation: 12 video (`POS_01`, `03`, `05`, `07`, `09`, `11`, `13`, `15`, `17`, `19`, `21`, `23`) với 17 sự kiện GT.
   - Split Locked Test: 12 video (`POS_02`, `04`, `06`, `08`, `10`, `12`, `14`, `16`, `18`, `20`, `22`, `24`) với 18 sự kiện GT.

2. **Dữ liệu Âm tính Thực Tế (5 video trên ổ đĩa):**
   - Lưu trữ tại [`datasets/negative_videos/`](file:///d:/workspace/DACN_NCKH/datasets/negative_videos/) và danh mục [`data/negative_videos_manifest.json`](file:///d:/workspace/DACN_NCKH/data/negative_videos_manifest.json):
     - `REAL_NEG_01` (NHIỄU 1.mp4): 6,727.87s = 1.8689h (Val split)
     - `REAL_NEG_02` (NHIỄU 2.mp4): 20.60s = 0.0057h (Val split)
     - `REAL_NEG_03` (NHIỄU 3.mp4): 35.13s = 0.0098h (Val split)
     - `REAL_NEG_04` (NHIỄU 4.mp4): 35.10s = 0.0098h (Test split)
     - `REAL_NEG_05` (NHIỄU 5.mp4): 5,516.00s = 1.5322h (Test split)
   - **Tổng thời lượng thực tế trên ổ đĩa:** **12,334.70 giây = 3.4263 giờ** (Val: 1.8843h, Test: 1.5420h).
   - **Thời lượng còn thiếu so với mốc $\ge 5.0\text{h}$:** **1.5737 giờ** (1 giờ 34 phút 25 giây).

---

## 3. Kết Quả Thực Nghiệm Độc Lập Trên Locked Test Set

Đánh giá thực nghiệm chạy trên toàn bộ 4.852 khung hình của Locked Test Set (12 video dương tính và 1.5420 giờ video âm tính):

| Chỉ số đánh giá | Pipeline A (Frame Baseline conf=0.35) | Pipeline B1 (ByteTrack K/N conf=0.35, K=5, N=12) | Pipeline B2 (ByteTrack + Feature Gate conf=0.35, K=3, N=8, feat=0.02, cd=5.0s) | Tiêu Chuẩn MT3 Đề Cương NCKH | Đánh Giá Kết Quả |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Event Recall** | **91.7%** (11/12 video) | 75.0% (9/12 video) | **91.7%** (11/12 video) | Giảm $\le 5.0\%$ so với Baseline | **✅ ĐẠT (Giảm 0.0%)** |
| **Event Instance Recall** | 72.2% (13/18 events) | 55.6% (10/18 events) | **66.7%** (12/18 events) | Càng cao càng tốt | **Tốt** |
| **Event Precision** | 19.1% | 50.0% | **50.0%** | Nâng cao độ chính xác cảnh báo | **Tăng từ 19.1% lên 50.0%** |
| **Event F1-Score** | 30.2% | 52.6% | **57.1%** | Càng cao càng tốt | **Tăng +26.9 điểm %** |
| **False Alarm Rate (FAR/giờ)** | **27.24 báo giả/h** | **3.89 báo giả/h** | **4.54 báo giả/h** | Giảm $\ge 50\%$ so với Baseline | **✅ ĐẠT (Giảm 83.33%)** |
| **Mức giảm báo giả (FAR Reduction)** | *Baseline (0%)* | Giảm 85.72% | **Giảm 83.33%** | $\ge 50.0\%$ | **✅ VƯỢT XA CHỈ TIÊU** |
| **Độ trễ trung bình (Mean Delay)** | **0.53 giây** | 0.86 giây | **0.55 giây** | $\le 10.0$ giây | **✅ ĐẠT (0.55s <= 10.0s)** |
| **Thời lượng âm tính kiểm thử** | 1.542 giờ (Test split) | 1.542 giờ (Test split) | 1.542 giờ (Test split) | $\ge 5.0$ giờ toàn hệ thống | **Chưa đạt (3.43h / 5.0h)** |

---

## 4. Phân Tích Chuyên Môn Khoa Học

1. **Hiệu quả vượt trội của Temporal Verifier kết hợp Feature Gate (Pipeline B2):**
   - Khi áp dụng cảnh báo từng khung đơn lẻ (Pipeline A), hệ thống bị nhiễu nghiêm trọng với **27.24 cảnh báo giả mỗi giờ** do ánh sáng đèn xe và các đốm sáng nhấp nháy.
   - Nhờ ByteTrack duy trì track ID liên tục kết hợp kiểm tra biến động chớp nháy/lan tỏa diện tích (`TemporalFeatureExtractor`), số báo giả giảm mạnh xuống chỉ còn **4.54 báo giả/giờ** (giảm tới **83.33%**, vượt xa chỉ tiêu 50%).
   - Precision sự kiện tăng vọt từ $19.1\%$ lên **$50.0\%$**, trong khi Video Event Recall giữ vững ở mức **$91.7\%$** (không bị suy giảm bất kỳ điểm phần trăm nào).

2. **Độ trễ phản ứng thời gian thực (Mean Delay = 0.55s):**
   - Bộ xác nhận đưa ra quyết định cảnh báo sau $0.55$ giây kể từ khi đám cháy bùng phát, hoàn toàn đáp ứng yêu cầu khẩn cấp trong PCCC và vượt xa giới hạn cho phép $\le 10$ giây.

---

## 5. Quyết Định Operating Point & Kết Luận Nghiệm Thu

1. **Điểm vận hành chính thức khuyến nghị:**
   - **Tham số:** `Confidence = 0.35, Confirm K = 3, Window N = 8, Feature Threshold = 0.02, Cooldown = 5.0s`.
   - **Tệp lưu trữ:** [`runs/benchmarks/week4_final/reconciled_test_evaluation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/reconciled_test_evaluation.json).
2. **Trạng thái nghiệm thu:** **`PARTIAL / BLOCKED_BY_NEGATIVE_DATA`**
   - **Lý do khoa học:** Toàn bộ thuật toán và tiêu chí phát hiện đã được chứng minh bằng thực nghiệm độc lập. Tuy nhiên, để tuân thủ liêm chính khoa học của đề cương NCKH (chỉ tiêu $\ge 5.0$ giờ video âm tính), việc nghiệm thu toàn phần cần đợi người dùng bổ sung thêm 1.5737 giờ video âm tính vật lý vào thư mục dữ liệu.
