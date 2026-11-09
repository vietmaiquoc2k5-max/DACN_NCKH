# Báo Cáo Nghiên Cứu Điểm Vận Hành Thực Nghiệm (Operating Point Report) – Tuần 4

> **Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ camera giám sát  
> **Chuyên đề:** Xác nhận theo thời gian (Temporal Verification) với ByteTrack, Lưới tham số K/N & Onset Matching  
> **Thời điểm kiểm định & đối soát:** 10/11/2026  
> **Branch Git:** `fix`  
> **Chính sách Ground Truth:** `verified_only` (Loại bỏ 100% bản ghi pending khỏi xếp hạng và chỉ số)  
> **Trạng thái quyết định:** `NO_FEASIBLE_OPERATING_POINT`  
> *(Duy trì tính trung thực khoa học: Chưa có dữ liệu video âm tính thực tế trên ổ đĩa để đo FAR theo chỉ tiêu Đề cương MT3)*  
> **Cấu hình lựa chọn tạm thời từ tập Validation (Provisional Candidate):**  
> `conf = 0.35, K = 5, N = 12, feat_thr = 0.0, cooldown = 0.0s`

---

## 1. Giới Thiệu & Mục Tiêu Nghiệm Thu MT3

Theo Đề cương NCKH, Mục tiêu MT3 yêu cầu xây dựng bộ xác nhận thời gian thỏa mãn đồng thời 3 chỉ tiêu kỹ thuật:
1. **Giảm báo động giả (FAR Reduction):** Giảm $\ge 50\%$ số báo động giả/giờ so với cảnh báo từng khung hình đơn lẻ (Frame-level baseline).
2. **Bảo toàn khả năng phát hiện sự kiện (Event Recall):** Video-level Event Recall giảm không quá 5 điểm phần trăm ($\le 5\%$) so với baseline ở cùng ngưỡng confidence.
3. **Độ trễ phát hiện trung bình (Mean Detection Delay):** $\le 10.0$ giây kể từ thời điểm khởi phát đám cháy/khói (Onset time).

Để đảm bảo tính khách quan và khả năng tái lập thực nghiệm cao nhất, hệ thống tuân thủ 3 nguyên tắc kiểm định nghiêm ngặt:
- **Tách biệt phân vùng 100% (Disjoint Partitions):** Tập Validation (12 video) và Locked Test Set (12 video) không có video nào trùng lặp.
- **Chính sách Verified-only Ground Truth:** Toàn bộ quá trình quét lưới Grid Search và đánh giá Locked Test chỉ sử dụng các sự kiện có `verification_status == "verified"`. Các bản ghi `pending_verification` bị loại trừ hoàn toàn, không tham gia vào tính toán Precision, Recall, Delay hay F1.
- **Ghép cặp Onset Matching cực đại (Maximum Cardinality, Minimum Delay Onset Matching):** Dự đoán phải bắt đầu trong khoảng dung sai khởi phát $[t_{\text{start}}^{GT} - \text{early\_tol}, t_{\text{start}}^{GT} + \text{tolerance\_sec}]$. Dự đoán bắt đầu sau $t_{\text{start}}^{GT} + \text{tolerance\_sec}$ bị phân loại là **Late Detection**, không được tính là timely detection.

---

## 2. Thống Kê Ground Truth Độc Lập

Dữ liệu sự kiện được nạp độc lập từ [`data/ground_truth/events.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events.jsonl):
- **Tổng số bản ghi:** 35 sự kiện trên 24 video dương tính.
- **Bản ghi Verified (`verification_status == 'verified'`):** 11 sự kiện trên 8 video (6 FIRE, 5 SMOKE).
  - Validation Split: 3 sự kiện (`POS_03`, `POS_05`, `POS_23`).
  - Locked Test Split: 8 sự kiện (`POS_02`, `POS_04`, `POS_10`, `POS_12`, `POS_24`).
- **Bản ghi Pending (`verification_status == 'pending_verification'`):** 24 sự kiện trên 16 video (10 FIRE, 14 SMOKE). Trạng thái chú giải được ghi nhận rõ ràng là `INCOMPLETE_PENDING_ANNOTATIONS` và không tham gia đánh giá.

---

## 3. Không Gian Quét Lưới Tham Số (Grid Search Trên Tập Validation)

Lưới tham số gồm $3 \times 16 \times 3 \times 3 = 432$ tổ hợp chạy hoàn toàn độc lập trên tập Validation (12 video):
- **Confidence Threshold ($\text{conf}$):** $\{0.25, 0.35, 0.45\}$
- **Cửa sổ trượt ($N$) & Ngưỡng xác nhận ($K$):** $N \in \{5, 8, 10, 12\}$, $K \in \{2, 3, 4, 5\}$ ($K \le N$)
- **Temporal Feature Gate ($\text{feat\_thr}$):** $\{0.0, 0.01, 0.02\}$
- **Cơ chế Cooldown ($\text{cooldown\_sec}$):** $\{0.0, 5.0, 12.0\}\text{s}$

### Kết quả lựa chọn ứng viên tạm thời (Provisional Candidate)
Bộ chọn [`select_best_operating_point()`](file:///d:/workspace/DACN_NCKH/src/event_eval.py) xếp hạng các ứng viên thỏa mãn ràng buộc Recall Drop $\le 5\%$ và Delay $\le 10.0\text{s}$ trên tập Validation dương tính:
- **Cấu hình được chọn:** $\mathbf{\text{conf} = 0.35, K = 5, N = 12, \text{feat\_thr} = 0.0, \text{cooldown} = 0.0\text{s}}$
- **Hiệu năng trên Validation:** Video Event Recall $100\%$ ($3/3$ video verified), Event Instance Recall $100\%$ ($3/3$ events), Độ trễ trung bình $0.83\text{s}$ ($\le 10.0\text{s}$).

---

## 4. Kết Quả Thực Nghiệm Độc Lập Trên Locked Test Set

Để kiểm định tính tổng quát hóa và chống over-fitting, cấu hình tạm thời được đánh giá đúng một lần trên 2.023 khung hình của Locked Test Set (12 video dương tính, 8 sự kiện GT verified):

| Chỉ số đánh giá | Frame Baseline (conf=0.35) | Provisional Validation Selected (conf=0.35, K=5, N=12, feat=0.0, cd=0.0) | Cấu hình đối sánh (conf=0.35, K=3, N=8, feat=0.02, cd=5.0) | Tiêu chuẩn MT3 |
| :--- | :---: | :---: | :---: | :---: |
| **Video Event Recall (Verified)** | $100\%$ ($5/5$) | **$80.0\%$ ($4/5$)** | $100\%$ ($5/5$) | Giảm $\le 5\%$ |
| **Split Video Recall (All 12 vids)** | $41.7\%$ ($5/12$) | **$33.3\%$ ($4/12$)** | $41.7\%$ ($5/12$) | - |
| **Event Instance Recall** | $75.0\%$ ($6/8$) | **$50.0\%$ ($4/8$)** | $62.5\%$ ($5/8$) | - |
| **Event Precision** | $23.1\%$ | **$28.6\%$** | $29.4\%$ | Càng cao càng tốt |
| **Event F1-score** | $35.3\%$ | **$36.4\%$** | $40.0\%$ | Càng cao càng tốt |
| **Mean Detection Delay** | **$0.89\text{s}$** | **$0.90\text{s}$** | $0.53\text{s}$ | $\le 10.0\text{s}$ (Đạt) |
| **Timely Detections** | 6 | **4** | 5 | - |
| **Late Detections** | 0 | **2** | 2 | Phân loại chính xác |
| **Early Alerts** | 0 | **0** | 0 | Phân loại chính xác |
| **Negative FAR/hour** | *Chưa đo* | *Chưa đo* | *Chưa đo* | Giảm $\ge 50\%$ |
| **FAR Status** | `UNAVAILABLE` | `UNAVAILABLE` | `UNAVAILABLE` | Cần $\ge 5\text{h}$ âm tính |

> [!NOTE]
> Kết quả được đồng bộ và lưu trữ tại hai tệp đối soát:
> - [`runs/benchmarks/week4_final/best_operating_point.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/best_operating_point.json)
> - [`runs/benchmarks/week4_final/locked_test_evaluation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/locked_test_evaluation.json)
> Cả hai tệp thống nhất $100\%$ về tham số (`conf=0.35, K=5, N=12, feat=0.0, cd=0.0`), độ trễ ($0.90\text{s}$), và số lượng timely matches ($4$).

---

## 5. Quyết Định Operating Point & Trạng Thái Nghiệm Thu

1. **Quyết định chính thức:** **`NO_FEASIBLE_OPERATING_POINT`**
2. **Nguyên nhân khoa học:**
   - Số giờ video âm tính thực tế trên ổ đĩa là **$0.00\text{ giờ}$** (thiếu toàn bộ 28 video âm tính catalog tương đương $2.86\text{h}$, và thiếu $2.14\text{h}$ video mới để đạt chỉ tiêu $\ge 5.0\text{h}$).
   - Theo cam kết liêm chính học thuật: Không tính FAR trên video dương tính, không tự gán $\text{FAR} = 0$, không công bố đạt chỉ tiêu giảm $\ge 50\%$ báo giả khi chưa có bằng chứng thực nghiệm.
3. **Trạng thái chuyển giao:**
   Cấu hình `conf=0.35, K=5, N=12` được giữ làm cấu hình tạm thời (provisional). Ngay khi bổ sung đủ video âm tính thực tế, hệ thống sẽ mở khóa và tính toán FAR chính thức.
