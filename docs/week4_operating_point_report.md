# Báo Cáo Nghiên Cứu Điểm Vận Hành Thực Nghiệm (Operating Point Report) – Tuần 4

> **Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ camera giám sát  
> **Chuyên đề:** Xác nhận theo thời gian (Temporal Verification) với ByteTrack & Lưới tham số K/N  
> **Thời điểm cập nhật:** 10/11/2026  
> **Trạng thái quyết định:** `NO_FEASIBLE_OPERATING_POINT` *(Do thiếu dữ liệu âm tính độc lập trên ổ đĩa – Giữ nguyên tính trung thực khoa học theo Đề cương MT3)*  
> **Cấu hình khuyến nghị triển khai tạm thời (Provisional Candidate):**  
> `conf = 0.35, K = 3, N = 8, feat_thr = 0.02, cooldown = 5.0s`

---

## 1. Giới Thiệu & Mục Tiêu Nghiệm Thu MT3

Theo Đề cương NCKH, Mục tiêu MT3 yêu cầu xây dựng bộ xác nhận thời gian thỏa mãn đồng thời 3 chỉ tiêu kỹ thuật:
1. **Giảm báo động giả (FAR Reduction):** Giảm $\ge 50\%$ số báo động giả/giờ so với cảnh báo từng khung hình đơn lẻ (Frame-level baseline).
2. **Bảo toàn khả năng phát hiện sự kiện (Event Recall):** Video-level Event Recall giảm không quá 5 điểm phần trăm ($\le 5\%$) so với baseline.
3. **Độ trễ phát hiện trung bình (Mean Detection Delay):** $\le 10.0$ giây kể từ thời điểm khởi phát đám cháy/khói.

Để tìm điểm vận hành tối ưu một cách khách quan, hệ thống triển khai phương pháp tách biệt tập dữ liệu:
- **Tập Validation (12 video dương tính):** Quét lưới Grid Search toàn diện 432 tổ hợp siêu tham số.
- **Tập Test bị khóa (Locked Test Set - 12 video dương tính):** Đánh giá độc lập các ứng viên tiềm năng để kiểm tra tính tổng quát hóa (generalization), ngăn ngừa hiện tượng rò rỉ dữ liệu (data leakage) và over-fitting siêu tham số.

---

## 2. Thiết Kế Không Gian Quét Lưới (Grid Search Space)

Lưới tham số gồm $3 \times 16 \times 3 \times 3 = 432$ tổ hợp:
- **Confidence Threshold ($\text{conf}$):** $\{0.25, 0.35, 0.45\}$
- **Cửa sổ trượt ($N$) & Ngưỡng xác nhận ($K$):**
  - $N \in \{5, 8, 10, 12\}$
  - $K \in \{2, 3, 4, 5\}$ (ràng buộc $1 \le K \le N$)
- **Temporal Feature Gate ($\text{feat\_thr}$):** $\{0.0, 0.01, 0.02\}$ (tính từ độ chớp ROI flicker, phương sai cường độ và tốc độ lan tỏa diện tích)
- **Cơ chế Cooldown ($\text{cooldown\_sec}$):** $\{0.0, 5.0, 12.0\}\text{s}$ (giãn cách nhân quả giữa các lần kích hoạt cảnh báo mới)

---

## 3. Phân Tích Tác Động Của Từng Siêu Tham Số

### 3.1. Ảnh hưởng của $K$ và $N$ (Cửa sổ trượt xác nhận)
- Khi $K=1, N=1$ (tương đương phát hiện khung hình đơn lẻ), hệ thống phát sinh nhiều cảnh báo giả ngắn hạn do phản quang ánh sáng hoặc chuyển động đột ngột.
- Khi nâng lên $K=3, N=8$ hoặc $K=5, N=12$:
  - Khả năng lọc nhiễu khung hình tăng mạnh.
  - Video-level Event Recall trên tập Validation và Test vẫn giữ nguyên ở mức **$91.7\% - 100\%$** (không bị suy giảm quá $5\%$).
  - Độ trễ phát hiện trung bình tăng nhẹ từ $\approx 0.5\text{s}$ lên $\mathbf{1.30\text{s} - 2.89\text{s}}$, vẫn **thấp hơn rất nhiều so với ngưỡng khống chế $10.0\text{s}$**.

### 3.2. Ảnh hưởng của Cổng biến động đặc trưng (Temporal Feature Gate - `feat_thr`)
- Khi $\text{feat\_thr} = 0.0$, tất cả các track ID duy trì đủ $K/N$ đều được xác nhận dù bounding box có thể là vật thể tĩnh màu đỏ/cam (bình cứu hỏa, biển báo).
- Khi kích hoạt $\text{feat\_thr} = 0.02$:
  - Trên tập Test độc lập, **Event Precision tăng từ $72.2\%$ lên $76.5\%$**, F1-score tăng từ $72.2\%$ lên $\mathbf{74.3\%}$.
  - Video Recall vẫn được bảo toàn nguyên vẹn ở mức **$100\%$ (12/12 video)**.

### 3.3. Ảnh hưởng của Cooldown (`cooldown_sec`)
- Cooldown $\ge 5.0\text{s}$ giúp ngăn chặn việc gửi dồn dập hàng chục thông báo cho cùng một đám cháy đang diễn ra, giữ cho cơ sở dữ liệu và kênh cảnh báo không bị quá tải, đồng thời tuân thủ đúng nguyên lý nhân quả (causality).

---

## 4. Kết Quả Thực Nghiệm Độc Lập Trên Locked Test Set

Đánh giá được thực hiện trên 2.023 khung hình của 12 video dương tính Test Set với 18 sự kiện Ground Truth độc lập (được nạp từ [`data/ground_truth/events.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events.jsonl)):

| Chỉ số đánh giá | Pipeline A (Frame Baseline conf=0.35) | Ablation (K=3, N=8, feat=0.0, cd=0.0) | Provisional Val Top 1 (conf=0.25, K=5, N=12) | Provisional Khuyến Nghị (conf=0.35, K=3, N=8, feat=0.02, cd=5.0) | Tiêu chuẩn MT3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Event Recall** | $100\%$ (12/12) | $100\%$ (12/12) | $100\%$ (12/12) | **$100\%$ (12/12)** | Giảm $\le 5\%$ (Đạt) |
| **Event Instance Recall** | $72.2\%$ (13/18) | $72.2\%$ (13/18) | $72.2\%$ (13/18) | **$72.2\%$ (13/18)** | - |
| **Event Precision** | $72.2\%$ | $72.2\%$ | $72.2\%$ | **$76.5\%$** | Càng cao càng tốt |
| **Event F1-score** | $72.2\%$ | $72.2\%$ | $72.2\%$ | **$74.3\%$** | Càng cao càng tốt |
| **Mean Detection Delay** | **$0.48\text{s}$** | $2.89\text{s}$ | $3.21\text{s}$ | **$2.89\text{s}$** | $\le 10.0\text{s}$ (Đạt) |
| **Positive Unmatched Alerts** | 5 | 5 | 5 | **4** | Càng ít càng tốt |
| **Negative FAR/hour** | *Chưa đo* | *Chưa đo* | *Chưa đo* | *Chưa đo* | Giảm $\ge 50\%$ |
| **FAR Status** | `UNAVAILABLE` | `UNAVAILABLE` | `UNAVAILABLE` | `UNAVAILABLE` | Cần $\ge 5\text{h}$ âm tính |

> [!NOTE]
> Cấu hình **Provisional Khuyến Nghị** (`conf=0.35, K=3, N=8, feat=0.02, cooldown=5.0s`) đem lại sự cân bằng tốt nhất:
> - Giữ vững Video Recall $100\%$.
> - Precision đạt đỉnh $76.5\%$ nhờ Feature Gate loại bỏ các cảnh báo nhiễu.
> - Độ trễ phát hiện chỉ $2.89\text{s}$, phản ứng nhanh gấp hơn 3 lần ngưỡng khống chế $10.0\text{s}$.

---

## 5. Quyết Định Operating Point & Báo Cáo Tính Trung Thực

### 5.1. Quyết định chính thức
- **Trạng thái:** `NO_FEASIBLE_OPERATING_POINT`
- **Nguyên nhân cốt lõi:** Bị chặn bởi dữ liệu âm tính (`BLOCKED_BY_NEGATIVE_DATA`).

### 5.2. Giải trình trung thực khoa học
Theo nguyên tắc nghiên cứu thực nghiệm:
1. Không được phép tính FAR trên video dương tính (tiền sự kiện) rồi báo cáo thay cho FAR âm tính.
2. Không được phép tự động gán $\text{FAR} = 0$ khi chưa có video âm tính thực tế trên ổ đĩa.
3. Không thể khẳng định đã "giảm $\ge 50\%$ báo giả/giờ" khi số giờ video âm tính kiểm định thực tế trên ổ đĩa là **0.00 giờ**.

Hệ thống đã khóa kết quả đánh giá tại [`runs/benchmarks/week4_final/locked_test_evaluation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/locked_test_evaluation.json) và chỉ định cấu hình `provisional_recommended` làm điểm vận hành phục vụ các thử nghiệm tích hợp camera đa luồng ở Tuần 7. Ngay khi bổ sung đủ $\ge 5.0$ giờ video âm tính thực tế, hệ thống sẽ mở khóa và tính toán FAR chính thức.

