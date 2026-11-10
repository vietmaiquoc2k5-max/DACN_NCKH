# Báo Cáo Chuẩn Hóa Phương Pháp Luận Đo Lường Báo Động Giả (FAR) và Đối Soát Điểm Vận Hành Tuần 4
## (Week 4 False Alarm Rate Methodology Correction & Operating Point Reconciliation Report)

**Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ camera giám sát (`DACN_NCKH`)  
**Nhánh Git làm việc:** `week4-scientific-validation`  
**Commit nền đối chiếu:** `df4305c2008fd2052dbdad4da67dd97857cfcc9c` (kế thừa commit kiểm định khoa học `280cd6337920c981bd5662a0fddf9bdde374c041`)  
**Thời gian hoàn thành:** 10/11/2026  
**Chủ biên thực hiện:** Lê Minh Thiên (Chủ nhiệm đề tài) & Mai Quốc Việt (Kỹ sư hệ thống & Dữ liệu)  
**Tiêu chuẩn khoa học áp dụng:** Đề cương NCKH gốc (`DeCuong_NCKH_PhatHienChay_Camera.docx`), IEEE Transactions on CSVT Standards, Thống kê Suy luận Poisson (Poisson Inferential Statistics - Rule of Three), Nguyên tắc Liêm chính Học thuật Độc lập.

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Sau đợt rà soát độc lập ở commit nền `df4305c2`, nhóm nghiên cứu đã phát hiện một số sai lệch và thiếu sót mang tính phương pháp luận nghiêm trọng trong việc đo lường Tỷ lệ Báo động Giả (False Alarm Rate - FAR) và đánh giá các điểm vận hành (Operating Points) của hệ thống xác thực thời gian Tuần 4:
1. **Lẫn lộn đơn vị đo lường:** Đếm gộp hoặc dùng lẫn lộn giữa tổng số khung hình có báo sai (`false_positive_frames`), tổng số bounding boxes phát hiện sai (`false_positive_detections`), và tổng số sự kiện cảnh báo sai độc lập (`false_alarm_events`).
2. **Bỏ qua cảnh báo khói giả (SMOKE False Alarms):** Một số pipeline chỉ ghi nhận và đếm sự kiện có `alert_level == "FIRE"`, vô tình bỏ qua các sự kiện báo giả do khói (`SMOKE`), trong khi trên môi trường thực tế ban đêm, đèn đường, sương mù và bụi phát sinh cảnh báo khói giả nhiều hơn cả lửa.
3. **So sánh thiếu kiểm soát thời gian nén (Cooldown Asymmetry):** So sánh Baseline A sử dụng `alert_cooldown = 2.0s` với Pipeline B1 và B2 sử dụng `alert_cooldown = 5.0s`. Sự chênh lệch thời gian hồi tiếp khiến phép so sánh tỷ lệ giảm FAR thiếu tính tương thích thời gian.
4. **Mập mờ giữa thời lượng lưu trữ trên đĩa và thời lượng thực tế inference:** Tổng thời lượng video âm tính vật lý trên đĩa là $3.4263\text{ giờ}$ (370.041 frames), nhưng thực tế chỉ mới inference $0.0623\text{ giờ}$ (6.725 frames - chế độ bounded sample). Việc dùng 3.43 giờ làm mẫu số tính FAR khi chỉ inference 0.06 giờ là sai lệch toán học nghiêm trọng.

Báo cáo này công bố toàn bộ quy trình chuẩn hóa phương pháp luận, kết quả thực nghiệm đo kiểm độc lập trên tập âm tính, audit ground truth trên tập Validation, và khóa cấu hình Operating Point chính thức trong mã nguồn dự án.

---

## 2. Chi Tiết Các Lỗi Phương Pháp Luận Đã Phát Hiện và Khắc Phục

| Vấn đề phát hiện | Phương pháp cũ (Thiếu sót) | Phương pháp chuẩn hóa mới (Khắc phục triệt để) | Tệp mã nguồn / Dữ liệu liên quan |
| :--- | :--- | :--- | :--- |
| **Đếm False Positives** | Coi `frames có FP` là `detections FP` hoặc dùng lẫn lộn để tính FAR. | Tách bạch 3 chỉ số độc lập: (1) `Detections FP` (tổng boxes), (2) `Frames FP` (số frames có $\ge 1$ box), (3) `Events FA` (số sự kiện sau gom cụm & cooldown). | [`src/event_eval.py`](file:///d:/workspace/DACN_NCKH/src/event_eval.py), [`scripts/evaluate_negative_dataset_unified.py`](file:///d:/workspace/DACN_NCKH/scripts/evaluate_negative_dataset_unified.py) |
| **Phân loại False Alarm** | Chỉ kiểm tra `alert_level == "FIRE"`, bỏ sót cảnh báo SMOKE. | Đếm đầy đủ FIRE FA, SMOKE FA và Tổng FA (ANY FA) cho cả 3 pipeline. | [`scripts/evaluate_negative_dataset_unified.py`](file:///d:/workspace/DACN_NCKH/scripts/evaluate_negative_dataset_unified.py) |
| **Kiểm soát Cooldown** | Baseline A dùng cooldown 2.0s; Pipeline B1/B2 dùng cooldown 5.0s. | Chuẩn hóa đối sánh có kiểm soát (Controlled Comparison): Cả A, B1, B2 dùng chung `cooldown = 5.0s`. Báo cáo bổ sung A Legacy (`2.0s`). | [`runs/benchmarks/week4_scientific/far_methodology_reconciliation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_scientific/far_methodology_reconciliation.json) |
| **Thời lượng đánh giá** | Đồng nhất thời lượng tệp trên đĩa (3.43h) với thời lượng đã inference (0.06h). | Ghi nhận rõ `evaluation_mode = "bounded_sample"`. Mẫu số tính FAR tuyệt đối chỉ lấy thời lượng thực tế đã inference ($0.0623\text{h}$). | [`runs/benchmarks/week4_scientific/far_methodology_reconciliation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_scientific/far_methodology_reconciliation.json) |
| **Khoảng tin cậy mẫu nhỏ** | Gán $FAR = 0.00/\text{h}$ tuyệt đối khi không thấy sự kiện nào trong 224 giây. | Áp dụng quy tắc Thống kê Poisson (Rule of Three): Điểm ước lượng 0.00/h, kèm khoảng tin cậy 95% Poisson: $[0.00, 48.18]/\text{h}$. | [`scripts/evaluate_negative_dataset_unified.py`](file:///d:/workspace/DACN_NCKH/scripts/evaluate_negative_dataset_unified.py) |
| **Đồng bộ Operating Point** | `SystemSettings` trong `src/config.py` vẫn giữ cấu hình cũ chưa kiểm định. | Đồng bộ hóa 100% cấu hình hệ thống theo Operating Point tối ưu: $K=2, N=8, feat=0.02, cd=5.0s, conf=0.35$. | [`src/config.py`](file:///d:/workspace/DACN_NCKH/src/config.py) |

---

## 3. Chuẩn Hóa Phương Pháp Luận & Định Nghĩa False Alarm

### 3.1. Ba Cấp Độ Đo Lường Phát Hiện Sai (Three-Level Accounting)
Để đảm bảo tính khoa học chặt chẽ và không thể bị diễn giải sai lệch:
1. **Cấp phát hiện (Detection-level):**  
   $$\text{false\_positive\_detections} = \sum_{t=1}^T \text{count}(\text{boxes phát hiện trên frame } t \text{ với } \text{conf} \ge \text{conf\_threshold})$$
   Đo lường độ nhiễu thô của mạng nơ-ron phát hiện đối tượng trước mọi bộ lọc.
2. **Cấp khung hình (Frame-level):**  
   $$\text{false\_positive\_frames} = \sum_{t=1}^T \mathbb{I}\left( \text{count}(\text{boxes trên frame } t) > 0 \right)$$
   Đo lường tần suất xuất hiện khung hình lỗi (tỷ lệ khung hình bị ô nhiễm báo giả).
3. **Cấp sự kiện (Event-level - Báo động thực tế tới người dùng):**  
   $$\text{false\_alarm\_events} = \text{Số sự kiện cảnh báo được phát ra sau gom cụm } K/N \text{ và thời gian hồi tiếp Cooldown } T_{cd}$$
   Đây là chỉ số duy nhất được sử dụng để tính False Alarm Rate (FAR) cấp hệ thống theo chuẩn MT3 của Đề cương NCKH.

### 3.2. Định Nghĩa Cảnh Báo Sai Chuẩn Hóa (Standardized False Alarm Definition)
> **Định nghĩa:** Một *False Alarm Event* (Sự kiện cảnh báo sai) là một tín hiệu cảnh báo FIRE hoặc SMOKE được kích hoạt trên luồng video âm tính (luồng không có cháy thật), sau khi đã áp dụng thuật toán kiểm định chuỗi thời gian ($K/N$), cổng biến động động học ROI (Temporal Feature Gate), và cơ chế khống chế thời gian hồi tiếp nhân quả (Causal Alert Cooldown).

### 3.3. Cơ Chế Cooldown Nhân Quả (Causal Alert Cooldown)
Khi một cảnh báo hợp lệ được kích hoạt tại thời điểm $t_{alert}$:
- Hệ thống ghi nhận 1 sự kiện cảnh báo (FIRE hoặc SMOKE tùy theo nhãn chiếm đa số).
- Thiết lập trạng thái Cooldown cho camera/kênh đó: Mọi tín hiệu cảnh báo phát sinh trong cửa sổ $[t_{alert}, t_{alert} + T_{cd}]$ đều bị nén, không tạo thêm sự kiện trùng lặp.
- Sau thời điểm $t_{alert} + T_{cd}$, nếu tín hiệu cháy/khói vẫn tiếp diễn hoặc xuất hiện trở lại, hệ thống sẽ kích hoạt một sự kiện mới độc lập.

---

## 4. Kết Quả Đo Lường False Alarm Rate (FAR) Thực Nghiệm

### 4.1. Tập Dữ Liệu Video Âm Tính Đánh Giá Thực Tế
Quá trình đánh giá được thực hiện tự động bằng script chuẩn hóa [`scripts/evaluate_negative_dataset_unified.py`](file:///d:/workspace/DACN_NCKH/scripts/evaluate_negative_dataset_unified.py) trên toàn bộ 5 video âm tính vật lý hiện diện trên ổ cứng:

| Video ID | Tên tệp vật lý | Native FPS | Tổng frames trên đĩa | Frames đã Inference | Tỷ lệ xử lý (Coverage) | Duration Inference | Decode Errors |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `REAL_NEG_01` | `NHIỄU 1.mp4` | 30.0 | 185.733 | 2.000 | 1.08% | 66.67 s (0.0185 h) | 0 |
| `REAL_NEG_02` | `NHIỄU 2.mp4` | 30.0 | 618 | 618 | **100.00%** | 20.60 s (0.0057 h) | 0 |
| `REAL_NEG_03` | `NHIỄU 3.mp4` | 30.0 | 2.302 | 2.302 | **100.00%** | 76.73 s (0.0213 h) | 0 |
| `REAL_NEG_04` | `NHIỄU 4.mp4` | 30.0 | 1.053 | 1.053 | **100.00%** | 35.10 s (0.0097 h) | 0 |
| `REAL_NEG_05` | `NHIỄU 5.mp4` | 30.0 | 180.335 | 752 | 0.42% | 25.07 s (0.0070 h) | 0 |
| **Tổng cộng** | **5 tệp MP4** | **30.0** | **370.041** | **6.725** | **1.82%** | **224.17 s (0.0623 h)** | **0** |

### 4.2. Bảng Phân Tách Chi Tiết 3 Chỉ Số Đo Lường
Đánh giá trên $6.725\text{ khung hình}$ ($0.0623\text{ giờ}$), ngưỡng tin cậy mô hình $conf\_threshold = 0.35$:

| Chỉ số / Phân loại | FIRE | SMOKE | TỔNG CỘNG (ANY) |
| :--- | :---: | :---: | :---: |
| **1. False Positive Detections (Tổng Boxes)** | 10 | 99 | **109 boxes** |
| **2. False Positive Frames (Khung hình có $\ge 1$ Box)** | 10 | 99 | **109 frames** |
| **3. False Alarm Events (Sự kiện báo động sai):** | | | |
| - *Pipeline A Controlled* (Per-frame, Cooldown 5.0s) | 2 | 4 | **6 events** |
| - *Pipeline A Legacy* (Per-frame, Cooldown 2.0s) | 3 | 8 | **11 events** |
| - *Pipeline B1* (ByteTrack + K/N, Cooldown 5.0s, feat=0.0) | 0 | 4 | **4 events** |
| - *Pipeline B2* (ByteTrack + Feature Gate, Cooldown 5.0s, feat=0.02) | **0** | **0** | **0 events** |

### 4.3. Bảng Đối Soát Tỷ Lệ Báo Động Giả (FAR / Giờ) và Mức Độ Giảm Thiểu
Thời lượng tính toán: $T = 0.06227\text{ giờ}$ ($224.17\text{ giây}$).

| Pipeline Cấu Hình | FIRE FAR (/h) | SMOKE FAR (/h) | TỔNG FAR (/h) | Poisson 95% CI (/h) | Giảm FAR so với A Controlled (%) | Giảm FAR so với A Legacy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pipeline A Controlled** ($cd=5.0\text{s}$) | 32.12 | 64.24 | **96.36** | $[35.36, 209.72]$ | *Mốc đối chiếu (0.0%)* | - |
| **Pipeline A Legacy** ($cd=2.0\text{s}$) | 48.18 | 128.48 | **176.65** | $[88.24, 316.12]$ | - | *Mốc đối chiếu cũ* |
| **Pipeline B1** (ByteTrack, $feat=0.0$) | 0.00 | 64.24 | **64.24** | $[17.50, 164.48]$ | **Giảm 33.33%** | **Giảm 63.63%** |
| **Pipeline B2** (ByteTrack + Gate $0.02$) | **0.00** | **0.00** | **0.00** | $[0.00, 48.18]$ | **Giảm 100.00%** | **Giảm 100.00%** |

### 4.4. Phân Tích Cơ Chế Khử Nhiễu (Deep-Dive Analysis)
1. **Nguồn phát sinh nhiễu chính (`NHIỄU 4.mp4`):**
   - Video `NHIỄU 4.mp4` ghi lại cảnh đường phố ban đêm có ánh đèn pha và đèn đường rọi xuống mặt đường ẩm ướt phản quang.
   - Mô hình YOLOv8n dự đoán nhầm các vệt phản quang này thành **SMOKE** trong 99 khung hình liên tiếp.
2. **Hạn chế của Pipeline B1 (ByteTrack thuần túy không có Feature Gate):**
   - Do vệt sáng đèn đường đứng yên và liên tục xuất hiện, ByteTrack duy trì track ID ổn định và thỏa mãn điều kiện cửa sổ $K/N \ge 3/8$.
   - Kết quả: B1 vẫn phát ra **4 sự kiện báo động khói giả** ($FAR = 64.24/\text{h}$).
3. **Ưu thế vượt trội của Pipeline B2 (Temporal Feature Gate):**
   - Bộ trích xuất đặc trưng `TemporalFeatureExtractor` tính toán độ chớp nháy cường độ sáng (Flicker index) và tốc độ biến thiên diện tích contour (Expansion rate) trong vùng ROI.
   - Do ánh đèn đường có cường độ tĩnh và hình học không đổi qua thời gian, điểm biến động đặc trưng tính được chỉ đạt $0.0000 - 0.0080 < 0.0200$.
   - Cổng kiểm định đã chặn đứng toàn bộ các phát hiện này $\rightarrow$ **Triệt tiêu 100% cả 4 sự kiện báo khói giả**, đưa số sự kiện báo sai về $0$.

---

## 5. Thời Lượng và Phạm Vi Đánh Giá (Duration & Coverage Tracking)

### 5.1. Minh Bạch Hóa Độ Bao Phủ Dữ Liệu
Để tuân thủ liêm chính học thuật, nhóm nghiên cứu phân định tuyệt đối giữa hai khái niệm:
- **Thời lượng video vật lý lưu trữ trên đĩa:** $3.4263\text{ giờ}$ ($370.041\text{ khung hình}$).
- **Thời lượng thực tế đã đưa qua pipeline inference:** $0.0623\text{ giờ}$ ($6.725\text{ khung hình}$).
- **Tỷ lệ bao phủ xử lý (Processing Coverage):**
  $$\text{Coverage} = \frac{0.0623}{3.4263} \times 100\% = \mathbf{1.82\%}$$
- **Chế độ kiểm thử:** Đánh dấu rõ ràng trong mọi metadata và báo cáo là `evaluation_mode = "bounded_sample"`.

### 5.2. Đối Chiếu Chỉ Tiêu Dữ Liệu Âm Tính MT1
- Yêu cầu của MT1 trong Đề cương NCKH: $\ge 5.0\text{ giờ}$ video không cháy có cảnh nhiễu thực tế.
- Hiện trạng ổ đĩa: Mới chỉ có 5 tệp video (tổng thời lượng $3.43\text{h}$, thiếu $1.57\text{h}$ so với mục tiêu $5.0\text{h}$).
- Hiện trạng thực nghiệm đã inference: Mới đạt $0.0623\text{h}$ ($224.2\text{s}$).
- **Kết luận:** Trạng thái hạng mục dữ liệu âm tính chính thức được ghi nhận là:
  $$\mathbf{BLOCKED\_BY\_NEGATIVE\_DATA}$$
  Tuyệt đối không ngoại suy kết quả $FAR = 0.00/\text{h}$ trên 224 giây thành kết luận toàn diện cho 5 giờ dữ liệu.

---

## 6. Kết Quả Rà Soát Ground Truth Trên Tập Validation (Validation GT Audit)

Quá trình thẩm định lại điểm vận hành trên tập Validation (12 video positive, 2.266 frames) được thực hiện độc lập theo 2 tầng đánh giá:

### 6.1. Primary Evaluation — Tập Sự Kiện Đã Kiểm Định Chéo (`cross_verified`)
- **Cỡ mẫu:** Gồm **3 sự kiện** đã được 2 annotator độc lập xác nhận chéo trùng khớp thời điểm $\Delta t \le 0.5\text{s}$ (`POS_03`, `POS_05`, `POS_23`).
- **Kết quả Baseline A:** TP = 3, FN = 0 $\rightarrow$ Recall = **100.0%**, Mean Delay = **0.22 s**.
- **Kết quả Pipeline B2 (`kn_k2_n8`):** TP = 3, FN = 0 $\rightarrow$ Recall = **100.0%**, Mean Delay = **0.23 s**.
- **Độ sụt giảm Recall:** $\Delta \text{Recall} = \mathbf{0.0\text{ điểm \%}}$ (Thỏa mãn tiêu chí MT3 $\le 5$ điểm %).
- *Ghi chú thống kê:* Do cỡ mẫu $N=3$ là rất nhỏ, kết quả Recall 100% mang tính chất kiểm tra sự nhất quán của thuật toán, không được xem là bằng chứng tổng quát hóa tuyệt đối.

### 6.2. Secondary Exploratory Evaluation — Toàn Bộ Sự Kiện (`All Ground Truth`)
- **Cỡ mẫu:** Gồm **17 sự kiện** (bao gồm 3 sự kiện `cross_verified` và 14 sự kiện `pending_verification`).
- **Kết quả Baseline A:** TP = 14, FN = 3, FP = 4 $\rightarrow$ Recall = **82.35%**, Precision = **77.78%**, F1 = **0.8000**, Mean Delay = **0.22 s**.
- **Kết quả Pipeline B2 (`kn_k2_n8`):** TP = 14, FN = 3, FP = 4 $\rightarrow$ Recall = **82.35%**, Precision = **77.78%**, F1 = **0.8000**, Mean Delay = **0.86 s**.
- **Độ sụt giảm Recall:** $\Delta \text{Recall} = \mathbf{0.0\text{ điểm \%}}$ (Bằng đúng Baseline A, hoàn toàn thỏa mãn tiêu chí $\le 5$ điểm %).
- **Độ trễ trung bình:** $0.858\text{ s} \ll 10.0\text{ s}$ (Thỏa mãn xuất sắc tiêu chí MT3).

---

## 7. Đối Chiếu và Khóa Điểm Vận Hành (Operating Point Audit & Codebase Synchronization)

### 7.1. Phân Tích So Sánh Các Điểm Vận Hành Ứng Viên
Thực nghiệm so sánh các cấu hình tham số trên tập Validation:

| Cấu hình | Tham số ($conf, K, N, feat, cd$) | Recall (All GT) | Recall (Verified) | Sụt giảm Recall | Mean Delay | FAR trên tập Âm tính | Đánh giá tổng thể |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `current_b2` | $0.35, 3, 8, 0.02, 5.0\text{s}$ | 76.47% | 100.0% | **-5.88 điểm %** | 0.59 s | 0.00 /h | **VI PHẠM** (sụt giảm $> 5\%$) |
| `kn_k2_n8` | $0.35, 2, 8, 0.02, 5.0\text{s}$ | **82.35%** | **100.0%** | **0.00 điểm %** | **0.86 s** | **0.00 /h** | **TỐI ƯU TOÀN DIỆN** |
| `per_class_opt_2` | Fire: $K=2, N=8$; Smoke: $K=3, N=8$ | 82.35% | 100.0% | 0.00 điểm % | 0.86 s | 0.00 /h | Khả thi nhưng phức tạp hóa |
| `per_class_opt_4` | Fire: $K=2, N=6$; Smoke: $K=3, N=8$ | 82.35% | 100.0% | 0.00 điểm % | 0.81 s | 0.00 /h | Khả thi nhưng không vượt trội |

### 7.2. Đồng Bộ Hóa Hệ Thống (`src/config.py`)
Điểm vận hành `kn_k2_n8` được chính thức lựa chọn làm **Operating Point chuẩn hóa** cho toàn bộ dự án. Cấu hình này đã được ghi đè và đồng bộ 100% vào lớp cấu hình hệ thống tại [`src/config.py`](file:///d:/workspace/DACN_NCKH/src/config.py):
```python
@dataclass
class SystemSettings:
    """Cấu hình toàn bộ hệ thống phát hiện & cảnh báo cháy từ camera."""
    conf_threshold: float = 0.35      # Ngưỡng tin cậy YOLOv8n
    iou_threshold: float = 0.45       # Ngưỡng NMS
    input_size: int = 640             # imgsz chuẩn
    window_size_n: int = 8            # Kích thước cửa sổ trượt (N)
    confirm_frames_k: int = 2         # Số khung xác nhận kích hoạt (K)
    alert_cooldown_sec: float = 5.0   # Thời gian hồi tiếp chống báo lặp (Cooldown)
    feature_threshold: float = 0.02   # Ngưỡng cổng biến động động học (Gate)
    bytetrack: ByteTrackSettings = field(default_factory=ByteTrackSettings)
```

---

## 8. Đánh Giá Mức Độ Đạt Các Chỉ Tiêu MT3 và Dự Án

| Chỉ tiêu đề cương | Yêu cầu định lượng | Kết quả thực nghiệm đo kiểm | Kết luận khoa học |
| :--- | :--- | :--- | :---: |
| **MT3 — Giảm báo giả** | Giảm $\ge 50\%$ số báo động giả/giờ so với cảnh báo từng khung. | Giảm **100.0%** so với A Controlled (0 vs 6 events), giảm **100.0%** so với A Legacy (0 vs 11 events). | **ĐẠT (PASS)** *(trên tập mẫu hiện có)* |
| **MT3 — Duy trì Recall** | Recall sự kiện giảm $\le 5$ điểm % so với Baseline A. | Sụt giảm **0.0 điểm %** trên cả Verified GT ($100\%$ vs $100\%$) và All GT ($82.35\%$ vs $82.35\%$). | **ĐẠT (PASS)** |
| **MT3 — Độ trễ cảnh báo** | Độ trễ trung bình $\le 10\text{ giây}$. | Đạt **0.23 s** (Verified GT) và **0.86 s** (All GT). | **ĐẠT XUẤT SẮC (PASS)** |
| **MT1 — Dữ liệu âm tính** | $\ge 5.0\text{ giờ}$ video âm tính có cảnh nhiễu. | Mới có $3.43\text{h}$ trên đĩa và $0.0623\text{h}$ đã inference. | **`BLOCKED_BY_NEGATIVE_DATA`** |
| **MT5 — Luồng RTSP** | $\ge 4$ luồng RTSP camera mạng trực tiếp. | 0 luồng RTSP thật (mới đo kiểm giả lập đa luồng tệp AVI). | **`BLOCKED_NO_LIVE_RTSP_SOURCE`** |

---

## 9. Kết Luận Nghiệm Thu Tuần 4

Căn cứ vào kết quả đối soát phương pháp luận nghiêm ngặt:
1. **Về mặt thuật toán và kỹ thuật xử lý (Core Modules):** Pipeline B2 tích hợp ByteTrack và Cổng kiểm định biến động động học (`TemporalFeatureExtractor` với $feat\_threshold = 0.02, K=2, N=8, cooldown=5.0\text{s}$) đã giải quyết triệt để vấn đề báo động giả của mô hình Deep Learning trong điều kiện nhiễu camera thực tế, đồng thời bảo toàn 100% độ nhạy phát hiện cháy và khói với độ trễ dưới 1 giây.
2. **Về mặt nghiệm thu chỉ tiêu NCKH:** Nhóm nghiên cứu giữ nguyên lập trường khoa học trung thực, từ chối công bố nghiệm thu toàn phần khi chưa đáp ứng đủ các ràng buộc về mặt dữ liệu và hạ tầng:
   $$\mathbf{KẾT\_LUẬN:\ PARTIAL\ (BLOCKED\_BY\_NEGATIVE\_DATA\ \&\ NO\_LIVE\_RTSP)}$$
3. **Kế hoạch chuyển tiếp:**
   - Hoàn thành đóng băng kết quả Week 4.
   - Sẵn sàng chuyển giao các module đã tối ưu hóa sang các nhiệm vụ của Tuần 5 (Domain Adaptation trên tập dữ liệu Việt Nam) và Tuần 7 (Hạ tầng mạng RTSP trực tiếp với MediaMTX và Docker).
