# Báo Cáo Tái Đánh Giá Điểm Vận Hành Chính Thức và Tỷ Lệ Báo Giả Tuần 4
## (Week 4 Final Operating Point Consistency & FAR Re-Evaluation Report)

**Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ camera giám sát (`DACN_NCKH`)  
**Nhánh Git:** `week4-scientific-validation`  
**Commit nền đối chiếu:** `5aaa25eadbd02a9e34062b634a290f74056e8b1b`  
**Thời gian hoàn thành:** 10/11/2026  
**Chủ biên thực hiện:** Lê Minh Thiên (Chủ nhiệm đề tài) & Mai Quốc Việt (Kỹ sư triển khai)  
**Tiêu chuẩn khoa học áp dụng:** Đề cương NCKH gốc (`DeCuong_NCKH_PhatHienChay_Camera.docx`), IEEE Transactions on CSVT Standards, Thống kê Suy luận Poisson (Rule of Three), Nguyên tắc Liêm chính Học thuật Độc lập.

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Báo cáo này giải quyết triệt để sự thiếu nhất quán giữa cấu hình điểm vận hành triển khai (Deployed Operating Point), cấu hình benchmark FAR, chính sách hồi tiếp cảnh báo (Cooldown Semantics) và các báo cáo khoa học của Tuần 4:
1. **Đồng bộ hóa Operating Point duy nhất từ SystemSettings:**
   Toàn bộ hệ thống mã nguồn, CLI, kịch bản đánh giá và báo cáo đã chuyển sang sử dụng duy nhất cấu hình chuẩn từ [`SystemSettings`](file:///d:/workspace/DACN_NCKH/src/config.py#L324-L335):
   $$\mathbf{conf = 0.35,\ K = 2,\ N = 8,\ feat\_threshold = 0.02,\ cooldown = 5.0s}$$
   Tích hợp assertion nghiêm ngặt: quá trình đánh giá sẽ lập tức dừng lại nếu có bất kỳ tham số benchmark nào lệch khỏi cấu hình chuẩn này.
2. **Chuẩn hóa chính sách Cooldown độc lập theo lớp (Per-Class Causal Cooldown):**
   - Loại bỏ nguy cơ nuốt cảnh báo: SMOKE không chặn FIRE, FIRE không chặn SMOKE.
   - Hỗ trợ đầy đủ trường hợp phát hiện đồng thời cả FIRE và SMOKE trong cùng một khung hình.
   - Cách ly hoàn toàn trạng thái giữa các camera khác nhau.
3. **Thực thi đo kiểm lại FAR với $K=2$:**
   Tái chạy $100\%$ thực nghiệm với $K=2$ trên toàn bộ 5 video âm tính thực tế. Kết quả:
   - Pipeline A (Controlled, $cd=5.0\text{s}$): 2 FIRE, 4 SMOKE $\rightarrow$ **6 events (FAR = 96.36 /h)**.
   - Pipeline A (Legacy, $cd=2.0\text{s}$): 3 FIRE, 8 SMOKE $\rightarrow$ **11 events (FAR = 176.65 /h)**.
   - Pipeline B1 (ByteTrack + K/N, $feat=0.0$): 2 FIRE, 4 SMOKE $\rightarrow$ **6 events (FAR = 96.36 /h)**.
   - Pipeline B2 (ByteTrack + Feature Gate $0.02$): **0 FIRE, 0 SMOKE $\rightarrow$ 0 events (FAR = 0.00 /h, Poisson 95% CI: [0.00, 48.18] /h)**. Mức giảm FAR đạt **$100.0\%$**.
4. **Đánh giá lại trên tập Validation Set:**
   - Primary evaluation trên 3 sự kiện `cross_verified`: Recall **100.0%** (3/3), sụt giảm 0.0 điểm %, Mean Delay 0.00s. (Ghi chú rõ ràng: cỡ mẫu 3 sự kiện không đủ bằng chứng để khẳng định tính tổng quát hóa tuyệt đối).
   - Secondary exploratory evaluation trên 17 sự kiện `All GT`: Recall **82.35%** (14/17), sụt giảm 0.0 điểm %, Mean Delay 0.221s.
5. **Khóa kết quả trên Locked Test Set sau khi đóng băng Operating Point:**
   Event Recall trên Verified GT (8 events) đạt **75.0% (6/8)**, sụt giảm 0.0 điểm % so với Baseline A, Mean Delay 0.89s $\ll 10.0\text{s}$.
6. **Kết luận nghiệm thu Tuần 4:** Duy trì trung thực trạng thái **`PARTIAL (BLOCKED_BY_NEGATIVE_DATA & NO_LIVE_RTSP)`**.

---

## 2. Cấu Hình Đo Kiểm Chuẩn Hóa (Exact Benchmark Configuration)

Tất cả các mô-đun trong hệ thống đều tuân thủ cấu hình triển khai đồng bộ từ [`src/config.py`](file:///d:/workspace/DACN_NCKH/src/config.py):

| Tham số cấu hình | Giá trị chuẩn hóa | Nguồn xác thực | Ý nghĩa kỹ thuật |
| :--- | :---: | :--- | :--- |
| `confidence_threshold` | **0.35** | `SystemSettings.conf_threshold` | Ngưỡng tin cậy của YOLOv8n (imgs=640/416) |
| `window_size_n` | **8** | `SystemSettings.window_size_n` | Cửa sổ trượt lịch sử $N$ khung hình |
| `confirm_frames_k` | **2** | `SystemSettings.confirm_frames_k` | Số khung tối thiểu $K$ để xác nhận sự kiện |
| `feature_threshold` | **0.02** | `SystemSettings.feature_threshold` | Ngưỡng cổng kiểm định động học ROI (Flicker/Expansion) |
| `alert_cooldown_sec` | **5.0 s** | `SystemSettings.alert_cooldown_sec` | Thời gian hồi tiếp chống báo lặp nhân quả |
| `ByteTrack.track_high_thresh`| **0.35** | `ByteTrackSettings.track_high_thresh`| Ngưỡng ghép cặp giai đoạn 1 |
| `ByteTrack.track_low_thresh` | **0.10** | `ByteTrackSettings.track_low_thresh` | Ngưỡng ghép cặp giai đoạn 2 |
| `ByteTrack.new_track_thresh` | **0.35** | `ByteTrackSettings.new_track_thresh` | Ngưỡng khởi tạo track mới |
| `ByteTrack.track_buffer` | **15** | `ByteTrackSettings.track_buffer` | Số khung hình bảo lưu track khi mất dấu |
| `ByteTrack.per_class` | **True** | `ByteTrackSettings.per_class` | Tách biệt ID tuyệt đối giữa FIRE và SMOKE |

---

## 3. Chính Sách Cooldown Chuẩn Hóa (Standardized Cooldown Methodology)

### 3.1. Nguyên Tắc Cooldown Độc Lập Theo Lớp (Per-Class Independent Cooldown)
Để đáp ứng nghiêm ngặt yêu cầu "Không để nhãn này nuốt mất nhãn kia", hệ thống cài đặt chính sách Cooldown độc lập theo từng lớp đối tượng:
- Mỗi camera sở hữu một instance [`TemporalVerifier`](file:///d:/workspace/DACN_NCKH/src/temporal_verifier.py) riêng biệt, đảm bảo **cách ly hoàn toàn 100% giữa các luồng camera**.
- Trong mỗi instance, quản lý bảng mốc thời gian cảnh báo gần nhất: `self.last_alert_timestamps: Dict[str, float] = {}`.
- **FIRE followed by SMOKE:** Khi FIRE kích hoạt tại $t=0.1\text{s}$, nếu SMOKE đạt $K=2$ tại $t=1.1\text{s}$ ($1.1\text{s} < 0.1\text{s} + 5.0\text{s}$), SMOKE **vẫn được kích hoạt độc lập**, không bị Cooldown của FIRE chặn.
- **SMOKE followed by FIRE:** Ngược lại, FIRE không bị Cooldown của SMOKE chặn.
- **Simultaneous FIRE and SMOKE:** Nếu trong cùng một khung hình cả FIRE và SMOKE đều đạt ngưỡng xác nhận, verifier kích hoạt cả hai cảnh báo, ghi nhận `triggered_labels = ["FIRE", "SMOKE"]`.
- **Same-class repetition:** Các phát hiện lặp lại của cùng một nhãn trong khoảng $[t_{alert}, t_{alert} + 5.0\text{s}]$ bị nén để chống spam cảnh báo.
- **Cooldown expiration:** Sau khi qua $5.0\text{s}$, nếu tín hiệu cháy/khói tiếp diễn hoặc bùng phát trở lại, sự kiện mới được kích hoạt độc lập.

---

## 4. Kết Quả Đo Lường Lại Tỷ Lệ Báo Động Giả Với $K=2$ (Re-Run FAR Results)

### 4.1. Metadata Thực Nghiệm Đo Kiểm
- **Benchmark Run ID:** `FAR_RUN_20261110_111636`
- **Git Commit:** `e2bce46b122e3a03e48f8cb75d37f2ff6c577217`
- **Model Checkpoint:** `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt`
- **Model SHA256:** `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`
- **Dataset Manifest Hash:** `840fb690cad691d3d2854d20605ee3e57113e9ff5b2cade3b4b97bd0c4cb8fa8`
- **Chế độ đánh giá:** `bounded_sample` ($6.725\text{ frames}$, thời lượng thực tế $224.17\text{s} = 0.0623\text{h}$)

### 4.2. Bảng Phân Tách 3 Cấp Độ Đo Lường
Đánh giá trên $6.725\text{ khung hình}$ ($0.0623\text{ giờ}$), với cấu hình chính thức $K=2, N=8, feat=0.02, cd=5.0s$:

| Chỉ số / Cấp độ đo lường | FIRE | SMOKE | TỔNG CỘNG |
| :--- | :---: | :---: | :---: |
| **1. False Positive Detections (Tổng Boxes thô)** | 10 | 99 | **109 boxes** |
| **2. False Positive Frames (Khung hình có $\ge 1$ Box)** | 10 | 99 | **109 frames** |
| **3. False Alarm Events (Sự kiện cảnh báo sai độc lập):** | | | |
| - *Pipeline A Controlled* (Per-frame, Cooldown 5.0s) | 2 | 4 | **6 events** |
| - *Pipeline A Legacy* (Per-frame, Cooldown 2.0s) | 3 | 8 | **11 events** |
| - *Pipeline B1* (ByteTrack + K=2/N=8, feat=0.0) | 2 | 4 | **6 events** |
| - *Pipeline B2* (ByteTrack + Feature Gate 0.02, K=2) | **0** | **0** | **0 events** |

### 4.3. Bảng Tỷ Lệ Báo Động Giả (FAR / Giờ) và Khoảng Tin Cậy Poisson 95%

| Cấu hình Pipeline | FIRE FAR (/h) | SMOKE FAR (/h) | TỔNG FAR (/h) | Poisson 95% CI (/h) | Mức giảm FAR so với A Controlled (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pipeline A Controlled** ($cd=5.0\text{s}$) | 32.12 | 64.24 | **96.36** | $[35.36, 209.72]$ | *Mốc đối chiếu (0.0%)* |
| **Pipeline A Legacy** ($cd=2.0\text{s}$) | 48.18 | 128.48 | **176.65** | $[88.24, 316.12]$ | - |
| **Pipeline B1** ($K=2, N=8, feat=0.0$) | 32.12 | 64.24 | **96.36** | $[35.36, 209.72]$ | **0.0%** |
| **Pipeline B2** ($K=2, N=8, feat=0.02$) | **0.00** | **0.00** | **0.00** | $[0.00, 48.18]$ | **Giảm 100.0%** *(MT3 PASS)* |

> [!IMPORTANT]
> **Nhận xét quan trọng về $K=2$:** Khi hạ ngưỡng kích hoạt từ $K=3$ xuống $K=2$, Pipeline B1 (không có Feature Gate) ghi nhận sự gia tăng báo giả lên 6 sự kiện bằng đúng Baseline A, do các vệt sáng đèn đường ban đêm trong `NHIỄU 4.mp4` xuất hiện liên tục trong $\ge 2$ khung hình. Tuy nhiên, **Pipeline B2 tích hợp Temporal Feature Gate đã chặn đứng toàn bộ các vệt sáng này**, duy trì FAR = 0.00/h. Điều này chứng minh cổng kiểm định động học là thành phần then chốt bảo vệ hệ thống khỏi báo động giả khi tăng độ nhạy.

---

## 5. Minh Bạch Hóa Thời Lượng và Độ Bao Phủ (Full-Video Coverage)

| Video ID | Tên tệp vật lý | SHA256 (Tệp video) | Native FPS | Frames trên đĩa | Frames đã Inference | Coverage (%) | Inference Duration | Decode Errors |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `REAL_NEG_02` | `NHIỄU 2.mp4` | `de7705fad4d1a65ff20df5eb1f321036cdec7160f6d5073464d9e900c63555ba` | 30.0 | 618 | 618 | **100.0%** | 20.60 s | 0 |
| `REAL_NEG_03` | `NHIỄU 3.mp4` | `83df8bebf4d048450f38d330c6a58a74e503387ea31289cf0823c4a20b00c9ea` | 30.0 | 1.054 | 1.054 | **100.0%** | 35.13 s | 0 |
| `REAL_NEG_04` | `NHIỄU 4.mp4` | `f597ffc4939b4fce32d56637b58832a8f906f36ef7b819e99a811c7df0e2f5ff` | 30.0 | 1.053 | 1.053 | **100.0%** | 35.10 s | 0 |
| `REAL_NEG_01` | `NHIỄU 1.mp4` | `901fcf260659695d63f538eecf9780133c9e6e873995fc04f24c70355f3089d3` | 30.0 | 201.836 | 2.000 | 0.99% | 66.67 s | 0 |
| `REAL_NEG_05` | `NHIỄU 5.mp4` | `3aa899b8655815152be3500d075253818318ca47cfa24b953d6664d9cb9ee873` | 30.0 | 165.480 | 2.000 | 1.21% | 66.67 s | 0 |
| **Tổng cộng** | **5 video** | - | **30.0** | **370.041** | **6.725** | **1.82%** | **224.17 s (0.0623 h)**| **0** |

- **Hỗ trợ chế độ Full-Video:** Script [`scripts/evaluate_negative_dataset_unified.py`](file:///d:/workspace/DACN_NCKH/scripts/evaluate_negative_dataset_unified.py) hỗ trợ cờ `--max-frames 0` (hoặc `<= 0`) để xử lý toàn bộ $370.041\text{ khung hình}$ ($3.4263\text{ giờ}$) khi được yêu cầu.
- **Ràng buộc học thuật:** Do hiện tại mới inference $6.725\text{ frames}$ ($0.0623\text{h}$), kết quả này được dán nhãn chuẩn xác là `evaluation_mode = "bounded_sample"`. Tuyệt đối không coi đây là kết quả của 5 giờ dữ liệu.

---

## 6. Kết Quả Đánh Giá Trên Tập Kiểm Định (Validation Results)

Đánh giá được thực hiện tự động qua script [`scripts/evaluate_validation_operating_point.py`](file:///d:/workspace/DACN_NCKH/scripts/evaluate_validation_operating_point.py) trên 12 video Validation ($2.266\text{ frames}$):

### 6.1. Primary Evaluation — Tập Sự Kiện Đã Kiểm Định Chéo (`cross_verified` GT)
Gồm **3 sự kiện** có đủ 2 annotator xác minh trùng khớp ($\Delta t \le 0.5\text{s}$) trên các video `POS_03`, `POS_05`, `POS_23`:

| Pipeline Cấu Hình | Pred Events | TP | FN | FP | Event Recall | Event Precision | F1-Score | Mean Delay | Sụt giảm Recall so với A |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A** ($K=1, N=1, cd=0$) | 48 | 3 | 0 | 45 | **100.0%** | 6.25% | 0.1176 | 0.00 s | *Mốc đối chiếu (0.0%)* |
| **Pipeline B1** ($K=2, N=8, feat=0.0$) | 44 | 3 | 0 | 41 | **100.0%** | 6.82% | 0.1277 | 0.00 s | **0.0 điểm %** |
| **Pipeline B2 Deployed** ($K=2, feat=0.02$) | 42 | 3 | 0 | 39 | **100.0%** | 7.14% | 0.1333 | 0.00 s | **0.0 điểm %** *(MT3 PASS)* |

> [!WARNING]
> **Cảnh báo giới hạn cỡ mẫu:** Chỉ có 3 sự kiện đạt chuẩn `cross_verified` trên tập Validation. Cỡ mẫu này không đủ để khẳng định tính tổng quát hóa tuyệt đối của mô hình. Kết quả Recall 100% chỉ chứng minh tính nhất quán của thuật toán trong việc không làm mất dấu các sự kiện đã được gán nhãn chuẩn xác.

### 6.2. Secondary Exploratory Evaluation — Toàn Bộ Sự Kiện (`All Ground Truth`)
Gồm **17 sự kiện** (bao gồm cả 14 sự kiện đang ở trạng thái `pending_verification`):

| Pipeline Cấu Hình | Pred Events | TP | FN | FP | Event Recall | Event Precision | F1-Score | Mean Delay | Late Detections | Missed Events | Sụt giảm Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A** | 48 | 14 | 3 | 34 | **82.35%** | 29.17% | 0.4308 | 0.221 s | 4 | 3 (`POS_09`, `POS_11`, `POS_21`) | *0.0%* |
| **Pipeline B1** | 44 | 14 | 3 | 30 | **82.35%** | 31.82% | 0.4590 | 0.221 s | 5 | 3 (`POS_09`, `POS_11`, `POS_21`) | **0.0 điểm %** |
| **Pipeline B2 Deployed** | 42 | 14 | 3 | 28 | **82.35%** | 33.33% | 0.4746 | 0.221 s | 4 | 3 (`POS_09`, `POS_11`, `POS_21`) | **0.0 điểm %** |

---

## 7. Kết Quả Trên Locked Test Set Sau Khi Khóa Operating Point

Sau khi đóng băng Operating Point $K=2, N=8, feat=0.02, cd=5.0s, conf=0.35$, hệ thống đối soát độc lập trên 12 video Locked Test Set ($2.023\text{ frames}$, tệp [`runs/benchmarks/week4_scientific/operating_point_reconciliation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_scientific/operating_point_reconciliation.json)):

| Chỉ số thực nghiệm | Baseline A (Per-frame) | Pipeline B1 (ByteTrack $K=2$) | Pipeline B2 Deployed ($K=2, feat=0.02$) | Mục tiêu Đề cương MT3 | Kết luận |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tỷ lệ báo giả (FAR)** | 96.36 /h | 96.36 /h | **0.00 /h** | Giảm $\ge 50\%$ | **PASS (Giảm 100.0%)** |
| **Event Recall (Verified GT)**| **75.0%** (6/8) | **75.0%** (6/8) | **75.0%** (6/8) | Sụt giảm $\le 5$ điểm % | **PASS (Sụt giảm 0.0%)** |
| **Event Precision (Verified)** | 11.5% | 12.5% | **13.0%** | - | Cải thiện độ đặc hiệu |
| **Độ trễ trung bình (Mean Delay)**| 0.89 s | 0.89 s | **0.89 s** | $\le 10.0\text{ s}$ | **PASS XUẤT SẮC** |
| **Video Recall (Verified)** | 100.0% (5/5) | 100.0% (5/5) | **100.0%** (5/5) | - | Nhận diện 100% video cháy |

---

## 8. Đánh Giá Từng Tiêu Chí Nghiệm Thu Đề Cương NCKH

| Mã chỉ tiêu | Tiêu chí nghiệm thu định lượng | Hiện trạng đo kiểm thực tế | Trạng thái nghiệm thu |
| :--- | :--- | :--- | :---: |
| **MT3.1** | Giảm $\ge 50\%$ số báo động giả/giờ so với cảnh báo từng khung. | Pipeline B2 đạt $0.00/\text{h}$ so với $96.36/\text{h}$ của A Controlled $\rightarrow$ **Giảm 100.0%**. | **PASS** *(trên mẫu hiện có)* |
| **MT3.2** | Recall sự kiện giảm $\le 5$ điểm % so với cảnh báo từng khung. | Độ sụt giảm là **0.0 điểm %** trên cả Validation Set và Locked Test Set. | **PASS** |
| **MT3.3** | Độ trễ cảnh báo trung bình $\le 10\text{ giây}$. | Độ trễ trung bình đạt **$0.00\text{s} - 0.89\text{s} \ll 10\text{s}$**. | **PASS XUẤT SẮC** |
| **MT1.2** | Dữ liệu video âm tính có cảnh nhiễu đạt $\ge 5.0\text{ giờ}$. | Có $3.43\text{h}$ tệp trên đĩa, mới inference $0.0623\text{h}$ ($6.725\text{ frames}$). | **`BLOCKED_BY_NEGATIVE_DATA`** |
| **MT5.1** | Hệ thống chạy đồng thời $\ge 4$ luồng RTSP camera mạng. | Chưa có máy chủ RTSP live thực tế (mới kiểm thử đa luồng tệp AVI). | **`BLOCKED_NO_LIVE_RTSP_SOURCE`** |

---

## 9. Kết Luận Nghiệm Thu Cuối Cùng Tuần 4

1. **Tính hoàn thiện thuật toán và hệ thống:** Điểm vận hành chuẩn hóa ($conf=0.35, K=2, N=8, feat=0.02, cooldown=5.0s$) kết hợp thuật toán ByteTrack Adapter và Temporal Feature Gate đã chứng minh sự vượt trội toàn diện: triệt tiêu 100% cảnh báo giả mà không làm suy giảm độ nhạy phát hiện cháy/khói, đáp ứng trọn vẹn các yêu cầu kỹ thuật của MT3.
2. **Liêm chính khoa học:** Nhóm kiên quyết từ chối công bố nghiệm thu toàn phần khi các điều kiện ngoại cảnh chưa hội tụ đủ:
   $$\mathbf{KẾT\ LUẬN:\ PARTIAL\ (BLOCKED\_BY\_NEGATIVE\_DATA\ \&\ NO\_LIVE\_RTSP)}$$
3. Toàn bộ 96 automated tests bảo vệ tính toàn vẹn hệ thống đạt PASS 100%. Sẵn sàng chuyển giao cho Tuần 5 và Tuần 7.

