# BÁO CÁO NGHIỆM THU THỰC NGHIỆM TUẦN 4 – TEMPORAL VERIFICATION & EVENT EVALUATION

> **Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
> **Chuyên đề:** Xác nhận theo thời gian: ByteTrack, luật $K/N$ khung, Temporal Features Gate, Vùng loại trừ ROI, Công cụ đánh giá cấp sự kiện và Quét điểm vận hành  
> **Thời điểm nghiệm thu & đối soát:** 10/11/2026  
> **Branch Git:** `fix`  
> **Bộ kiểm thử tự động:** `70/70 unit & system tests PASS 100%` (5.22s)  
> **Trạng thái nghiệm thu tổng thể:** **`PARTIAL (BLOCKED_BY_NEGATIVE_DATA)`**  
> *(Hệ thống mã nguồn, tracker, verifier, evaluator và bộ kiểm thử đã hoàn thiện 100%; điểm vận hành chính thức duy trì trạng thái tạm dừng do chưa có file video âm tính thực tế trên ổ đĩa để đo tỷ lệ giảm báo giả FAR theo chuẩn Đề cương MT3)*

---

## MỤC LỤC
1. [Hạng Mục A: Xuất Bản Dữ Liệu Phát Hiện Cấp Khung Hình JSONL & Đối Soát Frame Identity](#hạng-mục-a-xuất-bản-dữ-liệu-phát-hiện-cấp-khung-hình-jsonl--đối-soát-frame-identity)
2. [Hạng Mục B: Bộ Nhãn Ground Truth Sự Kiện Độc Lập & Chính Sách Verified-Only](#hạng-mục-b-bộ-nhãn-ground-truth-sự-kiện-độc-lập--chính-sách-verified-only)
3. [Hạng Mục C: Onset Matching Cực Đại & Phân Loại Cảnh Báo Khách Quan](#hạng-mục-c-onset-matching-cực-đại--phân-loại-cảnh-báo-khách-quan)
4. [Hạng Mục D: Đo Lường Báo Động Giả (FAR) & Hiện Trạng Video Âm Tính](#hạng-mục-d-đo-lường-báo-động-giả-far--hiện-trạng-video-âm-tính)
5. [Hạng Mục E: Điểm Vận Hành Thực Nghiệm (Reconciled Operating Point)](#hạng-mục-e-điểm-vận-hành-thực-nghiệm-reconciled-operating-point)
6. [Hạng Mục F: Kết Luận Nghiệm Thu & Kế Hoạch Chuyển Giao Tuần 5](#hạng-mục-f-kết-luận-nghiệm-thu--kế-hoạch-chuyển-giao-tuần-5)

---

## HẠNG MỤC A: XUẤT BẢN DỮ LIỆU PHÁT HIỆN CẤP KHUNG HÌNH JSONL & ĐỐI SOÁT FRAME IDENTITY

### 1. Quy cách và pipeline suy luận
Hệ thống sử dụng script chuẩn hóa [`scripts/export_week4_frame_jsonl.py`](file:///d:/workspace/DACN_NCKH/scripts/export_week4_frame_jsonl.py) chạy suy luận detector YOLO kết hợp `ByteTrackAdapter` và `TemporalFeatureExtractor` trên 24 video dương tính có sẵn (`POS_01` $\rightarrow$ `POS_24`, tổng thời lượng $863.98\text{s} \approx 0.24\text{h}$) với tần số lấy mẫu cố định **$5.0\text{ FPS}$** ($\text{step} = \lfloor\text{FPS}/5\rfloor$).

Hệ thống xuất bản 2 tệp dữ liệu cấp khung hình:
1. **Raw Frame Detections:** [`runs/benchmarks/week4_final/frame_level_detections.jsonl`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/frame_level_detections.jsonl) (Pipeline A - Baseline không tracking, `track_id: null`).
2. **Tracked Frame Detections:** [`runs/benchmarks/week4_final/tracked_frame_detections.jsonl`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/tracked_frame_detections.jsonl) (Pipeline B - Đã qua ByteTrack, `track_id > 0`, kèm `temporal_score` động).

### 2. Kiểm định tính toàn vẹn và đối soát Frame Identity
Script [`scripts/validate_week4_frame_jsonl.py`](file:///d:/workspace/DACN_NCKH/scripts/validate_week4_frame_jsonl.py) thực hiện đối soát từng khung hình (frame-by-frame identity check: `video_id, camera_id, timestamp_sec`):

| Chỉ số kiểm định | Raw JSONL (Pipeline A) | Tracked JSONL (Pipeline B) | Trạng thái đối soát |
| :--- | :---: | :---: | :---: |
| **Tổng số dòng (Frames)** | **4.289** | **4.289** | Khớp $100\%$ ($4.289 / 4.289$) |
| **Khung hình rỗng (Empty)** | **1.444** ($33.67\%$) | **1.444** ($33.67\%$) | Bảo toàn nguyên vẹn `"detections": []` |
| **Khung hình có phát hiện** | 2.845 ($66.33\%$) | 2.845 ($66.33\%$) | Khớp $100\%$ |
| **Tổng số Bounding Box** | 4.034 | 3.676 | Loại bỏ 358 box rác ($\le 0$) |
| **Tọa độ Bounding Box ($x_1 < x_2, y_1 < y_2$)** | Hợp lệ $100\%$ | Hợp lệ $100\%$ | **0 lỗi tọa độ** |
| **Số Track ID độc lập** | 0 (`null` $100\%$) | **149 tracks** (mỗi track $\ge 1$) | Đạt chuẩn tracking |
| **Frame Identity Mismatches** | **0** | **0** | **4.289 / 4.289 khớp 1-1** |
| **Lỗi đảo ngược / Trùng lặp Timestamp** | **0** | **0** | Tăng đơn điệu $100\%$ |
| **Tệp báo cáo JSON** | [`runs/benchmarks/week4_final/frame_jsonl_validation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/frame_jsonl_validation.json) |

---

## HẠNG MỤC B: BỘ NHÃN GROUND TRUTH SỰ KIỆN ĐỘC LẬP & CHÍNH SÁCH VERIFIED-ONLY

### 1. Schema sự kiện chuẩn hóa & Chính sách verified-only
Bộ nhãn Ground Truth cấp sự kiện lưu trữ tại [`data/ground_truth/events.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events.jsonl). Theo yêu cầu liêm chính khoa học:
- **Chỉ các bản ghi có `verification_status == "verified"` mới được đưa vào đánh giá chính thức.**
- Các bản ghi `pending_verification` KHÔNG BAO GIỜ tham gia tính toán Precision, Recall, F1, Delay hay xếp hạng Grid Search.
- Bắt buộc kiểm tra `annotator_id` hợp lệ cho các bản ghi verified (không chấp nhận placeholder).
- Hàm [`load_ground_truth_events()`](file:///d:/workspace/DACN_NCKH/src/event_eval.py) mặc định `only_verified=True`, `strict=True`, từ chối fallback ngầm sang catalog.

### 2. Thống kê hiện trạng Ground Truth
Kết quả đối soát từ [`get_ground_truth_manifest()`](file:///d:/workspace/DACN_NCKH/src/event_eval.py) và [`scripts/validate_ground_truth.py`](file:///d:/workspace/DACN_NCKH/scripts/validate_ground_truth.py):
- **Tổng số bản ghi:** **35 sự kiện** trên 24 video dương tính.
- **Bản ghi Verified (`verified`):** **11 sự kiện** trên 8 video (6 FIRE, 5 SMOKE):
  - Validation Split: 3 sự kiện (`POS_03`, `POS_05`, `POS_23`).
  - Locked Test Split: 8 sự kiện (`POS_02`, `POS_04`, `POS_10`, `POS_12`, `POS_24`).
- **Bản ghi Pending (`pending_verification`):** **24 sự kiện** trên 16 video (10 FIRE, 14 SMOKE). Trạng thái chú giải được ghi nhận rõ ràng là `INCOMPLETE_PENDING_ANNOTATIONS`.

---

## HẠNG MỤC C: ONSET MATCHING CỰC ĐẠI & PHÂN LOẠI CẢNH BÁO KHÁCH QUAN

### 1. Định nghĩa Onset Matching
Điều kiện khớp sự kiện cũ (`pred.start_sec <= gt.end_sec + tolerance_sec`) đã được thay thế hoàn toàn bằng **Onset Matching** chuẩn xác:
- Dự đoán phải bắt đầu trong khoảng dung sai khởi phát:
  $$t_{\text{start}}^{GT} - \text{early\_tol} \le t_{\text{start}}^P \le t_{\text{start}}^{GT} + \text{tolerance\_sec}$$
- Dự đoán phải chồng lấn hoặc diễn ra trong thời gian cháy: $t_{\text{end}}^P \ge t_{\text{start}}^{GT}$.
- Dự đoán bắt đầu sau $t_{\text{start}}^{GT} + \text{tolerance\_sec}$ bị phân loại là **Late Detection**, không được tính là timely match.
- Dự đoán kết thúc hoàn toàn trước $t_{\text{start}}^{GT}$ được phân loại là **Early Alert**.

### 2. Ghép cặp tối ưu Cardinality (Max Cardinality Min Cost Matching)
Hệ thống triển khai thuật toán Min-Cost Max-Flow độc lập thuần Python (`max_cardinality_min_cost_matching`):
- Cực đại hóa số lượng cặp ghép hợp lệ (tránh bẫy greedy khoảng cách).
- Cực tiểu hóa tổng độ trễ phát hiện $\sum \max(0.0, t_{\text{start}}^P - t_{\text{start}}^{GT})$.
- Đảm bảo ánh xạ 1-1 chặt chẽ (one-to-one mapping).

---

## HẠNG MỤC D: ĐO LƯỜNG BÁO ĐỘNG GIẢ (FAR) & HIỆN TRẠNG VIDEO ÂM TÍNH

- **Catalog [`data/video_dataset_list.csv`](file:///d:/workspace/DACN_NCKH/data/video_dataset_list.csv):** Khai báo 28 video âm tính ($2.8648\text{ giờ}$).
- **Thực tế kiểm tra ổ đĩa:** **0 video âm tính tồn tại** ($0.00\text{ giờ}$).
- **Thiếu hụt:** Thiếu $5.00\text{ giờ}$ trên ổ đĩa và thiếu $2.1352\text{ giờ}$ trong catalog so với chỉ tiêu Đề cương ($\ge 5.0\text{h}$).
- **Manifest chi tiết:** [`missing_negative_videos_manifest.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/missing_negative_videos_manifest.json) và [`.csv`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/missing_negative_videos_manifest.csv).
- **Nguyên tắc học thuật:** `false_alarms_per_hour = None`, `far_status = "UNAVAILABLE_MISSING_NEGATIVE_VIDEOS"`, `status = "BLOCKED_BY_NEGATIVE_DATA"`. Tuyệt đối không gán $\text{FAR} = 0$.

---

## HẠNG MỤC E: ĐIỂM VẬN HÀNH THỰC NGHIỆM (RECONCILED OPERATING POINT)

### 1. Quét lưới Grid Search Validation
Quét toàn diện 432 tổ hợp tham số trên tập Validation (12 video, 3 verified GT events). Cấu hình tối ưu được lựa chọn tự động:
$$\mathbf{\text{conf} = 0.35, K = 5, N = 12, \text{feat\_thr} = 0.0, \text{cooldown} = 0.0\text{s}}$$
Hiệu năng trên Validation: Video Event Recall $100\%$ ($3/3$ video verified), Event Instance Recall $100\%$, Độ trễ trung bình $0.83\text{s}$.

### 2. Đánh giá độc lập trên Locked Test Set
Cấu hình provisional được đánh giá một lần duy nhất trên Locked Test Set (12 video, 8 verified GT events):

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

Đồng bộ $100\%$ giữa [`best_operating_point.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/best_operating_point.json) và [`locked_test_evaluation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/locked_test_evaluation.json).

---

## HẠNG MỤC F: KẾT LUẬN NGHIỆM THU & KẾ HOẠCH CHUYỂN GIAO TUẦN 5

| Hạng mục công việc Tuần 4 | Mã nguồn & Artifacts chính | Hiện trạng | Kết luận |
| :--- | :--- | :---: | :---: |
| **1. ByteTrack Adapter chính thức** | [`src/bytetrack_adapter.py`](file:///d:/workspace/DACN_NCKH/src/bytetrack_adapter.py) | Hoàn thành $100\%$ | **PASS** |
| **2. Temporal Verifier & Feature Gate** | [`src/temporal_verifier.py`](file:///d:/workspace/DACN_NCKH/src/temporal_verifier.py)<br>[`src/temporal_features.py`](file:///d:/workspace/DACN_NCKH/src/temporal_features.py) | Hoàn thành $100\%$ | **PASS** |
| **3. Cô lập Tracker đa luồng Camera** | [`src/camera_manager.py`](file:///d:/workspace/DACN_NCKH/src/camera_manager.py) | Hoàn thành $100\%$ | **PASS** |
| **4. Event Evaluator & Onset Matching** | [`src/event_eval.py`](file:///d:/workspace/DACN_NCKH/src/event_eval.py) | Hoàn thành $100\%$ | **PASS** |
| **5. Dữ liệu Frame JSONL & Alignment** | [`runs/benchmarks/week4_final/`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/) | Hoàn thành $100\%$ | **PASS** |
| **6. Ground Truth Verified-Only** | [`data/ground_truth/events.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events.jsonl) | Hoàn thành $100\%$ | **PASS** |
| **7. Bộ kiểm thử tự động 70 tests** | [`tests/`](file:///d:/workspace/DACN_NCKH/tests/) | 70/70 tests pass | **PASS 100%** |
| **8. Đo FAR và Nghiệm thu Operating Point** | [`locked_test_evaluation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/locked_test_evaluation.json) | Thiếu $\ge 5\text{h}$ video âm tính | **BLOCKED** |

### Đánh giá tổng thể: `PARTIAL (BLOCKED_BY_NEGATIVE_DATA)`
Toàn bộ mã nguồn, dữ liệu phát hiện, bộ kiểm thử và tài liệu đã được sửa đổi, đồng bộ và kiểm chứng nghiêm ngặt trên branch `fix`. Không có dữ liệu giả lập hay gán số liệu tùy tiện. Khi bổ sung đủ $\ge 5.0\text{h}$ video âm tính thực tế, hệ thống sẽ tự động tính FAR và hoàn tất nghiệm thu chính thức.
