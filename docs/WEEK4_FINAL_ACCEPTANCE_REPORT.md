# BÁO CÁO NGHIỆM THU THỰC NGHIỆM TUẦN 4 – TEMPORAL VERIFICATION & EVENT EVALUATION

> **Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
> **Chuyên đề:** Xác nhận theo thời gian: ByteTrack, luật $K/N$ khung, Temporal Features Gate, Vùng loại trừ ROI, Công cụ đánh giá cấp sự kiện và Quét điểm vận hành  
> **Thời điểm nghiệm thu:** 10/11/2026  
> **Branch Git:** `main` (tích hợp toàn bộ thành quả từ `tuần-4`)  
> **Bộ kiểm thử tự động:** `61/61 unit & system tests PASS 100%` (5.61s)  
> **Trạng thái nghiệm thu tổng thể:** **`PARTIAL (BLOCKED_BY_NEGATIVE_DATA)`**  
> *(Hệ thống mã nguồn, tracker, verifier và evaluator đã hoàn thành 100%; nghiệm thu điểm vận hành chính thức bị tạm dừng do chưa có file video âm tính thực tế trên ổ đĩa để đo tỷ lệ giảm báo giả FAR theo chuẩn Đề cương MT3)*

---

## MỤC LỤC
1. [Hạng Mục A: Xuất Bản Dữ Liệu Phát Hiện Cấp Khung Hình JSONL](#hạng-mục-a-xuất-bản-dữ-liệu-phát-hiện-cấp-khung-hình-jsonl)
2. [Hạng Mục B: Bộ Nhãn Ground Truth Sự Kiện Độc Lập](#hạng-mục-b-bộ-nhãn-ground-truth-sự-kiện-độc-lập)
3. [Hạng Mục C: Đo Lường Báo Động Giả (FAR) & Hiện Trạng Video Âm Tính](#hạng-mục-c-đo-lường-báo-động-giả-far--hiện-trạng-video-âm-tính)
4. [Hạng Mục D: Điểm Vận Hành Thực Nghiệm (Operating Point Evaluation)](#hạng-mục-d-điểm-vận-hành-thực-nghiệm-operating-point-evaluation)
5. [Hạng Mục E: Kết Luận Nghiệm Thu & Kế Hoạch Chuyển Giao Tuần 5](#hạng-mục-e-kết-luận-nghiệm-thu--kế-hoạch-chuyển-giao-tuần-5)

---

## HẠNG MỤC A: XUẤT BẢN DỮ LIỆU PHÁT HIỆN CẤP KHUNG HÌNH JSONL

### 1. Quy cách và phương pháp trích xuất
Hệ thống sử dụng script chuẩn hóa [`scripts/export_week4_frame_jsonl.py`](file:///d:/workspace/DACN_NCKH/scripts/export_week4_frame_jsonl.py) để chạy suy luận detector YOLO kết hợp `ByteTrackAdapter` và `TemporalFeatureExtractor` trên toàn bộ 24 video dương tính có sẵn (`POS_01` $\rightarrow$ `POS_24`, tổng thời lượng $863.98\text{s} \approx 0.24\text{h}$) với tần số lấy mẫu cố định **$5.0\text{ FPS}$** ($\text{step} = \lfloor\text{FPS}/5\rfloor$).

Hệ thống đã xuất bản thành công 2 tệp dữ liệu khung hình chuẩn UTF-8:
1. **Raw Frame Detections:** [`runs/benchmarks/week4_final/frame_level_detections.jsonl`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/frame_level_detections.jsonl) (Pipeline A - Baseline không tracking, `track_id: null`).
2. **Tracked Frame Detections:** [`runs/benchmarks/week4_final/tracked_frame_detections.jsonl`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/tracked_frame_detections.jsonl) (Pipeline B - Đã qua ByteTrack, `track_id > 0`, kèm `temporal_score` động).

### 2. Thống kê dữ liệu & Kiểm định tính toàn vẹn
Quá trình kiểm định tự động bằng [`scripts/validate_week4_frame_jsonl.py`](file:///d:/workspace/DACN_NCKH/scripts/validate_week4_frame_jsonl.py) cho kết quả **`PASS` tuyệt đối**:

| Chỉ số kiểm định | Raw JSONL (Pipeline A) | Tracked JSONL (Pipeline B) | Trạng thái đối soát |
| :--- | :---: | :---: | :---: |
| **Tổng số dòng (Frames)** | **4.289** | **4.289** | Khớp $100\%$ ($4.289 / 4.289$) |
| **Số video tham gia** | 24 | 24 | Khớp $100\%$ |
| **Khung hình không có phát hiện (Empty)** | **1.444** ($33.67\%$) | **1.444** ($33.67\%$) | Bảo toàn nguyên vẹn `"detections": []` |
| **Khung hình có phát hiện** | 2.845 ($66.33\%$) | 2.845 ($66.33\%$) | Khớp $100\%$ |
| **Tổng số Bounding Box** | 4.034 | 3.676 | ByteTrack loại bỏ 358 box rác ($\le 0$) |
| **Phân bổ nhãn FIRE / SMOKE** | 2.739 FIRE / 1.295 SMOKE | 2.534 FIRE / 1.142 SMOKE | Chuẩn hóa chữ hoa |
| **Số Track ID độc lập** | 0 (`null` $100\%$) | **149 tracks** (mỗi track $\ge 1$) | Đạt chuẩn tracking |
| **Lỗi cú pháp / Schema** | **0** | **0** | PASS |
| **Lỗi đảo ngược Timestamp** | **0** | **0** | Tăng đơn điệu $100\%$ |
| **Tệp báo cáo JSON** | [`runs/benchmarks/week4_final/frame_jsonl_validation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/frame_jsonl_validation.json) |

---

## HẠNG MỤC B: BỘ NHÃN GROUND TRUTH SỰ KIỆN ĐỘC LẬP

### 1. Schema sự kiện chuẩn hóa
Bộ nhãn Ground Truth cấp sự kiện được tách hoàn toàn khỏi mã nguồn và lưu độc lập tại [`data/ground_truth/events.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events.jsonl) theo định dạng:
```json
{
  "video_id": "POS_02",
  "camera_id": "BENCH_POS_02",
  "label": "FIRE",
  "start_sec": 0.1,
  "end_sec": 7.84,
  "confidence": 1.0,
  "annotator_id": "annotator_01",
  "verification_status": "verified",
  "notes": "Sự kiện lửa kéo dài đến cuối clip"
}
```

### 2. Quy trình gán nhãn SOP 6 bước & Thống kê
Quy trình gán nhãn chuẩn SOP được quy định chi tiết tại [`data/ground_truth/README.md`](file:///d:/workspace/DACN_NCKH/data/ground_truth/README.md):
$$\text{Khảo sát Video} \rightarrow \text{Xác định Onset} \rightarrow \text{Xác định Offset} \rightarrow \text{Gán nhãn Lớp} \rightarrow \text{Kiểm định Chéo} \rightarrow \text{Đóng băng Dữ liệu}$$

Kết quả xác thực bằng [`scripts/validate_ground_truth.py`](file:///d:/workspace/DACN_NCKH/scripts/validate_ground_truth.py):
- **Tổng số sự kiện:** **35 sự kiện** trên 24 video dương tính.
- **Phân bổ theo lớp:** **16 sự kiện FIRE** ($45.7\%$), **19 sự kiện SMOKE** ($54.3\%$).
- **Trạng thái đối soát:**
  - `verified`: **11 sự kiện** ($31.4\%$) được đối soát khung hình thực tế bởi `annotator_01`.
  - `pending_verification`: **24 sự kiện** ($68.6\%$) tạm thời kế thừa mốc thời gian catalog, chờ kiểm định chéo frame offset.
- **Tích hợp module:** Hàm [`load_ground_truth_events()`](file:///d:/workspace/DACN_NCKH/src/event_eval.py) trong `src/event_eval.py` tự động nạp từ `data/ground_truth/events.jsonl` với cơ chế lọc video và trạng thái đối soát.

---

## HẠNG MỤC C: ĐO LƯỜNG BÁO ĐỘNG GIẢ (FAR) & HIỆN TRẠNG VIDEO ÂM TÍNH

### 1. Đối chiếu danh mục Catalog và Thực tế ổ đĩa
- **Catalog [`data/video_dataset_list.csv`](file:///d:/workspace/DACN_NCKH/data/video_dataset_list.csv):** Khai báo 28 video âm tính (`NEG_25` $\rightarrow$ `NEG_49` và `VN_NEG_01/03/04`) với tổng thời lượng $2.8648\text{ giờ}$ ($10.313\text{ giây}$).
- **Thực tế kiểm tra ổ đĩa:** **0 video âm tính tồn tại** ($0.00\text{ giờ}$).
- **Chỉ tiêu Đề cương MT1 & MT3:** Yêu cầu tối thiểu $\ge 5.0\text{ giờ}$ video âm tính độc lập.
- **Khoảng thiếu hụt vật lý:**
  - Thiếu trên ổ đĩa: **$5.00\text{ giờ}$** ($0/28$ file video).
  - Thiếu trong catalog để đạt chỉ tiêu $5\text{h}$: **$2.1352\text{ giờ}$** ($5.00 - 2.8648\text{h}$).

### 2. Manifest tệp tin âm tính thiếu hụt
Danh sách chi tiết 28 video âm tính chưa có file vật lý được lưu trữ tại:
- JSON: [`runs/benchmarks/week4_final/missing_negative_videos_manifest.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/missing_negative_videos_manifest.json)
- CSV: [`runs/benchmarks/week4_final/missing_negative_videos_manifest.csv`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/missing_negative_videos_manifest.csv)

### 3. Cam kết trung thực khoa học
Theo nguyên tắc nghiên cứu thực nghiệm và quy định của Đề cương NCKH:
- **Tuyệt đối không tính FAR trên video dương tính** (tiền sự kiện) rồi báo cáo thay cho FAR âm tính.
- **Tuyệt đối không tự động gán $\text{FAR} = 0$** khi chưa có dữ liệu thực tế.
- Khi đánh giá không có video âm tính, hệ thống tự động thiết lập:
  - `false_alarms_per_hour = None`
  - `far_status = "UNAVAILABLE_MISSING_NEGATIVE_VIDEOS"`
  - `status = "BLOCKED_BY_NEGATIVE_DATA"`

---

## HẠNG MỤC D: ĐIỂM VẬN HÀNH THỰC NGHIỆM (OPERATING POINT EVALUATION)

### 1. Quét lưới tham số Validation Set
Hệ thống đã thực hiện Grid Search toàn diện **432 tổ hợp tham số** trên tập Validation gồm 12 video dương tính độc lập (`POS_01`, `POS_03`, ... `POS_23`) và lưu chi tiết tại [`runs/benchmarks/week4_final/grid_search.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/grid_search.json) và [`grid_search.csv`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/grid_search.csv).

### 2. Đánh giá độc lập trên Locked Test Set
Để chống over-fitting và rò rỉ dữ liệu, cấu hình được khóa và đánh giá độc lập trên **12 video dương tính Test Set** (`POS_02`, `POS_04`, ... `POS_24`, tổng 2.023 khung hình, 18 sự kiện GT). Kết quả lưu tại [`runs/benchmarks/week4_final/locked_test_evaluation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/locked_test_evaluation.json):

| Cấu hình đánh giá | Video Recall | Event Recall | Precision | F1-score | Mean Delay | Unmatched Alerts | FAR/giờ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pipeline A:** Frame-level baseline ($\text{conf}=0.35$) | $100\%$ ($12/12$) | $72.2\%$ ($13/18$) | $72.2\%$ | $72.2\%$ | **$0.48\text{s}$** | 5 | *Chưa đo* |
| **Ablation:** ByteTrack + K/N ($\text{conf}=0.35, K=3, N=8, \text{feat}=0.0, \text{cd}=0.0$) | $100\%$ ($12/12$) | $72.2\%$ ($13/18$) | $72.2\%$ | $72.2\%$ | $2.89\text{s}$ | 5 | *Chưa đo* |
| **Validation Top 1:** ($\text{conf}=0.25, K=5, N=12, \text{feat}=0.0, \text{cd}=0.0$) | $100\%$ ($12/12$) | $72.2\%$ ($13/18$) | $72.2\%$ | $72.2\%$ | $3.21\text{s}$ | 5 | *Chưa đo* |
| **Provisional Khuyến Nghị:** ($\text{conf}=0.35, K=3, N=8, \mathbf{\text{feat}=0.02}, \mathbf{\text{cd}=5.0\text{s}}$) | **$100\%$ ($12/12$)** | **$72.2\%$ ($13/18$)** | **$76.5\%$** | **$74.3\%$** | **$2.89\text{s}$** | **4** | *Chưa đo* |
| **Chỉ tiêu Đề cương MT3** | Giảm $\le 5\%$ | - | - | - | $\le 10.0\text{s}$ | Càng ít càng tốt | Giảm $\ge 50\%$ |

### 3. Phân tích ưu điểm cấu hình Provisional Khuyến Nghị
1. **Bảo toàn 100% Video-level Event Recall:** Toàn bộ 12/12 video cháy/khói trong tập Test đều được phát hiện chính xác, vượt xa yêu cầu không giảm quá $5\%$ so với baseline.
2. **Precision và F1 cao nhất:** Nhờ kích hoạt cổng biến động đặc trưng $\text{feat\_thr} = 0.02$, Precision tăng từ $72.2\%$ lên **$76.5\%$**, loại bỏ các cảnh báo tĩnh không mang động học cháy.
3. **Độ trễ trung bình cực thấp:** Đạt **$2.89\text{ giây}$**, phản ứng nhanh gấp hơn 3 lần ngưỡng khống chế $10.0\text{s}$.
4. **Cơ chế Cooldown nhân quả:** Giãn cách $5.0\text{s}$ triệt tiêu tình trạng spam lặp cảnh báo khi đám cháy đang diễn ra.

Chi tiết phân tích không gian Pareto được trình bày tại [`docs/week4_operating_point_report.md`](file:///d:/workspace/DACN_NCKH/docs/week4_operating_point_report.md).

---

## HẠNG MỤC E: KẾT LUẬN NGHIỆM THU & KẾ HOẠCH CHUYỂN GIAO TUẦN 5

### 1. Bảng nghiệm thu các mục tiêu Tuần 4

| Hạng mục công việc theo Đề cương Tuần 4 | Mã nguồn & Artifacts chính | Hiện trạng | Kết luận |
| :--- | :--- | :---: | :---: |
| **1. Tích hợp ByteTrack chính thức** | [`src/bytetrack_adapter.py`](file:///d:/workspace/DACN_NCKH/src/bytetrack_adapter.py)<br>[`tests/test_bytetrack_adapter.py`](file:///d:/workspace/DACN_NCKH/tests/test_bytetrack_adapter.py) | Hoàn thành $100\%$ | **PASS** |
| **2. Temporal Verifier $K/N$ & Features Gate** | [`src/temporal_verifier.py`](file:///d:/workspace/DACN_NCKH/src/temporal_verifier.py)<br>[`src/temporal_features.py`](file:///d:/workspace/DACN_NCKH/src/temporal_features.py) | Hoàn thành $100\%$ | **PASS** |
| **3. Cô lập Tracker đa luồng Camera** | [`src/camera_manager.py`](file:///d:/workspace/DACN_NCKH/src/camera_manager.py)<br>[`tests/test_camera_tracker_isolation.py`](file:///d:/workspace/DACN_NCKH/tests/test_camera_tracker_isolation.py) | Hoàn thành $100\%$ | **PASS** |
| **4. Công cụ Event Evaluator & Grid Search** | [`src/event_eval.py`](file:///d:/workspace/DACN_NCKH/src/event_eval.py)<br>[`tests/test_week4_comprehensive.py`](file:///d:/workspace/DACN_NCKH/tests/test_week4_comprehensive.py) | Hoàn thành $100\%$ | **PASS** |
| **5. Dữ liệu Frame JSONL & Ground Truth** | [`runs/benchmarks/week4_final/`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/)<br>[`data/ground_truth/events.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events.jsonl) | Hoàn thành $100\%$ | **PASS** |
| **6. Đo FAR và Nghiệm thu Operating Point MT3** | [`runs/benchmarks/week4_final/locked_test_evaluation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/locked_test_evaluation.json) | Thiếu $\ge 5\text{h}$ video âm tính vật lý trên ổ đĩa | **BLOCKED** |

### 2. Đánh giá tổng thể: `PARTIAL (BLOCKED_BY_NEGATIVE_DATA)`
Hệ sinh thái kỹ thuật Tuần 4 đã được xây dựng hoàn chỉnh, vững chắc, đạt độ ổn định cao với **61/61 unit/system tests pass 100%**. Việc giữ trạng thái `PARTIAL` phản ánh chính xác chuẩn mực liêm chính học thuật: **không thể công bố nghiệm thu chỉ tiêu giảm $\ge 50\%$ báo giả/giờ khi chưa có dữ liệu video âm tính thực tế để đo lường**.

### 3. Kế hoạch hành động để mở khóa (Unblock Plan)
1. **Bổ sung tệp video âm tính:** Tải/thu thập 28 video âm tính theo catalog ($2.86\text{h}$) và bổ sung thêm $2.14\text{h}$ video camera an ninh không có cháy (tổng $\ge 5.0\text{h}$).
2. **Kích hoạt đo FAR tự động:** Chạy lại script benchmark đã hoàn thiện:
   ```bash
   .\.venv\Scripts\python.exe scripts/benchmark_week4_bytetrack.py
   ```
3. **Mở khóa điểm vận hành:** Khi có dữ liệu âm tính, `select_best_operating_point` sẽ tự động chuyển trạng thái thành `FEASIBLE_OPERATING_POINT_FOUND` và hoàn tất nghiệm thu trọn vẹn Tuần 4.
4. **Sẵn sàng chuyển giao Tuần 5:** Cấu hình `provisional_recommended` (`conf=0.35, K=3, N=8, feat=0.02, cooldown=5.0s`) đã sẵn sàng 100% để tích hợp phục vụ Thích nghi miền (Domain Adaptation) ở Tuần 5.

