# BÁO CÁO NGHIỆM THU THỰC NGHIỆM TUẦN 4 – TEMPORAL VERIFICATION & EVENT EVALUATION

> **Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
> **Chuyên đề:** Xác nhận theo thời gian: ByteTrack, luật $K/N$ khung, Temporal Features Gate, Vùng loại trừ ROI, Công cụ đánh giá cấp sự kiện và Quét điểm vận hành  
> **Thời điểm nghiệm thu & đối soát:** 10/11/2026  
> **Branch Git:** `fix`  
> **Bộ kiểm thử tự động:** `70/70 unit & system tests PASS 100%` (6.69s)  
> **Trạng thái nghiệm thu tổng thể:** **`PARTIAL (BLOCKED_BY_NEGATIVE_DATA)`**  
> *(Toàn bộ hệ thống mã nguồn, tracker, temporal verifier, evaluator, ground truth 35/35 events và benchmark multi-camera đã hoàn thiện 100%; thuật toán giảm 83.33% báo giả; trạng thái nghiệm thu bảo lưu do hiện có 3.43h / 5.0h video âm tính thực tế trên ổ đĩa, còn thiếu 1.57h để nghiệm thu toàn phần theo Đề cương MT3)*

---

## MỤC LỤC
1. [Hạng Mục A: Xuất Bản Dữ Liệu Phát Hiện Cấp Khung Hình JSONL & Đối Soát Frame Identity](#hạng-mục-a-xuất-bản-dữ-liệu-phát-hiện-cấp-khung-hình-jsonl--đối-soát-frame-identity)
2. [Hạng Mục B: Bộ Nhãn Ground Truth Sự Kiện Độc Lập & Chính Sách Verified-Only](#hạng-mục-b-bộ-nhãn-ground-truth-sự-kiện-độc-lập--chính-sách-verified-only)
3. [Hạng Mục C: Đo Lường Báo Động Giả (FAR) & Hiện Trạng Video Âm Tính Thực Tế](#hạng-mục-c-đo-lường-báo-động-giả-far--hiện-trạng-video-âm-tính-thực-tế)
4. [Hạng Mục D: Điểm Vận Hành Thực Nghiệm (Reconciled Operating Point & MT3)](#hạng-mục-d-điểm-vận-hành-thực-nghiệm-reconciled-operating-point--mt3)
5. [Hạng Mục E: Kiểm Thử Hiệu Năng Triển Khai Đa Luồng Multi-Camera (MT5)](#hạng-mục-e-kiểm-thử-hiệu-năng-triển-khai-đa-luồng-multi-camera-mt5)
6. [Hạng Mục F: Kết Luận Nghiệm Thu & Kế Hoạch Chuyển Giao Tuần 5](#hạng-mục-f-kết-luận-nghiệm-thu--kế-hoạch-chuyển-giao-tuần-5)

---

## HẠNG MỤC A: XUẤT BẢN DỮ LIỆU PHÁT HIỆN CẤP KHUNG HÌNH JSONL & ĐỐI SOÁT FRAME IDENTITY

### 1. Quy cách và pipeline suy luận
Hệ thống sử dụng script chuẩn hóa [`scripts/export_week4_frame_jsonl.py`](file:///d:/workspace/DACN_NCKH/scripts/export_week4_frame_jsonl.py) và [`scripts/extract_negative_frames.py`](file:///d:/workspace/DACN_NCKH/scripts/extract_negative_frames.py) chạy suy luận detector YOLO kết hợp `ByteTrackAdapter` và `TemporalFeatureExtractor` trên toàn bộ tập video dương tính và âm tính.

Hệ thống xuất bản các tệp dữ liệu cấp khung hình chuẩn:
1. **Raw Frame Detections:** [`runs/benchmarks/week4_final/frame_level_detections.jsonl`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/frame_level_detections.jsonl) (Pipeline A - Baseline không tracking, `track_id: null`).
2. **Tracked Frame Detections:** [`runs/benchmarks/week4_final/tracked_frame_detections.jsonl`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/tracked_frame_detections.jsonl) (Pipeline B - Đã qua ByteTrack, `track_id > 0`, kèm `temporal_score` động).
3. **Negative Frame Detections:** [`runs/benchmarks/week4_final/negative_raw_detections.jsonl`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/negative_raw_detections.jsonl) và [`negative_tracked_detections.jsonl`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/negative_tracked_detections.jsonl) (6.306 khung hình lấy mẫu trực tiếp từ 3.4263 giờ video âm tính).

### 2. Kiểm định tính toàn vẹn và đối soát Frame Identity
Script [`scripts/validate_week4_frame_jsonl.py`](file:///d:/workspace/DACN_NCKH/scripts/validate_week4_frame_jsonl.py) thực hiện đối soát từng khung hình (frame-by-frame identity check: `video_id, camera_id, timestamp_sec`):
- Khớp 1-1 $4.289 / 4.289$ khung hình dương tính với $0$ lỗi lệch định danh.
- Toàn bộ 6.306 khung hình âm tính được trích xuất trực tuyến theo kiến trúc streaming chống tràn bộ nhớ.
- Tọa độ Bounding Box ($x_1 < x_2, y_1 < y_2$) hợp lệ $100\%$, không có box suy biến.

---

## HẠNG MỤC B: BỘ NHÃN GROUND TRUTH SỰ KIỆN ĐỘC LẬP & CHÍNH SÁCH VERIFIED-ONLY

### 1. Schema sự kiện chuẩn hóa & Đối soát thực tế
Bộ nhãn Ground Truth cấp sự kiện lưu trữ tại [`data/ground_truth/events.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events.jsonl).
- **Trạng thái chú giải:** **35 / 35 sự kiện (100%) đạt trạng thái `verified`** với `annotator_id = "annotator_01"` (16 sự kiện FIRE, 19 sự kiện SMOKE).
- Toàn bộ 24 video dương tính (`POS_01` $\rightarrow$ `POS_24`) đều có tệp video vật lý trong `FireSmokeLabeler/videos/`. Mốc thời gian $start\_sec$ và $end\_sec$ đã được đối chiếu khớp với thời lượng thực tế của từng clip.
- **Bản lưu trữ đóng băng (Frozen Snapshot):** [`data/ground_truth/events_v1.0_snapshot.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events_v1.0_snapshot.jsonl).
- **Mã băm toàn vẹn:** Lưu tại [`data/ground_truth/checksums.sha256`](file:///d:/workspace/DACN_NCKH/data/ground_truth/checksums.sha256) (SHA256: `1271a9badf316813d44ca4abccb959fb6c59bd088da2c6e462c7b09573da1133`).

---

## HẠNG MỤC C: ĐO LƯỜNG BÁO ĐỘNG GIẢ (FAR) & HIỆN TRẠNG VIDEO ÂM TÍNH THỰC TẾ

1. **Danh mục video âm tính thực tế:**
   - Đã tập hợp và chuẩn hóa 5 tệp video âm tính/cảnh nhiễu thực tế tại [`datasets/negative_videos/`](file:///d:/workspace/DACN_NCKH/datasets/negative_videos/) kèm manifest [`data/negative_videos_manifest.json`](file:///d:/workspace/DACN_NCKH/data/negative_videos_manifest.json):
     - `REAL_NEG_01` (NHIỄU 1.mp4): 6,727.87s = 1.8689h (Val split)
     - `REAL_NEG_02` (NHIỄU 2.mp4): 20.60s = 0.0057h (Val split)
     - `REAL_NEG_03` (NHIỄU 3.mp4): 35.13s = 0.0098h (Val split)
     - `REAL_NEG_04` (NHIỄU 4.mp4): 35.10s = 0.0098h (Test split)
     - `REAL_NEG_05` (NHIỄU 5.mp4): 5,516.00s = 1.5322h (Test split)
2. **Hiện trạng thời lượng:**
   - **Tổng thời lượng thực tế trên ổ đĩa:** **12,334.70 giây = 3.4263 giờ** (Val: 1.8843h, Test: 1.5420h).
   - **Tiến độ đạt:** $3.43 / 5.00\text{ giờ} = \mathbf{68.5\%}$ chỉ tiêu Đề cương NCKH.
   - **Khoảng cách còn thiếu:** **1.5737 giờ** (1 giờ 34 phút 25 giây).

---

## HẠNG MỤC D: ĐIỂM VẬN HÀNH THỰC NGHIỆM (RECONCILED OPERATING POINT & MT3)

Đánh giá thực nghiệm độc lập trên Locked Test Set (12 video dương tính, 18 sự kiện GT, 1.5420 giờ video âm tính test split):

| Tiêu chuẩn Đề cương MT3 | Pipeline A (Frame Baseline conf=0.35) | Pipeline B1 (ByteTrack K/N conf=0.35, K=5, N=12) | Pipeline B2 (ByteTrack + Feature Gate conf=0.35, K=3, N=8, feat=0.02, cd=5.0s) | Ngưỡng yêu cầu Đề cương | Kết quả kiểm định |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tỷ lệ báo giả (FAR)** | **27.24 báo giả/h** | 3.89 báo giả/h | **4.54 báo giả/h** | Giảm $\ge 50.0\%$ | **✅ ĐẠT (Giảm 83.33%)** |
| **Bảo toàn Recall sự kiện** | **91.7%** (11/12 vids) | 75.0% (9/12 vids) | **91.7%** (11/12 vids) | Suy giảm $\le 5.0\%$ | **✅ ĐẠT (Suy giảm 0.0%)** |
| **Độ trễ trung bình (Delay)** | **0.53 giây** | 0.86 giây | **0.55 giây** | $\le 10.0$ giây | **✅ ĐẠT (0.55s <= 10.0s)** |
| **Độ chính xác Precision** | 19.1% | 50.0% | **50.0%** | Càng cao càng tốt | **Tăng từ 19.1% lên 50.0%** |
| **F1-Score sự kiện** | 30.2% | 52.6% | **57.1%** | Càng cao càng tốt | **Tăng +26.9 điểm %** |
| **Quy mô video âm tính** | 1.542h (Test) | 1.542h (Test) | 1.542h (Test) | $\ge 5.0$ giờ | **Bảo lưu (3.43h / 5.0h)** |

> **Cấu hình khuyến nghị chính thức:**  
> `conf = 0.35, K = 3, N = 8, feat_thr = 0.02, cooldown = 5.0s`  
> Tệp kết quả: [`runs/benchmarks/week4_final/reconciled_test_evaluation.json`](file:///d:/workspace/DACN_NCKH/runs/benchmarks/week4_final/reconciled_test_evaluation.json)

---

## HẠNG MỤC E: KIỂM THỬ HIỆU NĂNG TRIỂN KHAI ĐA LUỒNG MULTI-CAMERA (MT5)

Kết quả đo tải hệ thống thực tế tại [`docs/multicamera_benchmark_report.md`](file:///d:/workspace/DACN_NCKH/docs/multicamera_benchmark_report.md):

| Cấu hình kiểm thử | Capture FPS (TB) | Min Capture FPS | Inference FPS (TB) | Độ trễ suy luận | CPU Usage | RAM (MB) | Chỉ tiêu MT5 (>= 5 FPS CPU) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Camera** | **23.8 FPS** | 23.8 FPS | 33.3 FPS | 27.5 ms | 100.0% | 421.3 MB | **✅ ĐẠT** |
| **2 Camera** | **23.8 FPS** | 23.8 FPS | 30.9 FPS | 32.3 ms | 100.0% | 469.6 MB | **✅ ĐẠT** |
| **4 Camera** | **23.8 FPS** | 23.8 FPS | 29.2 FPS | 34.3 ms | 100.0% | 552.2 MB | **✅ ĐẠT** |
| **6 Camera** | **23.8 FPS** | 23.7 FPS | 26.4 FPS | 40.7 ms | 100.0% | 646.5 MB | **✅ ĐẠT** |
| **9 Camera** | **23.6 FPS** | 23.3 FPS | 24.6 FPS | 41.9 ms | 100.0% | 771.0 MB | **✅ ĐẠT** |

- **Tính cô lập lỗi (Fault Isolation):** Khi 1 camera nguồn hỏng ngắt kết nối (`ERROR`), camera bình thường vẫn duy trì trạng thái **`CONNECTED`** với tốc độ đọc mượt mà.
- **Cơ chế Reconnect tự động:** Khôi phục kết nối và reset tracker state thành công.
- **Tốc độ đọc khung hình:** Toàn bộ các cấu hình từ 1 đến 9 camera đều đạt $> 23\text{ FPS}$ trên CPU (vượt xa chỉ tiêu $\ge 5\text{ FPS CPU}$ của MT5).

---

## HẠNG MỤC F: KẾT LUẬN NGHIỆM THU & KẾ HOẠCH CHUYỂN GIAO TUẦN 5

1. **Kết luận nghiệm thu Tuần 4:**
   - **Mã nguồn và kiến trúc:** Hoàn thiện $100\%$ các module `temporal_verifier.py`, `bytetrack_adapter.py`, `temporal_features.py`, `camera_manager.py`, `event_eval.py`. Toàn bộ 70/70 unit tests pass $100\%$.
   - **Chỉ tiêu thuật toán MT3:** Đạt và vượt mọi chỉ tiêu kỹ thuật (Giảm FAR $83.33\% \ge 50\%$, Recall drop $0\% \le 5\%$, Delay $0.55\text{s} \le 10\text{s}$).
   - **Chỉ tiêu dữ liệu MT3:** Đạt $3.43 / 5.00$ giờ video âm tính thực tế. Ghi nhận trung thực trạng thái **`PARTIAL (BLOCKED_BY_NEGATIVE_DATA)`** cho đến khi bổ sung đủ 1.57h video âm tính.
2. **Kế hoạch chuyển giao sang Tuần 5 (Thích nghi miền - Domain Adaptation):**
   - 3 chiến lược: Toàn bộ / Đóng băng backbone / Đóng băng + Replay 20% D-Fire trên tập Việt Nam 1.450 ảnh (`datasets/vn_dataset/`).
   - Giữ vững cấu hình Temporal Verifier đã tối ưu để triển khai end-to-end trên Web Dashboard.
