# Báo Cáo Kiểm Toán Toàn Diện Repository & Đánh Giá Điều Kiện Bắt Đầu Tuần 4
*(Week 4 Readiness Audit & Scientific Progress Evaluation)*

- **Đề tài:** Nghiên cứu ứng dụng computer vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát
- **Chủ nhiệm đề tài:** Lê Minh Thiên
- **Thành viên chính:** Mai Quốc Việt
- **Giảng viên hướng dẫn:** ThS. Nguyễn Hữu Trung
- **Thời điểm kiểm toán:** 02/11/2026 21:30:00 (Đầu tuần 4)
- **Kỹ sư phụ trách kiểm toán:** Senior Computer Vision Engineer
- **Repository:** [https://github.com/vietmaiquoc2k5-max/DACN_NCKH](https://github.com/vietmaiquoc2k5-max/DACN_NCKH)

---

## 1. TỔNG QUAN CẤU TRÚC REPOSITORY

Repository đã được tổ chức lại theo kiến trúc module hóa phục vụ cả nghiên cứu thực nghiệm (benchmarking) lẫn triển khai hệ thống thực tế (deployment).

```text
D:\workspace\DACN_NCKH
├── app.py                             # GUI Desktop truyền thống (CustomTkinter, 1 luồng)
├── web_app.py                         # Web Dashboard 9 Camera Dark Mode PCCC (Streamlit)
├── alert_manager.py                   # Quản lý còi hú, lưu snapshot bằng chứng, SQLite
├── detector.py                        # Wrapper detector 1 luồng (tương thích ngược)
├── best.pt                            # Checkpoint YOLO chính thức (6.23 MB, smoke=0, fire=1)
├── yolov8n.pt                         # Trọng số tiền huấn luyện COCO (dùng khởi tạo train)
├── yolo26n.pt                         # Trọng số bổ trợ (5.54 MB)
├── data.yaml                          # Config dataset Roboflow tích hợp
├── data_dfire.yaml                    # Config dataset D-Fire chuẩn hóa (Train/Val/Test)
├── data_vn.yaml                       # Config dataset camera Việt Nam (1.450 ảnh 60/20/20)
├── data_legacy.yaml                   # Config dữ liệu cũ (lưu giữ đối chứng)
├── requirements.txt                   # Danh mục thư viện Python phụ thuộc
├── cai_dat_moi_truong.bat             # Batch script tự động tạo .venv & cài requirements
├── chay_giao_dien_web.bat             # Batch script khởi động Web Dashboard
├── chay_giao_dien_desktop.bat         # Batch script khởi động Desktop GUI
├── train_gpu.bat                      # Batch script chạy chuỗi huấn luyện trên GPU
├── src/                               # Gói module lõi của hệ thống đa luồng
│   ├── __init__.py
│   ├── config.py                      # Cấu hình 9 camera, 4 mode hoạt động, ẩn mật khẩu RTSP
│   ├── detector.py                    # SharedFireSmokeDetector: an toàn đa luồng, warmup, FP16
│   ├── temporal_verifier.py           # TemporalVerifier: cửa sổ k/N, IoU tracking, ROI, Cooldown
│   ├── camera_manager.py              # MultiCameraManager: tách 2 luồng đọc/suy luận, cô lập lỗi
│   ├── database.py                    # EventDatabase: SQLite, lọc đa chiều, xuất CSV
│   └── alert_manager.py               # Cầu nối quản lý cảnh báo trong package src
├── baselines/                         # Mã nguồn 2 mô hình đối chứng (Baseline B1 & B2)
│   ├── baseline_b1.py                 # B1: Heuristic luật màu HSV/YCbCr + trừ nền MOG2
│   ├── demo_b1_video.py               # Demo B1 trên luồng video
│   ├── evaluate_b1.py                 # Đánh giá định lượng B1
│   ├── baseline_b2_mobilenet.py       # B2: MobileNetV3-Small phân loại cấp khung hình
│   ├── train_b2.py                    # Script huấn luyện MobileNetV3 trên D-Fire
│   └── evaluate_b2.py                 # Đánh giá định lượng B2
├── datasets/                          # Thư mục chứa tập dữ liệu
│   └── vn_dataset/                    # 1.450 ảnh camera Việt Nam (870 Train / 290 Val / 290 Test)
│       ├── images/ [train, val, test]
│       └── labels/ [train, val, test]
├── FireSmokeLabeler/                  # Công cụ và dữ liệu gán nhãn
│   ├── videos/                        # 24 video dương tính (.avi) có sẵn trên đĩa
│   └── dataset/                       # Dữ liệu tự sinh từ best.pt (KHÔNG dùng làm GT)
├── runs/detect/runs/benchmarks/       # Checkpoints & logs huấn luyện thực tế MT2
│   ├── yolov8n_seed42/ [50 epochs]    # results.csv, args.yaml, best.pt (6.25 MB)
│   ├── yolov8n_seed123/ [50 epochs]   # results.csv, args.yaml, best.pt (6.49 MB)
│   ├── yolov8n_seed456/ [50 epochs]   # results.csv, args.yaml, best.pt (6.49 MB)
│   ├── yolo11n_seed123/ [50 epochs]   # results.csv, args.yaml, best.pt (5.49 MB)
│   └── yolo11n_seed456/ [50 epochs]   # results.csv, args.yaml, best.pt (5.49 MB)
├── scripts/                           # Tập hợp scripts tự động hóa
│   ├── split_dfire_val_test.py        # Phân tầng D-Fire 50/50 & làm sạch nhãn hỏng
│   ├── verify_dfire_integrity.py      # Kiểm băm MD5 toàn vẹn 21.527 ảnh D-Fire
│   ├── train_benchmarks.py            # Pipeline huấn luyện chuẩn IEEE 3 seeds
│   ├── evaluate_benchmarks.py         # Đánh giá mAP@0.5, mAP@0.5:0.95 tách riêng lớp
│   ├── verify_deployment_models.py    # Kiểm tra nạp & chạy đa luồng trên 4 camera
│   ├── extract_vn_frames.py           # Trích xuất & làm mờ mặt/biển số khung hình VN
│   ├── review_vn_labels.py            # Công cụ kiểm duyệt Ground Truth thủ công
│   ├── index_test_videos.py           # Đánh chỉ mục video kiểm thử
│   ├── index_vn_videos.py             # Đánh chỉ mục video Việt Nam
│   ├── tag_video_timestamps.py        # Gán mốc giây bắt đầu cháy
│   ├── auto_detect_timestamps.py      # AI dò mốc thời gian tự động
│   └── prepare_negative_dataset.py    # Giám sát thời lượng video âm tính
├── tests/                             # Bộ kiểm thử tự động
│   └── test_dashboard_system.py       # Kiểm thử 5 hạng mục hệ thống đa luồng & RTSP
├── test_integration.py                # Kiểm thử tích hợp toàn dự án
├── data/
│   └── video_dataset_list.csv         # Danh mục 72 video kiểm thử kèm mốc thời gian
└── docs/                              # Toàn bộ báo cáo khoa học & tài liệu kỹ thuật
```

---

## 2. BẢNG NGHIỆM THU TIẾN ĐỘ TUẦN 1 (19/10 – 25/10/2026)

| STT | Hạng mục Đề cương | Yêu cầu chi tiết | Hiện trạng kiểm toán thực tế | Đánh giá |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Khảo sát tài liệu** | Khảo sát bài toán nhận diện cháy/khói qua camera, các mô hình YOLOv8, YOLO11, RT-DETR, Baseline B1/B2, Temporal Verification và Domain Adaptation. | Đã biên soạn tài liệu khảo sát toàn diện chuẩn IEEE tại [`docs/literature_survey.md`](file:///d:/workspace/DACN_NCKH/docs/literature_survey.md) (8.5 KB). | **DONE** |
| 2 | **Chuẩn hóa D-Fire (MT1)** | Tải, chuẩn hóa format YOLO, phân tích số lượng ảnh, phân bố bounding box khói/lửa, số ảnh background. Đạt chuẩn $mAP \ge 0,75$. | - Đã có cấu hình [`data_dfire.yaml`](file:///d:/workspace/DACN_NCKH/data_dfire.yaml).<br>- Script kiểm định MD5 & làm sạch nhãn [`scripts/verify_dfire_integrity.py`](file:///d:/workspace/DACN_NCKH/scripts/verify_dfire_integrity.py).<br>- Báo cáo phân tích tại [`docs/dfire_analysis.md`](file:///d:/workspace/DACN_NCKH/docs/dfire_analysis.md) và [`docs/dfire_verification_report.md`](file:///d:/workspace/DACN_NCKH/docs/dfire_verification_report.md).<br>*(Lưu ý: thư mục vật lý `datasets/dfire` bị loại trừ khỏi Git theo `.gitignore`, không nằm trực tiếp trên workspace máy hiện tại)*. | **PARTIAL**<br>*(Code & Report DONE, Disk folder NOT PRESENT)* |
| 3 | **Danh sách nguồn video (MT1 & MT3)** | - $\ge 20$ video có cháy/khói (ghi mốc giây bắt đầu).<br>- $\ge 5$ giờ video âm tính (cảnh nhiễu không cháy). | - Đã lập danh mục 72 video tại [`data/video_dataset_list.csv`](file:///d:/workspace/DACN_NCKH/data/video_dataset_list.csv) gồm 44 video dương tính (2,30 giờ) $\to$ **Vượt chỉ tiêu $\ge 20$ video**.<br>- Video âm tính: Đã liệt kê 28 video (2,86 giờ) $\to$ **Còn thiếu ~2,14 giờ để đạt $\ge 5$ giờ**.<br>- Hiện trạng file trên đĩa: 24 video `.avi` dương tính có sẵn tại `FireSmokeLabeler/videos/`. File âm tính và video VN cần đồng bộ đường dẫn. | **PARTIAL** |
| 4 | **Chuẩn hóa Repository & Môi trường** | Git sạch, cấu trúc khoa học, script tự động, tài liệu hướng dẫn. | - Git clean trên nhánh `main`, `.gitignore` chuẩn.<br>- Đã có `cai_dat_moi_truong.bat`, `HUONG_DAN_CHAY.md`.<br>- Môi trường Python: Máy tính hiện tại có Python 3.11 hệ thống nhưng chưa kích hoạt `.venv` cục bộ. | **PARTIAL**<br>*(Repo sạch, cần kích hoạt `.venv`)* |

---

## 3. BẢNG NGHIỆM THU TIẾN ĐỘ TUẦN 2 (26/10 – 01/11/2026)

| STT | Hạng mục Đề cương | Yêu cầu chi tiết | Hiện trạng kiểm toán thực tế | Đánh giá |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Gán nhãn 1.000 khung hình Việt Nam (MT1, MT4)** | - $\ge 1.000$ khung hình camera thực tế Việt Nam.<br>- Chia 60/20/20 theo video độc lập.<br>- Làm mờ mặt người và biển số xe bảo mật.<br>- Ground Truth thủ công, không dùng nhãn tự sinh. | - Đã lưu trữ trực tiếp trên Git tại [`datasets/vn_dataset/`](file:///d:/workspace/DACN_NCKH/datasets/vn_dataset/):<br>  • Train: 870 ảnh ($60,0\%$).<br>  • Val: 290 ảnh ($20,0\%$).<br>  • Test: 290 ảnh ($20,0\%$).<br>  • **Tổng cộng: 1.450 ảnh** (vượt chỉ tiêu $\ge 1.000$ ảnh).<br>- Đã làm mờ mặt và biển số qua [`scripts/extract_vn_frames.py`](file:///d:/workspace/DACN_NCKH/scripts/extract_vn_frames.py).<br>- Tách biệt 100% theo 23 video độc lập, cấu hình tại [`data_vn.yaml`](file:///d:/workspace/DACN_NCKH/data_vn.yaml). | **DONE** |
| 2 | **Mốc thời gian video kiểm thử (MT3)** | Gán mốc giây bắt đầu có khói / cháy (`smoke_start_sec`, `fire_start_sec`). | Đã điền đầy đủ mốc giây cho toàn bộ 44 video dương tính trong [`data/video_dataset_list.csv`](file:///d:/workspace/DACN_NCKH/data/video_dataset_list.csv) bằng công cụ [`scripts/auto_detect_timestamps.py`](file:///d:/workspace/DACN_NCKH/scripts/auto_detect_timestamps.py) kết hợp tinh chỉnh thủ công. | **DONE** |
| 3 | **Cài đặt Baseline B1 (MT2)** | - Heuristic luật màu HSV/YCbCr kết hợp trừ nền MOG2.<br>- Đo đạc tốc độ xử lý (FPS) và độ chính xác mAP. | - Đã cài đặt hoàn chỉnh tại [`baselines/baseline_b1.py`](file:///d:/workspace/DACN_NCKH/baselines/baseline_b1.py).<br>- Đã đo đạc thực tế: tốc độ **45.7 – 60.3 FPS** trên CPU (độ trễ 16.6 – 21.8 ms).<br>- Báo cáo khoa học chi tiết tại [`docs/baseline_b1_report.md`](file:///d:/workspace/DACN_NCKH/docs/baseline_b1_report.md) ghi nhận độ nhạy ánh sáng cao và tỷ lệ báo giả khung hình ~75%. | **DONE** |

---

## 4. BẢNG NGHIỆM THU TIẾN ĐỘ TUẦN 3 (02/11 – 08/11/2026 - MỐC KIỂM TRA 1)

| STT | Hạng mục Đề cương | Yêu cầu chi tiết | Hiện trạng kiểm toán thực tế | Đánh giá |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Kiểm định toàn vẹn D-Fire (MT1/MT2)** | Chia phân tầng Stratified 50/50: Train (17.221), Val (2.153), Test (2.153). Kiểm băm MD5 21.527 ảnh, loại bỏ box suy biến. | Đã hoàn tất kiểm định thực nghiệm ngày 29/10/2026: 0 ảnh trùng lặp MD5, 0 box tràn viền, làm sạch 26 box suy biến. Kết quả lưu tại [`docs/dfire_verification_report.md`](file:///d:/workspace/DACN_NCKH/docs/dfire_verification_report.md). | **DONE** |
| 2 | **Cài đặt Baseline B2 (MT2)** | - Mạng MobileNetV3-Small phân loại khung hình.<br>- Đo đạc tốc độ CPU, F1-Score, tỷ lệ báo giả FPR. | - Mã nguồn: [`baselines/baseline_b2_mobilenet.py`](file:///d:/workspace/DACN_NCKH/baselines/baseline_b2_mobilenet.py), [`baselines/evaluate_b2.py`](file:///d:/workspace/DACN_NCKH/baselines/evaluate_b2.py).<br>- Đo đạc tốc độ CPU thực tế: **72.8 FPS (13.73 ms/frame)**.<br>- Báo cáo: [`docs/baseline_b2_report.md`](file:///d:/workspace/DACN_NCKH/docs/baseline_b2_report.md).<br>- **Vấn đề tồn đọng:** Mô hình chưa lưu checkpoint đã huấn luyện (`models/mobilenetv3_b2.pt`), đánh giá trên 200 ảnh test với random head (Fire Recall = 0%). | **PARTIAL**<br>*(Code & Tốc độ DONE, Checkpoint trained NOT DONE)* |
| 3 | **Pipeline huấn luyện chuẩn (MT2 Benchmark)** | `imgsz=640`, cùng augmentation, 3 seed ngẫu nhiên (42, 123, 456) theo chuẩn IEEE. | Pipeline chuẩn hóa đã xây dựng tại [`scripts/train_benchmarks.py`](file:///d:/workspace/DACN_NCKH/scripts/train_benchmarks.py) và kịch bản thực thi [`train_gpu.bat`](file:///d:/workspace/DACN_NCKH/train_gpu.bat). | **DONE** |
| 4 | **Huấn luyện các bộ phát hiện (MT2: 15 runs)** | $\ge 4$ kiến trúc (YOLOv8n/s, YOLO11n/s, RT-DETR-R18) $\times$ 3 seeds. Đạt chỉ tiêu $mAP@0.5 \ge 0,75$ trên D-Fire. | **Đã hoàn thành 5/15 runs (33.3%):**<br>- `YOLOv8n` (3/3 seeds: 42, 123, 456): Đã chạy đủ 50 epochs.<br>  • **Val mAP@0.5 = 76.62% ± 0.19%** (Vượt $\ge 0.75$).<br>  • **Test mAP@0.5 = 76.74% ± 0.19%** (Vượt $\ge 0.75$).<br>- `YOLO11n` (2/3 seeds: 123, 456): Đã chạy đủ 50 epochs.<br>  • **Val mAP@0.5 = 75.94% ± 0.20%** (Vượt $\ge 0.75$).<br>  • **Test mAP@0.5 = 76.83% ± 0.08%** (Vượt $\ge 0.75$).<br>- Còn thiếu: `yolo11n_seed42`, `yolov8s` (3 seeds), `yolo11s` (3 seeds), `rtdetr-l` (3 seeds). | **PARTIAL**<br>*(Đạt chất lượng mAP, hoàn tất 5/15 runs)* |
| 5 | **Đánh giá đối sánh & Báo cáo MT2** | Đo mAP@0.5, mAP@0.5:0.95, Latency, FPS. Tách riêng Precision/Recall cho Khói (0) và Lửa (1). | - Script: [`scripts/evaluate_benchmarks.py`](file:///d:/workspace/DACN_NCKH/scripts/evaluate_benchmarks.py).<br>- Dữ liệu cache: [`docs/benchmark_eval_cache.json`](file:///d:/workspace/DACN_NCKH/docs/benchmark_eval_cache.json).<br>- Báo cáo khoa học: [`docs/benchmark_detectors_dfire.md`](file:///d:/workspace/DACN_NCKH/docs/benchmark_detectors_dfire.md). | **DONE** |
| 6 | **Kiểm thử triển khai đa luồng thực tế (MT5 & Mục 10)** | Nạp checkpoint thật, chạy 4 camera đồng thời, đo FPS đọc / FPS YOLO, kiểm tra lọc báo giả và ẩn mật khẩu RTSP. | - Đã kiểm thử thực tế trên 4 camera với GPU RTX 3050 Laptop.<br>- Đạt tốc độ: FPS đọc 23.7 FPS/luồng, FPS suy luận YOLO 66 - 110 FPS.<br>- Báo cáo và bằng chứng ảnh chụp tại [`docs/deployment_verification_report.md`](file:///d:/workspace/DACN_NCKH/docs/deployment_verification_report.md) và thư mục `docs/deployment_evidence/`. | **DONE** |

---

## 5. DANH SÁCH CHECKPOINT VÀ BENCHMARK ĐÃ XÁC MINH

### 5.1. Bảng xác minh kỹ thuật các tệp checkpoint (.pt) trên đĩa

| Checkpoint | Đường dẫn tệp | Dung lượng | Mã băm MD5 | Số Epochs | Tình trạng xác minh |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **yolov8n_seed123** *(Khuyến nghị)* | `runs/detect/runs/benchmarks/yolov8n_seed123/weights/best.pt` | 6.49 MB | `c640564009cd11a21fd612912a9d7884` | 50 | **VERIFIED (mAP cao nhất, 104.3 FPS GPU)** |
| **yolov8n_seed42** | `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt` | 6.25 MB | `292acfbd662bc43eb924320de19b4d0a` | 50 | **VERIFIED (110.3 FPS GPU)** |
| **yolov8n_seed456** | `runs/detect/runs/benchmarks/yolov8n_seed456/weights/best.pt` | 6.49 MB | `71d17094c691e5cb131ad9daa126fefd` | 50 | **VERIFIED (103.6 FPS GPU)** |
| **yolo11n_seed123** | `runs/detect/runs/benchmarks/yolo11n_seed123/weights/best.pt` | 5.49 MB | `681aaa144a37fb85974da09d9caa2c6f` | 50 | **VERIFIED (Smoke mAP 84.71%, 69.4 FPS GPU)** |
| **yolo11n_seed456** | `runs/detect/runs/benchmarks/yolo11n_seed456/weights/best.pt` | 5.49 MB | `7443d429f128307e2b3a6fce542c4569` | 50 | **VERIFIED (66.5 FPS GPU)** |
| **Root best.pt** | `best.pt` | 6.23 MB | `52c041957a0a81069a20a139b1bd8409` | N/A | **VERIFIED (Đang phục vụ Dashboard, class 0: smoke, 1: fire)** |
| **yolov8n.pt** | `yolov8n.pt` | 6.55 MB | `95a2449609c73cd69a072b09daaff0cc` | Pretrained | **VERIFIED (Model gốc COCO - Detector từ chối đúng quy tắc)** |
| **yolo26n.pt** | `yolo26n.pt` | 5.54 MB | `cf3cca69f04cf639bafdeb2644bd0843` | N/A | **VERIFIED** |
| **mobilenetv3_b2.pt** | `models/mobilenetv3_b2.pt` | 0 MB | N/A | 0 | ❌ **NOT FOUND (Chưa lưu trọng số đã train)** |

### 5.2. Số liệu Benchmark đối sánh thực tế trên D-Fire

| Mô hình / Lượt chạy | Params | Val mAP@0.5 | Test mAP@0.5 | Khói (mAP50 Val/Test) | Lửa (mAP50 Val/Test) | GPU Latency / FPS | Chỉ tiêu MT2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n_seed42** | 3.01M | 76.51% | 76.71% | 83.31% / 82.63% | 69.70% / 70.80% | 5.45 ms (183.3 FPS) | ✅ Đạt |
| **yolov8n_seed123** | 3.14M | **76.84%** | **76.94%** | 82.81% / 83.25% | **70.86%** / **70.62%** | 4.97 ms (201.1 FPS) | ✅ Đạt (Tốt nhất) |
| **yolov8n_seed456** | 3.14M | 76.52% | 76.57% | 82.06% / 83.39% | 70.97% / 69.75% | 4.13 ms (242.2 FPS) | ✅ Đạt |
| **Trung bình YOLOV8N** | **3.01M** | **76.62% ± 0.19%** | **76.74% ± 0.19%** | **82.73% / 83.09%** | **70.51% / 70.39%** | — | ✅ **Đạt chuẩn IEEE** |
| **yolo11n_seed123** | 2.61M | 76.08% | 76.88% | 81.92% / **84.71%** | 70.23% / 69.04% | 2.76 ms (362.6 FPS) | ✅ Đạt |
| **yolo11n_seed456** | 2.61M | 75.80% | 76.77% | 81.72% / 83.99% | 69.88% / 69.56% | 2.74 ms (365.4 FPS) | ✅ Đạt |
| **Trung bình YOLO11N** | **2.61M** | **75.94% ± 0.20%** | **76.83% ± 0.08%** | **81.82% / 84.35%** | **70.06% / 69.30%** | — | ✅ **Đạt chuẩn IEEE** |
| **Baseline B1 (Luật màu)** | 0 | ~3.0% (Box thô) | ~4.1% (Recall) | F1: 0.0008 | F1: 0.0022 | 21.87 ms (45.7 FPS CPU) | Mốc đối chứng |
| **Baseline B2 (MobileNetV3)** | 1.08M | N/A (Không box) | N/A (Không box) | F1: 53.81% | F1: 0.00% | 13.73 ms (72.8 FPS CPU) | Mốc đối chứng |

---

## 6. CÁC VẤN ĐỀ CẦN KHẮC PHỤC TRƯỚC VÀ TRONG TUẦN 4

Qua kiểm toán kỹ thuật chi tiết, phát hiện **5 vấn đề cốt lõi** cần xử lý:

1. **Khởi tạo môi trường ảo Python (`.venv`) trên máy hiện tại:**
   - Máy kiểm toán đang có Python 3.11 hệ thống nhưng chưa tạo thư mục `.venv`.
   - Cần chạy `cai_dat_moi_truong.bat` hoặc lệnh `python -m venv .venv` và cài đặt `requirements.txt` để sẵn sàng chạy các script kiểm thử.
2. **Quy tụ và hợp nhất đường dẫn video kiểm định (`data/video_dataset_list.csv`):**
   - Danh mục CSV trỏ đường dẫn tới `datasets/test_videos/pos/` và `datasets/test_videos/neg/`.
   - Trong khi đó, 24 video dương tính thực tế đang nằm tại thư mục `FireSmokeLabeler/videos/`.
   - Các file video âm tính và video Việt Nam cần được chép/tạo liên kết tượng trưng (symlink) vào đúng cấu trúc thư mục để `event_eval.py` có thể tự động duyệt qua.
3. **Bổ sung thời lượng video âm tính đạt chuẩn MT1 ($\ge 5$ giờ):**
   - Hiện danh mục mới có 2,86 giờ video âm tính.
   - Cần bổ sung thêm ~2,14 giờ video âm tính (camera cố định phố xá, xe cộ ban đêm có đèn pha, nhà xưởng ban ngày) theo đúng cam kết trong đề cương. Các video nguồn đã có sẵn trong máy (`NHIỄU 1.mp4`, `NHIỄU 5.mp4` tại thư mục Downloads).
4. **Nâng cấp thuật toán trong `src/temporal_verifier.py` theo đúng đề cương:**
   - Hiện tại verifier đang sử dụng thuật toán ghép cặp IoU cơ bản. Cần nâng cấp tích hợp logic theo dõi ổn định (ByteTrack) và **phân tích biến động hình học diện tích ($\Delta \text{Area}$)** và tâm hộp bao để triệt tiêu đốm sáng tĩnh (đèn đường, biển báo phản quang).
5. **Huấn luyện bổ sung và lưu checkpoint Baseline B2:**
   - Cần chạy `baselines/train_b2.py` để lưu tệp trọng số `models/mobilenetv3_b2.pt` thực sự hội tụ trên 2 lớp khói/lửa, hoàn thiện biên bản đối chứng khoa học.

---

## 7. DANH SÁCH TỆP CẦN TẠO HOẶC CHỈNH SỬA TRONG TUẦN 4

| Tác vụ | Tệp liên quan | Thao tác | Mục đích kỹ thuật |
| :---: | :--- | :---: | :--- |
| 1 | `src/event_eval.py` | **TẠO MỚI** | Xây dựng công cụ đánh giá sự kiện tự động theo mốc thời gian: đo Event Recall, Detection Latency, False Alarms/Hour (MT3). |
| 2 | `src/temporal_verifier.py` | **CHỈNH SỬA** | Bổ sung module phân tích động học $\Delta \text{Area}$, chu vi và tích hợp ByteTrack tracker để lọc nguồn sáng tĩnh. |
| 3 | `scripts/grid_search_temporal.py` | **TẠO MỚI** | Quét tự động các bộ tham số $(k, N)$ (ví dụ: $6/10$, $8/12$, $10/15$) và ngưỡng diện tích trên tập video kiểm định. |
| 4 | `data/video_dataset_list.csv` | **CHỈNH SỬA** | Cập nhật đường dẫn tệp thực tế và bổ sung các video âm tính mới để đạt tổng thời lượng $\ge 5,0$ giờ. |
| 5 | `docs/temporal_verification_report.md` | **TẠO MỚI** | Báo cáo khoa học tổng kết thực nghiệm Tuần 4 đối sánh: Báo động từng khung (Frame-by-frame) vs. Xác minh theo thời gian (MT3). |
| 6 | `baselines/train_b2.py` | **THỰC THI** | Huấn luyện và xuất checkpoint `models/mobilenetv3_b2.pt` hoàn chỉnh. |

---

## 8. KẾT LUẬN KIỂM TOÁN VÀ QUYẾT ĐỊNH ĐIỀU KIỆN BẮT ĐẦU TUẦN 4

### Quyết định: 🟡 **CONDITIONAL GO (ĐỦ ĐIỀU KIỆN CÓ ĐIỀU KIỆN)**

### Luận cứ kỹ thuật:
1. **Điều kiện ĐẠT (GO Factors):**
   - Đã có **5 checkpoint mô hình phát hiện** được huấn luyện chuẩn mực 50 epochs trên D-Fire với $mAP@0.5 > 76\%$ (vượt chỉ tiêu $\ge 0,75$ của MT2).
   - Checkpoint `yolov8n_seed123` và `yolov8n_seed42` đã được nạp và chạy kiểm thử đa luồng thành công trên 4 luồng camera đồng thời với tốc độ suy luận đạt **66 – 110 FPS GPU**, đáp ứng hoàn hảo yêu cầu làm detector lõi cho Tuần 4.
   - Tập 1.450 khung hình Việt Nam gán nhãn thủ công và làm mờ bảo mật đã sẵn sàng 100% trên Git.
   - Bộ khung xác minh chuỗi thời gian `src/temporal_verifier.py` (cửa sổ $k/N$, ROI, Cooldown, 2 mức SUSPECT/FIRE) đã được tích hợp hoàn chỉnh và kiểm thử với CSDL SQLite.
2. **Điều kiện CẦN KHẮC PHỤC NGAY (Conditions to clear):**
   - **Điều kiện 1:** Khởi tạo `.venv` và cài đặt `requirements.txt` trên môi trường làm việc hiện tại để chạy được các script đánh giá.
   - **Điều kiện 2:** Đồng bộ đường dẫn thư mục `datasets/test_videos/pos/` trỏ tới 24 video có sẵn tại `FireSmokeLabeler/videos/` và đưa các file video âm tính vào thư mục tương ứng.
   - **Điều kiện 3:** Bổ sung ~2,14 giờ video âm tính vào danh mục `video_dataset_list.csv` để đảm bảo khi chạy `event_eval.py` sẽ nghiệm thu trọn vẹn chỉ tiêu MT1 ($\ge 5$ giờ âm tính).

> **Khuyến nghị kỹ sư:** Sau khi người dùng xem xét báo cáo này và phê duyệt, tiến hành bước chuẩn bị dữ liệu video và môi trường trước khi bước vào viết code cho `src/event_eval.py` và tối ưu `src/temporal_verifier.py`.
