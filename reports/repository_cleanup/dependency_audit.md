# BÁO CÁO PHÂN TÍCH PHỤ THUỘC MÃ NGUỒN VÀ RỦI RO DỌN DẸP REPOSITORY
**Dự án:** DACN_NCKH — Hệ thống Phát hiện & Cảnh báo Cháy/Khói Thông minh (FireWatch)  
**Tài liệu tham chiếu:** `DeCuong_NCKH_PhatHienChay_Camera.docx`  
**Nhánh Git:** `fix-4` | **Commit HEAD:** `d4b3bdca7c6fce964b06355303fd13f1ea42071c`  
**Ngày thực hiện:** 17/11/2026  
**Công cụ phân tích:** `scripts/audit_repository_size.py` & AST/Regex Dependency Scanner  

---

## 1. TỔNG QUAN HỆ THỐNG VÀ PHƯƠNG PHÁP PHÂN TÍCH

Quá trình kiểm toán phụ thuộc (Dependency Analysis) được thực hiện trên toàn bộ **35.369 tệp tin** đang được Git theo dõi (tổng dung lượng **1.991,27 MB ~ 1,945 GB**).
Mục tiêu là xác định chính xác các thành phần mã nguồn, cấu hình, dữ liệu, checkpoint nào đang được các module hoạt động (`src/`, `web_app.py`, `app.py`, `train.py`, `detector.py`, `scripts/`, `tests/`) nhập khẩu hoặc truy cập động trong thời gian chạy (runtime), nhằm đảm bảo:
1. Không làm gián đoạn bất kỳ chức năng nào của hệ thống đang hoạt động.
2. Bảo toàn 100% tính tái lập khoa học cho các mục tiêu Đề cương NCKH (MT1, MT2, MT3, MT4, MT5).
3. Đề xuất phương án giảm tải dung lượng Git hiệu quả nhất mà không gây mất mát dữ liệu đĩa cục bộ.

---

## 2. PHÂN LOẠI 5 NHÓM ỨNG VIÊN DỌN DẸP

### 2.1. Nhóm 1: `KEEP_TRACKED` (Bắt buộc duy trì theo dõi trên Git)
- **Số lượng tệp:** 9.274 tệp
- **Dung lượng:** 122,77 MB (6,17% dung lượng Git)
- **Thành phần cốt lõi:**
  1. **Mã nguồn chức năng & Test suite:**
     - Toàn bộ thư mục `src/` (31 tệp, 0,35 MB): `camera_manager.py`, `detector.py`, `config.py`, `bytetrack_adapter.py`, `temporal_verifier.py`, `temporal_features.py`, `database.py`, `alert_manager.py`, `event_eval.py`, `domain_adaptation.py`...
     - Toàn bộ thư mục `tests/` (48 tệp, 0,37 MB): 106 unit & system tests đảm bảo chất lượng hệ thống (pass 100%).
     - Các ứng dụng chính tại root: `web_app.py` (Flask Dashboard), `app.py` (Desktop GUI), `detector.py`, `train.py`, `evaluate_model.py`, `alert_manager.py`.
     - Thư mục `scripts/` (86 tệp, 0,99 MB): Các công cụ benchmark, đánh giá FAR, trích xuất dữ liệu, kiểm toán.
     - Thư mục `configs/` (10 tệp, 0,02 MB): Cấu hình benchmark và camera.
     - Thư mục `docs/` (107 tệp, 2,40 MB): Toàn bộ báo cáo nghiệm thu Tuần 1–4, IEEE survey, kiến trúc hệ thống.
     - Thư mục `data/` (22 tệp, 1,36 MB): Danh mục video kiểm định `video_dataset_list.csv` (192 video), metadata video.
     - Đề cương gốc: `DeCuong_NCKH_PhatHienChay_Camera.docx` (0,30 MB).
     - Âm thanh cảnh báo: `alarm.wav` (1,96 MB) do `alert_manager.py` phát khi có cháy.
  2. **Trọng số mô hình cốt lõi (Core Model Weights):**
     - `best.pt` (root, 5,94 MB, SHA256: `b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f`): Mô hình suy luận chính thức phục vụ Web Dashboard và Desktop GUI.
     - `yolov8n.pt` (root, 6,25 MB, SHA256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`): Mô hình nền tảng COCO phục vụ huấn luyện và baseline.
     - `runs/detect/runs/benchmarks/*/weights/best.pt` (5 tệp, tổng 28,8 MB): Bằng chứng thực nghiệm khoa học bắt buộc của MT2 (huấn luyện 3 seed độc lập: `yolov8n` seed 42/123/456, `yolo11n` seed 123/456).
  3. **Metadata và Báo cáo Benchmark Khoa học:**
     - Các tệp kết quả JSON, CSV tổng hợp hiệu năng (`results.csv`, `metrics.json`, `summary.json`) chứng minh MT2 (mAP@0.5 >= 0.75) và MT3 (FAR giảm 57.39%).

---

### 2.2. Nhóm 2: `UNTRACK_KEEP_LOCAL` (Hủy theo dõi Git, giữ nguyên trên đĩa cục bộ)
- **Số lượng tệp:** 12.005 tệp
- **Dung lượng:** 946,75 MB (47,54% dung lượng Git)
- **Thành phần chi tiết:**
  1. **Thư mục `uploads/` (6 tệp video, 221,00 MB):**
     - Các video `VN_POS_04.mp4`, `VN_POS_06.mp4`, `VN_POS_09.mp4`, `VN_POS_11.mp4`, `VN_POS_15.mp4`, `VN_POS_20.mp4`.
     - *Phân tích phụ thuộc:* Được hàm `get_available_test_videos()` trong `src/config.py` quét động từ thư mục đĩa cục bộ `uploads/`. Không yêu cầu phải track trong Git.
     - *Rủi ro nếu untrack:* Rủi ro THẤP (LOW). Giữ file trên đĩa cục bộ giúp giao diện web vẫn quét thấy video kiểm thử, nhưng khi clone trên máy khác thì thư mục trống (hoặc chỉ chứa `.gitkeep`), người dùng tải video qua web UI.
  2. **Thư mục `FireSmokeLabeler/videos/` (24 tệp video .avi, 215,98 MB):**
     - Các video AVI phục vụ gán nhãn giai đoạn đầu (`testpos02.818.avi`, `posVideo5.874.avi`, v.v.).
     - *Phân tích phụ thuộc:* Được tham chiếu bởi một số script kiểm thử cũ, nhưng không nằm trong bộ 79 video âm tính hoặc 104 video dương tính chuẩn hóa của Tuần 4.
     - *Rủi ro nếu untrack:* Rủi ro THẤP (LOW). Giữ nguyên trên ổ đĩa để chạy các công cụ legacy khi cần.
  3. **Thư mục `FireSmokeLabeler/dataset/` (7.077 tệp, 122,94 MB):**
     - Chứa 3.538 ảnh và 3.539 tệp txt nhãn tự động sinh ra bởi `best.pt` trong các tuần đầu.
     - *Ràng buộc đề cương:* Đề cương NCKH ghi rõ: "Nhãn trong FireSmokeLabeler/dataset do chính best.pt tự sinh: không dùng làm ground truth để đánh giá hay báo cáo".
     - *Rủi ro nếu untrack:* Rủi ro TRUNG BÌNH (MEDIUM). Cấu hình `data.yaml` cũ trỏ tới thư mục này. Tuy nhiên, quy trình huấn luyện MT2 và MT4 hiện tại dùng `data_vn.yaml` và `data_dfire.yaml`. Khi giữ trên đĩa cục bộ (`git rm --cached`), file không mất, mã nguồn chạy bình thường.
  4. **Thư mục `dataset/` (15.161 tệp, 300,06 MB):**
     - Chứa 7.407 ảnh và 7.407 tệp txt nhãn trong `dataset/images/train/images` và `dataset/images/train/labels`.
     - *Phân tích phụ thuộc:* Đây là tập dữ liệu Roboflow cũ với nhãn bị đảo ngược (0: fire, 1: smoke) theo `data_legacy.yaml`.
     - *Rủi ro nếu untrack:* Rủi ro THẤP (LOW). Không sử dụng trong bất kỳ pipeline huấn luyện chính thức nào của Tuần 4/5. Giữ nguyên trên đĩa cục bộ.
  5. **Các tệp vết thực thi lớn trong `runs/benchmarks/` (~86,77 MB):**
     - Điển hình: các tệp timeline thô `*.stdout` (như `VN_NEG_47/timeline.stdout` 11,3 MB, `VN_NEG_20/timeline.stdout` 6,3 MB) và các tệp log chi tiết từng frame.
     - *Rủi ro nếu untrack:* Rủi ro THẤP (LOW). Các báo cáo tổng hợp JSON và CSV đã ghi lại đầy đủ chỉ số khoa học.

---

### 2.3. Nhóm 3: `EXTERNAL_STORAGE_REQUIRED` (Lưu trữ ngoài, tách biệt khỏi Git)
- **Dung lượng trên đĩa cục bộ:** ~25,58 GB (đang untracked)
- **Thành phần:**
  1. `datasets/negative_videos/`: 22,60 GB (79 video âm tính thực tế, 7,20 giờ, 929.939 frames phục vụ nghiệm thu MT3 Tuần 4).
  2. `datasets/dfire_author_linked_kaggle_v1/`: 2,97 GB (Tập dữ liệu D-Fire tải từ Kaggle phục vụ MT1 & MT2).
- **Hiện trạng:** Đã nằm ngoài Git tracking (untracked). Cần duy trì quy tắc `.gitignore` để đảm bảo không ai vô tình commit thêm video âm tính vào Git.

---

### 2.4. Nhóm 4: `DELETE_CANDIDATE` (Ứng viên xóa bỏ an toàn)
- **Số lượng tệp:** 6 tệp
- **Dung lượng:** 0,61 MB
- **Thành phần:**
  - `runs/benchmarks/ground_truth_studio/studio_review_log.json` (bản nháp review studio cũ).
  - `runs/benchmarks/week4_dataset_recovery/run_20261113_v2/manifest_audit.json` (bản audit trung gian).
  - `week4_recovery_toolkit/test_run.log` (log chạy thử toolkit).
  - `handoff/week5/gpu_pilot.log` (log chạy thử GPU handoff).
- **Rủi ro:** CỰC THẤP (LOW). Có thể xóa an toàn mà không ảnh hưởng mã nguồn.

---

### 2.5. Nhóm 5: `MANUAL_REVIEW` (Cần đánh giá và quyết định kỹ lưỡng)
- **Số lượng tệp:** 14.084 tệp
- **Dung lượng:** 921,14 MB (46,26% dung lượng Git)
- **Thành phần trọng yếu:**
  1. **Tập dữ liệu khung hình Việt Nam `datasets/vn_dataset/` (6.244 tệp, 894,98 MB):**
     - Bao gồm: 3.121 ảnh JPEG và 3.121 tệp TXT nhãn (chia train 1.868, val 623, test 630).
     - *Ý nghĩa khoa học:* Đây là tập dữ liệu thực nghiệm then chốt của Đề cương NCKH (MT1 yêu cầu >= 1.000 khung hình Việt Nam gán nhãn tay; thực tế dự án đã chuẩn bị 3.121 khung hình có làm mờ mặt và biển số xe).
     - *Mâu thuẫn kỹ thuật:* Nếu tiếp tục track trong Git, dung lượng repo vẫn xấp xỉ 1 GB. Nếu untrack, người clone mới sẽ không có sẵn ảnh để chạy ngay lệnh huấn luyện Domain Adaptation (Tuần 5).
     - *Khuyến nghị kiến trúc:*
       - **Phương án A (Bảo toàn nguyên trạng - Khuyến nghị):** Tiếp tục `KEEP_TRACKED` trong Git để đảm bảo tính sẵn sàng cao nhất cho nhóm nghiên cứu trong quá trình hoàn thiện Tuần 5.
       - **Phương án B (Tối ưu kho lưu trữ):** Giữ tệp nhãn (`.txt`) và cấu hình `data_vn.yaml` trong Git (chỉ vài MB), nén 3.121 ảnh thành tệp lưu trữ phát hành (`vn_dataset_images.tar.gz` ~850 MB) đưa lên GitHub Releases hoặc Google Drive/Kaggle, và cung cấp script tải/giải nén tự động `scripts/download_vn_dataset.py`.
  2. **Gói chuyển giao Tuần 5 `handoff/week5/` (2.368 tệp, 23,54 MB):**
     - Chứa các manifest và tập con dữ liệu phục vụ huấn luyện máy GPU thứ 2.
     - Tệp `handoff/week5/dataset_manifest.json` chiếm 10,42 MB.
     - Cần giữ lại trên Git nhánh `fix-4` vì vừa được tạo để hỗ trợ đồng đội chạy GPU ngoài.
  3. **Trọng số COCO dư thừa `yolo26n.pt` (root, 5,29 MB):**
     - Kiểm tra metadata cho thấy đây là checkpoint tiền huấn luyện trên 80 lớp COCO (`person`, `car`...), không được bất kỳ script huấn luyện hay suy luận nào trong repo tham chiếu tới.
     - Khuyến nghị: Có thể untrack hoặc xóa an toàn.

---

## 3. BẢNG MA TRẬN PHỤ THUỘC CHI TIẾT (COMPONENT DEPENDENCY MATRIX)

| Thành phần đường dẫn | Dung lượng | Tệp | Trạng thái Git | Tiêu thụ bởi (Consumers) | Đề xuất hành động | Mức rủi ro |
|---|---|---|---|---|---|---|
| `src/` | 0,35 MB | 31 | Tracked | Toàn bộ hệ thống, GUI, Web Dashboard, Scripts | `KEEP_TRACKED` | CRITICAL |
| `best.pt` (root) | 5,94 MB | 1 | Tracked | `web_app.py`, `app.py`, `detector.py`, benchmarks | `KEEP_TRACKED` | CRITICAL |
| `yolov8n.pt` (root) | 6,25 MB | 1 | Tracked | `train.py`, baseline scripts, benchmark seeds | `KEEP_TRACKED` | HIGH |
| `yolo26n.pt` (root) | 5,29 MB | 1 | Tracked | Không có consumer nào trong repo | `UNTRACK_KEEP_LOCAL` | LOW |
| `alarm.wav` (root) | 1,96 MB | 1 | Tracked | `alert_manager.py` (âm báo động PCCC) | `KEEP_TRACKED` | MEDIUM |
| `runs/detect/runs/benchmarks/*/best.pt` | 28,8 MB | 5 | Tracked | Báo cáo MT2 (3-seed evaluation), paper IEEE | `KEEP_TRACKED` | HIGH |
| `uploads/*.mp4` | 221,00 MB | 6 | Tracked | `src/config.py` (quét video test), Web Dashboard | `UNTRACK_KEEP_LOCAL` | LOW |
| `FireSmokeLabeler/videos/*.avi` | 215,98 MB | 24 | Tracked | Legacy test runner, initial video pool | `UNTRACK_KEEP_LOCAL` | LOW |
| `FireSmokeLabeler/dataset/` | 122,94 MB | 7.077 | Tracked | `data.yaml` (legacy), không dùng cho MT2/MT4 | `UNTRACK_KEEP_LOCAL` | MEDIUM |
| `dataset/` (Roboflow legacy) | 300,06 MB | 15.161 | Tracked | `data_legacy.yaml` (nhãn ngược, không dùng) | `UNTRACK_KEEP_LOCAL` | LOW |
| `runs/benchmarks/**/*.stdout` | 55,18 MB | 174 | Tracked | Trace log gỡ lỗi thực thi Tuần 4 | `UNTRACK_KEEP_LOCAL` | LOW |
| `datasets/vn_dataset/` | 894,98 MB | 6.244 | Tracked | `data_vn.yaml`, thực nghiệm MT4 Tuần 5 | `MANUAL_REVIEW` | HIGH |
| `handoff/week5/` | 23,54 MB | 2.368 | Tracked | Handoff chạy GPU Tuần 5 | `KEEP_TRACKED` | MEDIUM |

---

## 4. TỔNG HỢP HIỆU QUẢ DỰ KIẾN KHI DỌN DẸP

Nếu thực thi kế hoạch untrack an toàn cho các thư mục không cần thiết trên Git (`uploads/`, `FireSmokeLabeler/`, `dataset/`, và các file trace `.stdout`):
- **Số lượng tệp Git theo dõi:** Giảm từ **35.369 tệp** xuống còn **~11.000 tệp** (giảm **~69% số lượng tệp**).
- **Dung lượng cây Git hiện tại:** Giảm từ **1.991,27 MB** xuống còn **~1.076 MB** (tiết kiệm **~915 MB** ngay lập tức trên working tree).
- **Dữ liệu trên ổ cứng cục bộ:** Không suy giảm 1 byte nào; toàn bộ video, nhãn và checkpoint vẫn nằm nguyên vị trí trên đĩa.

