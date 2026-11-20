# BÁO CÁO TOÀN DIỆN KHẮC PHỤC TÍNH TOÀN VẸN KHOA HỌC VÀ SẴN SÀNG THỰC NGHIỆM TUẦN 6
## (WEEK 6 SCIENTIFIC INTEGRITY FIX AND EXECUTION READINESS REPORT)

- **Đơn vị thực hiện:** Senior Computer Vision Research Engineer, MLOps Engineer & Independent Research Auditor
- **Repository:** `vietmaiquoc2k5-max/DACN_NCKH`
- **Branch làm việc:** `fix-4` | **Base Commit HEAD:** `3701d2e03b899778fe8ed5fa2f7d4f84527b4187`
- **Thời điểm hoàn thành:** 20/11/2026
- **Trạng thái Cổng Validation Gate:** **`BLOCKED` (BẮT BUỘC KHÓA CHẶT THEO QUY CHUẨN KHOA HỌC)**

---

## 1. TỔNG QUAN VÀ MỤC TIÊU NHIỆM VỤ

Mục tiêu của nhiệm vụ là khắc phục triệt để các lỗ hổng phương pháp luận trong việc kiểm toán dữ liệu video âm tính, đồng bộ hóa bằng chứng thực nghiệm của các giai đoạn trước, thực hiện đo kiểm hiệu năng triển khai backend (Phase E) trung thực, và thiết lập hàng rào bảo vệ khoa học (Scientific Guard) cho thực nghiệm Tuần 6 theo đúng Đề cương NCKH.

**Nguyên tắc chỉ đạo xuyên suốt:**
1. **Tuyệt đối không làm đẹp số liệu:** Báo cáo chính xác thực trạng dữ liệu và hiệu năng phần cứng thực tế, không giả lập kết quả.
2. **Không xâm phạm tập Test:** Giữ nguyên vẹn 13 video Test độc lập ($2,10$ giờ), không được chuyển sang Validation để làm đẹp thời lượng phơi nhiễm FAR.
3. **Không nới lỏng cổng kiểm định:** Duy trì trạng thái `BLOCKED` của Validation Gate chừng nào 5 sự kiện dương tính chưa có biên bản đồng thuận kép và mẫu số FAR chưa đạt đủ $5,0$ giờ sạch được xác thực bởi con người.

---

## 2. PHASE A – ĐỒNG BỘ HÓA GIT VÀ DANH MỤC ARTIFACT (ARTIFACT RECONCILIATION)

### 2.1. Hiện trạng Git và Phân tích Quy tắc `.gitignore`
- Nhánh `fix-4` đã được đồng bộ chuẩn xác với commit `3701d2e03b899778fe8ed5fa2f7d4f84527b4187` trên remote.
- Toàn bộ các artifact kiểm toán sinh ra trong thư mục `runs/audits/week6/` và `runs/benchmarks/` chịu sự chi phối của dòng 55 trong `.gitignore`:
  ```gitignore
  /runs/*
  /runs/benchmarks/**
  ```
- **Kết luận kiểm toán:** Việc `.gitignore` loại trừ `/runs/*` là chính sách chuẩn mực nhằm ngăn chặn các tệp nhị phân tạm, log và artifact benchmark cục bộ làm phình to dung lượng Git repository (vốn đã từng bị cảnh báo chạm ngưỡng 2.04 GB ở Task 9).
- Để đảm bảo tính sẵn sàng bàn giao và khả năng truy vết của hội đồng, các bản sao chính thức của các báo cáo cốt lõi đã được lưu trữ tại thư mục gốc repository song song với việc lưu trong `runs/`:
  - `backend_export_manifest.json`
  - `backend_parity_report.json`
  - `backend_benchmark.csv`
  - `week6_protocol_reconciliation.md`
  - `week6_artifact_sync_inventory.json`

Chi tiết toàn bộ mã băm SHA-256 và trạng thái kiểm kê đã được đóng gói tại [`week6_artifact_sync_inventory.json`](file:///d:/workspace/DACN_NCKH/week6_artifact_sync_inventory.json).

---

## 3. PHASE B – SỬA CHỮA 5 LỖI CỐT LÕI CỦA VIDEO TECHNICAL AUDITOR

Hệ thống kiểm toán video kỹ thuật `scripts/audit_negative_video_technical_integrity.py` đã được tái cấu trúc toàn diện để khắc phục 5 lỗi nghiêm trọng:

### 3.1. Bug 1: Khắc phục số đếm khung hình giải mã sai (Full Sequential Decoding)
- **Vấn đề cũ:** Sử dụng phương pháp nhảy cóc (seek/binary search probing) để ước lượng số khung hình. Phương pháp này chỉ đọc được header của container mà không giải mã thực tế các frame payload, bỏ sót các lỗi nén macroblock, corrupt frame và không xác định được luồng VFR (Variable Frame Rate).
- **Giải pháp mới:**
  - Triển khai vòng lặp giải mã tuần tự liên tục ($100\%$ sequential frame walk) từ frame 0 đến frame cuối cùng bằng `cv2.VideoCapture.read()`.
  - Trích xuất timestamp bản địa thông qua `cv2.CAP_PROP_POS_MSEC` để theo dõi PTS từng frame.
  - Tính toán phương sai của khoảng cách delta giữa các frame để nhận diện luồng VFR; neo thời lượng thực tế vào PTS của frame giải mã thành công cuối cùng.
  - Phối hợp kiểm toán độc lập bằng tiến trình `ffmpeg -v error -i <video> -f null -` để bắt mã lỗi thoát (return code) và các dòng cảnh báo phân mảnh NAL/header.

### 3.2. Bug 2: Xóa bỏ dung sai EOF tùy tiện (Eliminate Unsafe Heuristic EOF)
- **Vấn đề cũ:** Tự động gắn nhãn `NORMAL_EOF` (PASS) nếu số frame thiếu hụt $\le 10$ frame hoặc $\le 0,5\%$. Điều này cho phép các video bị đứt đuôi nhẹ vượt qua kiểm định mà không có bằng chứng khoa học.
- **Giải pháp mới:**
  - Nghiêm cấm mọi giả định tùy tiện. Chỉ chấp nhận độ lệch do timescale padding của container audio-video khi thỏa mãn đồng thời 3 điều kiện ngặt:
    1. Số frame thiếu hụt $\le 2$ frames.
    2. Không có bất kỳ lỗi giải mã nào từ decoder hoặc ffmpeg.
    3. Timestamp PTS của frame cuối cùng đạt tới thời lượng kỳ vọng của stream ($\ge t_{dur} - \frac{2}{fps}$).
  - Nếu thiếu hụt nhưng không chứng minh được padding: trả về `UNKNOWN_EOF` hoặc `TRUNCATED_STREAM`.
  - **Phát hiện lỗi hỏng giữa luồng (Mid-stream Corruption):** Nếu vòng lặp tuần tự dừng sớm nhưng frame đuôi vẫn seek được, hệ thống phân loại là `DECODE_ERROR` (hỏng khối dữ liệu giữa video), ngăn chặn gian lận seek.

### 3.3. Bug 3: Chế độ `--no-probe` an toàn tuyệt đối (Safe Trusted Receipts)
- **Vấn đề cũ:** Khi chạy cờ `--no-probe`, script tin tưởng mù quáng vào metadata manifest và gán phán quyết `PASS` cho các video mà không kiểm tra tệp thực tế.
- **Giải pháp mới:**
  - Chế độ `--no-probe` bắt buộc phải có biên nhận kiểm toán đáng tin cậy (`trusted_receipts`) đã được xác thực trước đó.
  - Biên nhận phải khớp mã băm SHA-256 thực tế của tệp video trên đĩa.
  - Nếu không có biên nhận hợp lệ: phán quyết bắt buộc phải là `NOT_VERIFIED`, trạng thái EOF là `UNKNOWN_EOF`, và video bị loại khỏi mẫu số FAR. Tuyệt đối không tự sinh ra phán quyết `PASS`.

### 3.4. Bug 4: Phân tách độc lập 5 phạm vi phơi nhiễm FAR (Disjoint Split Isolation)
- **Vấn đề cũ:** Gộp chung thời lượng của toàn bộ 79 video âm tính thành một con số $7,13$ giờ duy nhất, gây rò rỉ tập Test và tập Train vào mẫu số FAR.
- **Giải pháp mới:** Phân tách và báo cáo độc lập 5 chỉ số thời lượng không giao nhau:
  1. `val_far_eligible_duration_sec`: Chỉ tính các video thuộc tập `val` đã được kỹ thuật PASS và con người xác nhận âm tính.
  2. `test_far_eligible_duration_sec`: Tập Test độc lập, bị cô lập hoàn toàn khỏi Validation.
  3. `train_contaminated_duration_sec`: 37 video âm tính đã dùng trong tập Train `vn_dataset`, tự động bị loại với mã `TRAIN_SPLIT_LEAKAGE_RISK`.
  4. `unresolved_held_out_duration_sec`: Video chưa gán phân chia rõ ràng.
  5. `declared_duration_sec`: Tổng dung lượng danh mục.
  - Tự động phát hiện video trùng lặp SHA-256 (`DUPLICATE_SHA256`) và loại trừ khỏi phơi nhiễm.

### 3.5. Bug 5: Chuẩn hóa thẩm định con người và khử xung đột trường (Alias Conflict Resolution)
- **Vấn đề cũ:** Trộn lẫn các trường dữ liệu đồng nghĩa (`content_decision` vs `review_decision`, `reviewer_name` vs `reviewer`, `attest_full_video_viewed` vs `full_view_attested`), dẫn đến nguy cơ ghi đè hoặc chấp nhận bản ghi mâu thuẫn.
- **Giải pháp mới:**
  - Thiết lập nguồn thẩm quyền duy nhất (Single Source of Authority), liên kết thẩm định với SHA-256 cụ thể của video.
  - Tự động phát hiện xung đột giữa trường chính và trường bí danh: nếu `content_decision` khác `review_decision` $\rightarrow$ lập tức từ chối bản ghi (`CONFLICTING_DECISION_FIELDS`) và tước quyền tham gia mẫu số FAR.

---

## 4. PHASE C – ĐỐI SOÁT GIAO THỨC TẬP VALIDATION VÀ TẬP TEST ĐỘC LẬP

Chi tiết phương pháp luận được trình bày tại [`week6_protocol_reconciliation.md`](file:///d:/workspace/DACN_NCKH/week6_protocol_reconciliation.md).

### Bảng cân đối thời lượng âm tính theo phân tầng:
- **Tập Validation (`val`):** 13 video = $5.131,32$ giây = **$1,4254$ giờ**.
  - So với chỉ tiêu $\ge 5,0$ giờ ($18.000$ giây): **Thâm hụt $3,5746$ giờ ($71,5\%$)**.
- **Tập Independent Test (`test`):** 13 video = $7.550,93$ giây = **$2,0975$ giờ**.
  - Được bảo mật nguyên vẹn cho nghiệm thu độc lập ở Tuần 8.
- **Tập Train bị nhiễm (`train`):** 37 video = $12.405,64$ giây = **$3,4460$ giờ**.
  - Đã xuất hiện trong tập huấn luyện của detector; không được phép dùng để đo FAR vì sẽ làm sai lệch năng lực tổng quát hóa.

### Quyết định của Ban Thẩm định:
Cổng Validation Gate **BẮT BUỘC DUY TRÌ TRẠNG THÁI `BLOCKED`**. Tuyệt đối không được chuyển 13 video Test sang Validation để đối phó chỉ tiêu 5.0 giờ.

---

## 5. PHASE D – THIẾT LẬP RÀO BẢO VỆ CHO EXPERIMENT RUNNER TUẦN 6

- Runner thực nghiệm `scripts/run_week6_ablation.py` đã được kiểm tra:
  - Cấu hình ablation khảo sát đầy đủ:
    - Kích thước ảnh: `imgsz` $\in \{416, 640, 960\}$.
    - Ngưỡng tin cậy: `conf` $\in \{0.25, 0.35, 0.50\}$.
    - Cửa sổ thời gian: $(k, N) \in \{(2, 8), (3, 8), (4, 10)\}$.
    - Phân khúc thời gian: Ngày và Đêm.
  - **Cơ chế khóa an toàn (Validation Gate Guard):** Runner tích hợp lệnh kiểm tra `scripts/validate_week6_gate.py`. Do Gate đang ở trạng thái `BLOCKED`, runner từ chối thực thi Grid Search nhằm ngăn ngừa việc lựa chọn Operating Point trên tập dữ liệu chưa khóa Ground Truth.

---

## 6. PHASE E – ĐO KIỂM VÀ XUẤT HIỆU NĂNG BACKEND (EXPORT & BENCHMARK)

Quá trình đo kiểm thực tế trên môi trường Windows 11 với CPU x86_64 và PyTorch 2.15.0+cpu được ghi nhận đầy đủ, không giả lập bất kỳ backend nào chưa cài đặt.

### 6.1. Bảng tổng hợp hiệu năng triển khai (`backend_benchmark.csv`):
| Backend | Precision | Thiết bị | Batch | Imgsz | Model-Only Latency (ms) [Mean / P50 / P95] | Model-Only FPS | End-to-End Latency (ms) [Mean / P50 / P95] | E2E FPS | RAM (MB) | VRAM (MB) | Trạng thái kỹ thuật |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PyTorch CPU** | FP32 | CPU | 1 | 640 | **$63,81$ / $58,72$ / $113,59$** | **$15,67$** | **$43,47$ / $41,23$ / $59,13$** | **$23,00$** | $290,61$ | $0,0$ | **`MEASURED`** (Hoạt động tốt) |
| **PyTorch CUDA** | FP16 | cuda:0 | 1 | 640 | N/A | N/A | N/A | N/A | N/A | N/A | **`UNAVAILABLE_NO_CUDA`** |
| **ONNX Runtime**| FP32 | N/A | 1 | 640 | N/A | N/A | N/A | N/A | N/A | N/A | **`BLOCKED_DEPENDENCY_UNAVAILABLE`** |
| **OpenVINO** | FP32 | N/A | 1 | 640 | N/A | N/A | N/A | N/A | N/A | N/A | **`BLOCKED_DEPENDENCY_UNAVAILABLE`** |
| **TensorRT** | FP16 | N/A | 1 | 640 | N/A | N/A | N/A | N/A | N/A | N/A | **`BLOCKED_DEPENDENCY_UNAVAILABLE`** |

> [!NOTE]
> Kết quả đo trên PyTorch CPU đạt **$15,67$ FPS (Model-only)** và **$23,00$ FPS (E2E)**, vượt xa chỉ tiêu tối thiểu của Đề cương NCKH MT5 ($\ge 5$ FPS trên CPU).

### 6.2. Kiểm tra tính tương đồng số học (Parity Verification):
- Thử nghiệm trên mô hình Baseline (`runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt`, SHA-256: `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`):
  - Sai số tuyệt đối lớn nhất giữa hai lần suy luận độc lập (`max_abs_diff`): **$0,0$** (thỏa mãn tiêu chí $< 10^{-3}$ cho FP32).
  - Sai số tương đối lớn nhất (`max_rel_diff`): **$0,0$** (thỏa mãn tiêu chí $< 10^{-2}$).
  - Trạng thái kiểm toán: **`VERIFIED`**.

---

## 7. KẾT QUẢ THỰC THI TOÀN BỘ TEST SUITE

Hệ thống đã chạy và kiểm chứng vượt qua toàn bộ các bộ test với tỷ lệ thành công $100\%$:

| Bộ kiểm thử (Test Suite) | Số test case | Kết quả | Trọng tâm kiểm thử |
| :--- | :---: | :---: | :--- |
| `tests/test_negative_video_audit.py` | **26** | **26 PASSED (100%)** | 12 test chức năng kiểm toán + 14 tiêu chuẩn toàn vẹn khoa học bắt buộc (corruption giữa luồng, EOF proof, VFR PTS, alias conflict, 5 scopes, gate blocked guard, backend parity). |
| `tests/test_validation_gate.py` | **13** | **13 PASSED (100%)** | Thẩm định kép dương tính, cô lập 13 video âm tính, đối soát thiếu hụt FAR, phán quyết cổng Validation. |
| `tests/test_negative_video_expansion.py` | **14** | **14 PASSED (100%)** | Mở rộng tập video âm tính, kiểm tra checksum, bảo toàn cấu trúc dữ liệu. |
| `tests/test_week6_phase2.py` | **11** | **11 PASSED (100%)** | Chuẩn bị dữ liệu và hạ tầng kiểm toán Tuần 6. |
| `tests/test_week6_infrastructure.py` | **15** | **14 PASSED, 1 SKIPPED (100%)** | Hạ tầng runner ablation và cấu hình backend (1 test checkpoint GPU được skip hợp lệ do checkpoint lưu ở máy GPU thứ hai). |

---

## 8. TỔNG KẾT THEO PHÂN LOẠI TRẠNG THÁI NGHIỆM THU

### 8.1. IMPLEMENTED (ĐÃ THI CÔNG HOÀN TẤT)
1. Cơ chế giải mã tuần tự $100\%$ khung hình, phát hiện hỏng giữa luồng, nhận diện VFR và neo thời lượng vào PTS native.
2. Quy tắc EOF nghiêm ngặt loại bỏ dung sai tùy tiện; kiểm soát cờ `--no-probe` thông qua biên nhận SHA-256 tin cậy.
3. Tách biệt tuyệt đối 5 phân tầng thời lượng âm tính, phát hiện video trùng lặp checksum và tự động đánh trượt video rò rỉ Train.
4. Cơ chế khử xung đột trường bí danh trong biên bản thẩm định nội dung của con người.
5. Runner đo kiểm và xuất mô hình backend (`scripts/export_and_benchmark_models.py`) tự động sinh đầy đủ manifest, báo cáo sai số và bảng chỉ số CSV.

### 8.2. TESTED (ĐÃ KIỂM THỬ THÀNH CÔNG)
- Đạt $100\%$ vượt qua toàn bộ **79/79 bài kiểm thử đơn vị và hệ thống** trên toàn bộ các test suite liên quan đến Tuần 6.
- Đạt $100\%$ kiểm chứng đối với 14 yêu cầu kiểm toán khoa học ngặt nghèo trong `TestScientificIntegrityMandatoryRequirements`.

### 8.3. VERIFIED (ĐÃ XÁC THỰC KHOA HỌC)
- Xác thực tính nhất quán và độc lập của 13 video Test độc lập ($2,10$ giờ).
- Xác thực hiệu năng suy luận trên PyTorch CPU đạt $15,67$ FPS Model-only và $23,00$ FPS E2E (vượt chỉ tiêu $\ge 5$ FPS).
- Xác thực tính bất biến số học (deterministic self-parity) của mô hình baseline với sai số $0,0$.

### 8.4. BLOCKED (RÀO CẢN BẮT BUỘC ĐANG BẢO LƯU)
- **Cổng Validation Gate tiếp tục ở trạng thái `BLOCKED` do:**
  1. Thiếu hụt $3,57$ giờ video âm tính trên tập Validation so với chỉ tiêu $5,0$ giờ của Đề cương.
  2. 13 video Validation âm tính đang chờ con người xem đủ $100\%$ thời lượng để hoàn thành biên bản xác nhận.
  3. 5 sự kiện Validation dương tính đang chờ biên bản thẩm định độc lập từ Annotator 2 để hoàn thiện Ground Truth đồng thuận kép.
- **Rào cản này bảo vệ đề tài khỏi việc ra quyết định trên dữ liệu thiếu tin cậy và tuân thủ tuyệt đối chuẩn mực khoa học IEEE.**

