# HƯỚNG DẪN KIỂM ĐỊNH KỸ THUẬT VÀ PHƠI NHIỄM VIDEO ÂM TÍNH (NEGATIVE VIDEO AUDIT GUIDE)

> **Dự án:** FireWatch – Hệ thống phát hiện và cảnh báo sớm cháy/khói từ camera giám sát  
> **Tài liệu tham chiếu:** `DeCuong_NCKH_PhatHienChay_Camera.docx` (MT1, MT3, MT5)  
> **Module công cụ:** `scripts/audit_negative_video_technical_integrity.py`  

---

## 1. MỤC ĐÍCH VÀ PHẠM VI ÁP DỤNG

Tài liệu này quy định quy trình chuẩn (SOP) để kiểm định tính toàn vẹn kỹ thuật, xác định điểm kết thúc file (EOF), phát hiện rò rỉ dữ liệu phân chia (split leakage), và tính toán thời lượng phơi nhiễm hợp lệ (FAR exposure) cho toàn bộ tập video âm tính (không cháy/khói) trong dự án FireWatch.

Mục tiêu cốt lõi: **Đảm bảo từng giây phơi nhiễm được đưa vào mẫu số tính tỷ lệ báo động giả (False Alarm Rate - FAR/giờ) đều có cơ sở kỹ thuật và liêm chính dữ liệu tuyệt đối.**

---

## 2. NGUYÊN TẮC KHOA HỌC BẮT BUỘC (SCIENTIFIC INVARIANTS)

1. **Không suy luận đơn tín hiệu:** Tuyệt đối không suy luận `NORMAL_EOF` chỉ từ `VideoCapture.grab() == False` hoặc `cap.read()[0] == False`. Một frame lỗi cục bộ (ví dụ frame hỏng do nhiễu NAL unit như tại frame 26.556 của `VN_NEG_10`) có thể khiến `grab()` trả về `False`, nhưng video vẫn tiếp tục đến frame 29.804.
2. **Loại trừ triệt để video lỗi:** Video bị đứt luồng (`TRUNCATED_STREAM`) như `VN_NEG_48` (chỉ có 2.533 frame trong tổng số 8.445 frame khai báo, mất 70% luồng) bị loại vĩnh viễn khỏi mẫu số FAR.
3. **Phòng chống rò rỉ huấn luyện (Train Contamination Prevention):** Bất kỳ video âm tính nào đã bị trích xuất khung hình vào tập `train` của `vn_dataset` (37 video) đều **bị cấm** sử dụng làm phơi nhiễm âm tính cho tập Test hay Validation.
4. **Quy tắc xác minh con người (Human Content Verification Gate):** Không một video nào được phép tự động phong cấp thành `VERIFIED_NEGATIVE`. Bắt buộc phải có tên chuyên viên đánh giá hợp lệ (không phải SYSTEM, AUTO, BOT, PENDING), thời điểm thẩm định dạng ISO 8601, và cam kết đã xem trọn vẹn toàn bộ thời lượng (`attest_full_video_viewed == True`).
5. **Tuyệt đối cấm dữ liệu nhân tạo:** Cấm tạo video giả bằng AI, cấm kéo dài thời lượng bằng cách lặp lại khung hình (looping), cấm đổi video dương tính thành âm tính.

---

## 3. PHÂN LOẠI TRẠNG THÁI EOF (EOF TAXONOMY)

Hệ thống phân loại trạng thái kết thúc luồng video thành 4 lớp:

| Trạng thái EOF | Tiêu chí kỹ thuật | Kết luận Kỹ thuật | Đủ điều kiện FAR? |
| :--- | :--- | :---: | :---: |
| **`NORMAL_EOF`** | Giải mã thành công Frame 0, 25%, 50%, 75%, 95%. Độ lệch $|N_\text{declared} - N_\text{decoded}| \le 10$ khung hình hoặc $\le 0,5\%$ (sai số timescale audio padding tiêu chuẩn của container MP4). | **`PASS`** | Có thể (nếu đạt kiểm định con người & không rò rỉ Train). |
| **`TRUNCATED_STREAM`** | Luồng video dừng trước thời hạn khai báo đáng kể ($|N_\text{declared} - N_\text{decoded}| > 10$ khung hình và $> 0,5\%$). Các frame ở nửa sau hoặc cuối file không thể giải mã (như `VN_NEG_48`). | **`FAIL`** | **LOẠI TRỪ 100%** |
| **`DECODE_ERROR`** | OpenCV không thể mở stream (`isOpened() == False`), hoặc Frame 0 không thể giải mã, hoặc bitstream bị hỏng hoàn toàn. | **`FAIL`** | **LOẠI TRỪ 100%** |
| **`UNKNOWN_EOF`** | Header atom bị lỗi, FPS $\le 0$, kích thước frame $\le 0$, hoặc metadata container mâu thuẫn không thể xác định điểm cuối. | **`FAIL`** | **LOẠI TRỪ 100%** |

---

## 4. BẢNG 6 CHỈ SỐ THỜI LƯỢNG VÀ ĐỐI CHIẾU PHƠI NHIỄM

Mọi báo cáo nghiệm thu và đánh giá FAR phải xuất trình đủ 6 chỉ số:

```text
1. declared_duration_sec: Tổng thời lượng khai báo từ container metadata / manifest ban đầu.
2. decoded_duration_sec: Tổng thời lượng giải mã thực tế (N_decoded / FPS).
3. verified_technical_duration_sec: Thời lượng các video vượt qua kiểm định kỹ thuật (PASS).
4. human_verified_negative_duration_sec: Thời lượng các video được con người xác nhận sạch (VERIFIED_NEGATIVE).
5. far_eligible_duration_sec: Thời lượng phơi nhiễm hợp lệ đưa vào mẫu số FAR (Đạt kỹ thuật + Con người xác nhận + Không rò rỉ).
6. remaining_duration_to_5h_sec: max(0.0, 18000.0 - far_eligible_duration_sec) (Chỉ tiêu tối thiểu: 18.000s = 5,0 giờ).
```

---

## 5. HƯỚNG DẪN VẬN HÀNH CÔNG CỤ

### A. Kiểm định tự động toàn bộ video âm tính
Chạy script kiểm định kỹ thuật từ thư mục gốc dự án:

```powershell
python scripts/audit_negative_video_technical_integrity.py
```

Các tùy chọn dòng lệnh:
- `--manifest <path>`: Đường dẫn manifest video âm tính (mặc định: `data/negative_videos_manifest.json`).
- `--split-manifest <path>`: Đường dẫn manifest chia tập (mặc định: `datasets/vn_dataset/split_manifest.csv`).
- `--output-dir <path>`: Thư mục xuất báo cáo (mặc định: `runs/audits/week6`).
- `--val-review-manifest <path>`: Bảng kết quả review của con người.
- `--target-hours <float>`: Chỉ tiêu số giờ phơi nhiễm (mặc định: `5.0`).
- `--no-probe`: Bỏ qua bước giải mã trực tiếp (chỉ dùng khi kiểm tra nhanh cú pháp).

### B. Quy trình chuyên viên xác minh nội dung (Human Review Workflow)
1. Mở file bảng làm việc: `runs/audits/week6/validation_negative_review_manifest.csv`.
2. Mở từng video bằng trình phát video (VLC Media Player, Windows Media Player hoặc OpenCV viewer).
3. Xem trọn vẹn từ giây đầu tiên đến giây cuối cùng:
   - Nếu tuyệt đối không có tia lửa, ngọn lửa, khói thuốc lá, khói cháy, tàn tro: Ghi nhận cột `content_decision` là `VERIFIED_NEGATIVE`.
   - Nếu phát hiện có đám cháy hoặc khói thực: Ghi nhận `FIRE_PRESENT` hoặc `SMOKE_PRESENT`.
   - Nếu không chắc chắn: Ghi nhận `UNCERTAIN`.
4. Điền tên người thẩm định vào cột `reviewer_name` (ví dụ: `quocviet_mai`).
5. Điền thời điểm thẩm định vào cột `review_timestamp` dạng ISO 8601 (ví dụ: `2026-11-17T18:00:00Z`).
6. Đánh dấu `attest_full_video_viewed` thành `TRUE`.
7. Chạy lại công cụ để tái kiểm tra tính hợp lệ và cập nhật mẫu số FAR.

---

## 6. CÁC ARTIFACT ĐẦU RA VÀ Ý NGHĨA

Khi hoàn tất, công cụ tự động sinh ra 4 artifact tại `runs/audits/week6/`:
1. `negative_video_technical_audit_report.md`: Báo cáo chi tiết định dạng Markdown đối chiếu 6 chỉ số thời lượng, phân loại EOF, danh sách video bị loại trừ và lý do.
2. `negative_video_integrity_manifest.csv`: Bảng dữ liệu chi tiết từng video gồm 28 trường (mã băm SHA-256, số frame, độ phân giải, chênh lệch frame, trạng thái EOF, phán quyết kỹ thuật, điều kiện FAR).
3. `negative_video_integrity_manifest.json`: Phiên bản JSON có cấu trúc phục vụ tích hợp API và pipeline tự động.
4. `far_exposure_reconciliation.json`: Tệp đối chiếu thời lượng, chứa đầy đủ số liệu giây, giờ, thâm hụt tới mốc 5,0h và trạng thái cổng (PASSED / BLOCKED).
