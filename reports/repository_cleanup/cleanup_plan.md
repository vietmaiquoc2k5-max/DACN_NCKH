# KẾ HOẠCH DỌN DẸP AN TOÀN REPOSITORY (SAFE CLEANUP PLAN)
**Dự án:** DACN_NCKH — Hệ thống Phát hiện & Cảnh báo Cháy/Khói Thông minh (FireWatch)  
**Tài liệu tham chiếu:** `DeCuong_NCKH_PhatHienChay_Camera.docx`  
**Nhánh Git:** `fix-4` | **Commit HEAD:** `d4b3bdca7c6fce964b06355303fd13f1ea42071c`  
**Ngày lập kế hoạch:** 17/11/2026  
**Nguyên tắc tối thượng:** Không xóa mù quáng, không mất dữ liệu cục bộ, không tự ý viết lại Git history, không phá vỡ tính tái lập khoa học.

---

## 1. TỔNG QUAN KẾ HOẠCH DỌN DẸP

Hiện tại, repository đang theo dõi **35.369 tệp tin** với tổng dung lượng **1.991,27 MB (~1,945 GB)** trên nhánh `fix-4`.
Kế hoạch này đề xuất một quy trình dọn dẹp theo 2 tầng:
- **Tầng 1 (Staging & Untracking):** Hủy theo dõi (`git rm --cached`) các thư mục dữ liệu lớn, video kiểm thử và file log thô, đưa chúng vào `.gitignore` để ngăn commit lại. Toàn bộ tệp trên ổ cứng vật lý được giữ nguyên 100%.
- **Tầng 2 (Lưu trữ ngoài & Tối ưu LFS/Releases):** Đánh giá tập dữ liệu `datasets/vn_dataset/` và đề xuất phương án tối ưu dài hạn.

---

## 2. DANH SÁCH CHI TIẾT CÁC TỆP/THƯ MỤC ĐỀ XUẤT HỦY THEO DÕI (`git rm --cached`)

| Thứ tự | Thư mục / Tệp tin | Số lượng tệp | Dung lượng | Lý do & Căn cứ kỹ thuật | Trạng thái sau dọn dẹp |
|---|---|---|---|---|---|
| 1 | `uploads/` (các video `.mp4`) | 6 | 221,00 MB | Video kiểm thử do người dùng tải lên giao diện web. Hàm `get_available_test_videos()` trong `src/config.py` đọc trực tiếp từ đĩa cục bộ. | Giữ nguyên trên đĩa; Git chỉ giữ `.gitkeep`. |
| 2 | `FireSmokeLabeler/videos/` (video `.avi`) | 24 | 215,98 MB | Các video quay thử nghiệm giai đoạn đầu. Không thuộc tập 79 video âm tính hoặc 104 video dương tính nghiệm thu. | Giữ nguyên trên đĩa; untrack khỏi Git. |
| 3 | `dataset/` (Roboflow legacy) | 15.161 | 300,06 MB | Tập dữ liệu cũ với nhãn bị ngược (0: fire, 1: smoke) theo `data_legacy.yaml`. Không dùng cho MT2/MT4. | Giữ nguyên trên đĩa; untrack khỏi Git. |
| 4 | `FireSmokeLabeler/dataset/` | 7.077 | 122,94 MB | Chứa nhãn do chính `best.pt` tự sinh. Đề cương quy định: *không dùng làm ground truth để đánh giá hay báo cáo*. | Giữ nguyên trên đĩa; untrack khỏi Git. |
| 5 | `runs/benchmarks/**/*.stdout` | 174 | 55,18 MB | Vết thực thi stdout thô từng frame (nhiều tệp 6–11 MB). Đã có các tệp JSON/CSV tổng hợp đầy đủ số liệu. | Giữ nguyên trên đĩa; untrack khỏi Git. |
| 6 | `yolo26n.pt` (root) | 1 | 5,29 MB | Checkpoint COCO 80 lớp không sử dụng, không có tham chiếu nào trong code. | Giữ nguyên trên đĩa; untrack khỏi Git. |
| 7 | Các tệp log chạy thử (DELETE_CANDIDATE) | 6 | 0,61 MB | Log chạy thử toolkit và studio draft review cũ. | Có thể xóa hoặc untrack. |
| **TỔNG CỘNG** | **Tầng 1 Clean** | **22.449 tệp** | **921,06 MB** | **Giảm 63,5% số tệp Git và 46,3% dung lượng Git** | **0 bytes mất mát trên đĩa** |

---

## 3. CÁC THÀNH PHẦN TUYỆT ĐỐI BẢO TỒN TRÊN GIT (`KEEP_TRACKED`)

1. **Mã nguồn và ứng dụng:**
   - Toàn bộ `src/`, `scripts/`, `configs/`, `tests/` (106 tests pass 100%).
   - `web_app.py`, `app.py`, `detector.py`, `train.py`, `alert_manager.py`.
2. **Mô hình cốt lõi:**
   - `best.pt` (root, 5,94 MB, SHA256: `b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f`): Model phục vụ suy luận thực tế.
   - `yolov8n.pt` (root, 6,25 MB, SHA256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`): Base COCO model.
   - `runs/detect/runs/benchmarks/*/weights/best.pt` (5 checkpoints, 28,8 MB): Bằng chứng nghiệm thu khoa học MT2 (3 seed độc lập).
3. **Bằng chứng nghiệm thu khoa học và tài liệu:**
   - `docs/` (báo cáo nghiệm thu Tuần 4, IEEE survey).
   - `data/video_dataset_list.csv` (danh mục 192 video kiểm định).
   - `DeCuong_NCKH_PhatHienChay_Camera.docx`.
   - `alarm.wav` (1,96 MB): Âm thanh còi báo cháy PCCC.
   - Toàn bộ summary JSON/CSV chứng minh MT3 (FAR giảm 57.39%, 79 video âm tính 7.2 giờ).

---

## 4. QUY TRÌNH THỰC THI DỌN DẸP TỪNG BƯỚC (CHỜ PHÊ DUYỆT)

> [!CAUTION]
> Các lệnh dưới đây chỉ được thực thi sau khi Người dùng (User) xem xét và phê duyệt bản kế hoạch này.

### Bước 1: Cập nhật quy tắc `.gitignore`
Bổ sung các rule sau vào `.gitignore` để chặn việc thêm lại các tệp sau khi untrack:
```gitignore
# Legacy & Auto-generated Datasets (Keep local on disk, prevent re-tracking)
/dataset/
/FireSmokeLabeler/dataset/
/FireSmokeLabeler/videos/
/uploads/
!/uploads/.gitkeep

# Checkpoint rules: prevent accidental committing of arbitrary weights
/*.pt
!/best.pt
!/yolov8n.pt
!/runs/detect/runs/benchmarks/**/weights/best.pt

# Large stdout execution traces
/runs/benchmarks/**/*.stdout
/runs/benchmarks/**/*.stderr
/runs/benchmarks/**/*.stdout.json
/runs/benchmarks/**/*.log
```

### Bước 2: Tạo `.gitkeep` cho thư mục `uploads/`
```powershell
New-Item -ItemType File -Path "uploads\.gitkeep" -Force
```

### Bước 3: Thực thi untrack an toàn trên Git index (KHÔNG XÓA ĐĨA)
```powershell
git rm --cached -r dataset/
git rm --cached -r FireSmokeLabeler/dataset/
git rm --cached -r FireSmokeLabeler/videos/
git rm --cached uploads/VN_POS_04.mp4 uploads/VN_POS_06.mp4 uploads/VN_POS_09.mp4 uploads/VN_POS_11.mp4 uploads/VN_POS_15.mp4 uploads/VN_POS_20.mp4
git rm --cached yolo26n.pt
git rm --cached runs/benchmarks/**/*.stdout
```

### Bước 4: Kiểm tra tính toàn vẹn (Safety Verification)
```powershell
# 1. Xác nhận tệp vẫn tồn tại trên đĩa cục bộ
Test-Path uploads/VN_POS_20.mp4
Test-Path dataset/images/train/images
Test-Path FireSmokeLabeler/videos/testpos02.818.avi

# 2. Kiểm tra biên dịch mã nguồn
python -m compileall -q src scripts

# 3. Chạy toàn bộ test suite
pytest -q
```

---

## 5. KẾ HOẠCH HỒI PHỤC RỦI RO (ROLLBACK PLAN)

Nếu có bất kỳ sự cố nào xảy ra trong quá trình kiểm tra sau khi untrack:
1. **Phục hồi staging index:**
   ```powershell
   git reset HEAD .
   ```
2. **Khôi phục trạng thái branch sạch:**
   ```powershell
   git checkout HEAD -- .
   ```
3. Toàn bộ tệp trên đĩa vật lý không bị tác động nên không thể xảy ra hiện tượng mất mát dữ liệu do các lệnh trên.

