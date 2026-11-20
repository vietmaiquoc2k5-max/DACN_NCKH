# HƯỚNG DẪN KHÔI PHỤC VÀ TẢI TÀI NGUYÊN (RESTORATION PLAN)
**Dự án:** DACN_NCKH — Hệ thống Phát hiện & Cảnh báo Cháy/Khói Thông minh (FireWatch)  
**Tài liệu tham chiếu:** `DeCuong_NCKH_PhatHienChay_Camera.docx`  
**Ngày lập tài liệu:** 17/11/2026  

---

## 1. MỤC ĐÍCH

Tài liệu này cung cấp quy trình chuẩn hóa giúp một thành viên mới, một máy tính GPU khác, hoặc môi trường CI/CD sau khi thực hiện `git clone` repository tinh gọn (đã dọn dẹp) có thể:
1. Nắm được danh mục đầy đủ các tài nguyên dữ liệu, mô hình và video cần thiết.
2. Tải và thiết lập đúng vị trí thư mục cục bộ mà không gặp lỗi đường dẫn.
3. Xác thực tính toàn vẹn (checksum SHA-256) của từng tài nguyên được khôi phục.

---

## 2. DANH MỤC VÀ HƯỚNG DẪN KHÔI PHỤC THEO TỪNG THÀNH PHẦN

### 2.1. Thư mục tải lên video kiểm thử (`uploads/`)
- **Vị trí cục bộ:** `uploads/`
- **Mục đích:** Chứa các video MP4 phục vụ tính năng kiểm thử nguồn video trên từng camera từ Web Dashboard.
- **Cách khôi phục:**
  - Thư mục được tự động tạo nếu chưa tồn tại khi khởi chạy `web_app.py`.
  - Người dùng có thể copy các video kiểm thử (ví dụ: `VN_POS_04.mp4` đến `VN_POS_20.mp4` hoặc video bất kỳ) vào thư mục này.
  - Web Dashboard sẽ tự động quét và hiển thị trong danh sách chọn video kiểm thử camera.

---

### 2.2. Tập dữ liệu D-Fire (MT1 & MT2 - Kaggle / External Storage)
- **Vị trí cục bộ:** `datasets/dfire_author_linked_kaggle_v1/` hoặc `datasets/dfire/`
- **Mục đích:** Tập dữ liệu huấn luyện và kiểm thử chính quy chuẩn (21.527 ảnh, 39.521 bboxes).
- **Cách khôi phục:**
  1. Tải bộ dữ liệu D-Fire từ nguồn công bố của tác giả (Kaggle: `dfire-dataset`).
  2. Giải nén vào thư mục `datasets/dfire/` theo cấu trúc:
     ```
     datasets/dfire/
     ├── train/
     │   ├── images/
     │   └── labels/
     ├── val/
     │   ├── images/
     │   └── labels/
     └── test/
         ├── images/
         └── labels/
     ```
  3. Cấu hình kiểm tra: `data_dfire.yaml`.
  4. Chạy script kiểm tra toàn vẹn:
     ```powershell
     python scripts/verify_dfire_integrity.py
     ```

---

### 2.3. Tập video âm tính kiểm định (MT3 - 79 Video âm tính thực tế)
- **Vị trí cục bộ:** `datasets/negative_videos/` (hoặc `datasets/test_videos/neg/`)
- **Mục đích:** Đánh giá tỉ lệ báo động giả (FAR - False Alarm Rate) trên 7,20 giờ video âm tính thực tế có cảnh nhiễu (đáp ứng chỉ tiêu MT1 & MT3).
- **Cách khôi phục:**
  1. Sao chép 79 video âm tính từ máy chủ lưu trữ dữ liệu nghiên cứu nội bộ (Google Drive / NAS) vào `datasets/negative_videos/`.
  2. Kiểm tra danh mục đối chiếu: `data/video_dataset_list.csv`.
  3. Chạy lệnh kiểm toán thời lượng và tính hợp lệ:
     ```powershell
     python scripts/run_negative_evaluation_final.py --help
     ```

---

### 2.4. Tập dữ liệu khung hình Việt Nam (MT1 & MT4 - `datasets/vn_dataset/`)
- **Vị trí cục bộ:** `datasets/vn_dataset/`
- **Hiện trạng:** Đang được theo dõi trên Git nhánh `fix-4` (3.121 ảnh JPEG, 3.121 txt nhãn).
- **Quy tắc bảo vệ:** Nếu trong tương lai chuyển sang lưu trữ dạng GitHub Release Asset:
  1. Tải gói lưu trữ `vn_dataset_release_v1.tar.gz` từ GitHub Releases.
  2. Giải nén vào thư mục `datasets/vn_dataset/`.
  3. Đối chiếu cấu hình với `data_vn.yaml`.

---

### 2.5. Trọng số mô hình (Model Checkpoints)
- **Mô hình suy luận chính (`best.pt`):** Đã được theo dõi trực tiếp tại gốc repo (SHA256: `b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f`). Không cần tải thêm.
- **Mô hình nền tảng COCO (`yolov8n.pt`):** Đã được theo dõi trực tiếp tại gốc repo (SHA256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`). Nếu cần tải lại, Ultralytics sẽ tự động tải khi gọi `YOLO('yolov8n.pt')`.
- **Mô hình 3 seed benchmark MT2:** Nằm sẵn trong `runs/detect/runs/benchmarks/*/weights/best.pt`.

---

## 3. CHECKLIST KIỂM TRA MÔI TRƯỜNG SAU KHÔI PHỤC

Trước khi chạy huấn luyện hoặc triển khai giám sát, chạy chuỗi kiểm tra sau:

```powershell
# 1. Kiểm tra môi trường Python
python -c "import torch, ultralytics, cv2; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Ultralytics:', ultralytics.__version__)"

# 2. Kiểm tra unit test hệ thống
pytest -q

# 3. Khởi chạy thử nghiệm Web Dashboard
python web_app.py --test-mode
```

