# Báo Cáo Chuẩn Hóa & Phân Chia Tập Dữ Liệu D-Fire (MT1)

- **Ngày cập nhật:** 29/10/2026
- **Tập dữ liệu:** D-Fire (Dataset for Fire and Smoke Detection)
- **Định dạng:** YOLO format (`0: smoke`, `1: fire`)
- **Tập tin cấu hình:** `data_dfire.yaml` (Đã chuẩn hóa 3 tập Train / Val / Test độc lập)

---

## 1. Nguyên Tắc Khoa Học Về Phân Chia Tập Dữ Liệu
Nhằm khắc phục triệt để lỗi rò rỉ dữ liệu (*Data Leakage* - trước đây trỏ `val: test/images`), tập dữ liệu đã được phân định rõ ràng thành 3 tập độc lập:
1. **Tập Huấn luyện (Train - 80%):** Tối ưu hóa trọng số mô hình.
2. **Tập Kiểm định (Validation / Val - 10%):** Đánh giá loss, tính mAP trong quá trình huấn luyện, lựa chọn trọng số tối ưu `best.pt` và điều chỉnh siêu tham số.
3. **Tập Kiểm thử (Test - 10%):** Giữ kín hoàn toàn (*Held-out set*), chỉ được đưa vào đánh giá ở bước cuối cùng sau khi hoàn tất huấn luyện để đo năng lực suy diễn thực tế khách quan.

---

## 2. Bảng Thống Kê Chi Tiết Số Lượng & Phân Bố Lớp

| Chỉ số | Tập Train (80%) | Tập Val (10%) | Tập Test (10%) | Tổng cộng (100%) | Tỷ lệ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tổng số hình ảnh** | **17,221** | **2,153** | **2,153** | **21,527** | 100.0% |
| **Ảnh chỉ có Khói (Smoke only)** | 4,681 | 617 | 569 | 5,867 | 27.25% |
| **Ảnh chỉ có Lửa (Fire only)** | 944 | 116 | 104 | 1,164 | 5.41% |
| **Ảnh có cả Khói & Lửa (Both)** | 3,763 | 429 | 466 | 4,658 | 21.64% |
| **Ảnh âm tính (Background / Cảnh nhiễu)** | 7,833 | 991 | 1,014 | 9,838 | 45.70% |
| **Số Bounding Box Khói (Class 0: smoke)** | 9,550 | 1,164 | 1,151 | 11,865 | 44.68% |
| **Số Bounding Box Lửa (Class 1: fire)** | 11,814 | 1,418 | 1,460 | 14,692 | 55.32% |
| **Tổng số Bounding Box** | **21,364** | **2,582** | **2,611** | **26,557** | 100.0% |

---

## 3. Nhận Xét Khoa Học
1. **Khắc phục lỗi rò rỉ dữ liệu:** Tập `val` và `test` hoàn toàn không giao nhau (0 ảnh trùng lặp).
2. **Cân bằng phân phối:** Phân phối giữa các lớp Khói, Lửa và cảnh âm tính đồng đều tương thích giữa tập Val và tập Test (~45% background, ~44% box khói, ~55% box lửa).
3. **Ý nghĩa của 9.838 ảnh âm tính (Background):** Chiếm tới 45.7% toàn tập, cung cấp dữ liệu huấn luyện cảnh nhiễu thực tế (đèn đường, ánh sáng, xe cộ, sương mù...) giúp giảm tối đa tỷ lệ báo động giả (False Positive Rate) theo mục tiêu MT3.
