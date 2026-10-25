# Báo Cáo Phân Tích Tập Dữ Liệu D-Fire (MT1)

- **Ngày cập nhật:** 24/10/2026 23:26:46
- **Tập dữ liệu:** D-Fire (Dataset for Fire and Smoke Detection)
- **Định dạng:** YOLO format (Classes: `0: smoke`, `1: fire`)

## 1. Bảng Thống Kê Chi Tiết

| Chỉ tiêu | Tập Huấn Luyện (Train) | Tập Kiểm Thử (Test/Val) | Tổng Cộng |
| :--- | :--- | :--- | :--- |
| **Tổng số hình ảnh** | 17,221 | 4,306 | 21,527 |
| **Số ảnh dương tính (chứa khói/lửa)** | 9,388 | 2,301 | 11,689 |
| **Số ảnh âm tính (Background / Cảnh nhiễu)** | 7,833 | 2,005 | 9,838 |
| **Bounding Box Khói (Class 0: smoke)** | 9,550 | 2,315 | 11,865 |
| **Bounding Box Lửa (Class 1: fire)** | 11,814 | 2,878 | 14,692 |
| **Tổng số Bounding Box** | 21,364 | 5,193 | 26,557 |

## 2. Nhận Xét & Đánh Giá Đạt Chuẩn Đề Cương

- **Tính cân bằng và đa dạng:** Tập dữ liệu D-Fire sở hữu số lượng ảnh âm tính (background không có lửa/khói) đáng kể, đóng vai trò sống còn trong việc huấn luyện mô hình giảm thiểu hiện tượng báo động giả (False Positives) theo chỉ tiêu MT3.
- **Đồng bộ hệ thống:** Cấu trúc phân lớp đã được kiểm chứng khớp 100% với kiến trúc của dự án (`0: smoke`, `1: fire`), sẵn sàng cho quá trình huấn luyện các mô hình theo tiến độ Tuần 3.

Agent
tuần 2 tạm thời bỏ qua video vietnam
11:00 PM
