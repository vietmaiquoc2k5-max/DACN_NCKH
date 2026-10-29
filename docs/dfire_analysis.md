# Báo Cáo Chuẩn Hóa & Phân Chia Tập Dữ Liệu D-Fire (MT1)

**Nguyên tắc khoa học:** Tách biệt độc lập 3 tập Train (80%), Validation (10%) và Test (10%) bằng phương pháp **Chia phân tầng theo lớp (Stratified Split, seed=42)**.
- **Tập Train:** Tối ưu hóa trọng số mô hình.
- **Tập Val:** Đánh giá trong quá trình huấn luyện, chọn checkpoint `best.pt` và điều chỉnh siêu tham số.
- **Tập Test:** Hoàn toàn độc lập, chỉ dùng để đánh giá năng lực suy diễn cuối cùng (tránh rò rỉ dữ liệu / Data Leakage).

## 1. Bảng Thống Kê Chi Tiết Số Lượng & Phân Bố Lớp (Sau Làm Sạch & Phân Tầng)

| Chỉ số | Tập Train (80%) | Tập Val (10%) | Tập Test (10%) | Tổng cộng (100%) |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng số hình ảnh** | 17,221 | 2,153 | 2,153 | 21,527 |
| **Tổng số file nhãn (.txt)** | 17,221 | 2,153 | 2,153 | 21,527 |
| **Ảnh chỉ có Khói (Smoke)** | 4,681 | 593 | 593 | 5,867 |
| **Ảnh chỉ có Lửa (Fire)** | 944 | 110 | 110 | 1,164 |
| **Ảnh có cả Khói & Lửa** | 3,763 | 447 | 448 | 4,658 |
| **Ảnh âm tính (Background)** | 7,833 | 1,003 | 1,002 | 9,838 |
| **Bounding Box Khói (0)** | 9,543 | 1,155 | 1,156 | 11,854 |
| **Bounding Box Lửa (1)** | 11,807 | 1,417 | 1,461 | 14,685 |
| **Tổng số Box** | 21,350 | 2,572 | 2,617 | 26,539 |

## 2. Kết Luận Kiểm Định

- Đã khắc phục triệt để lỗi cấu hình `val: test/images`.
- Đã loại bỏ 18 bounding box suy biến (`w=0` hoặc `h=0`) và chuẩn hóa 8 bounding box tràn viền nhẹ về đúng miền `[0, 1]`.
- Phân tầng theo lớp (Stratified Split) đảm bảo tỷ lệ ảnh từng nhóm (`Smoke only`, `Fire only`, `Both`, `Background`) giữa Val và Test cân bằng 50/50 chính xác.
