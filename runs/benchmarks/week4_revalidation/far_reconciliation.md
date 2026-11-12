# Báo Cáo Đối Soát & Tái Tính Toán Tỷ Lệ Báo Giả (FAR) Tuần 4

## 1. Phương Pháp Luận
- **Công thức:** $\text{FAR} = \frac{K}{\text{Exposure Hours}}$.
- **Khoảng tin cậy:** Garwood Exact Poisson 95% Confidence Interval dựa trên phân phối Chi-square $\chi^2$.
- **Hệ quy chiếu:**
  - **Scope 1 (All Evaluated):** Tính trên toàn bộ 79 luồng video được hệ thống đưa vào suy luận ($T = 7.1312$ giờ).
  - **Scope 2 (Clean Streams Only):** Tính trên 77 luồng video vượt qua 100% kiểm định giải mã stream sạch ($T = 6.9846$ giờ).

## 2. Bảng So Sánh Chỉ Số

| Phạm vi đánh giá | Số video | Thời lượng (giờ) | Pipeline A (Baseline) | Pipeline B2 (Operating Point) | 95% Poisson CI | Mức giảm FAR | Mục tiêu MT3 ($\ge 50\%$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scope 1 (Toàn bộ luồng)** | 79 | 7.1312 h | 138.55 /h | **59.04 /h** | [53.53, 64.95] | **57.39%** | **ĐẠT (PASS)** |
| **Scope 2 (Luồng kiểm định sạch)** | 77 | 6.9846 h | 139.02 /h | **59.70 /h** | [54.11, 65.72] | **57.05%** | **ĐẠT (PASS)** |

## 3. Kết Luận
Cả hai phạm vi phân tích đều xác nhận mức giảm báo động giả của Pipeline B2 vượt xa chỉ tiêu MT3 (giảm $\ge 50\%$), bảo toàn tính khoa học và độ tin cậy tái lập.
