# Báo Cáo Tính Toán Tỷ Lệ Báo Động (Alert Rate) Khoa Học Tuần 4

## 1. Phương Pháp Luận
- **Chỉ số đo lường:** Alert-Event Rate = K / Exposure Hours.
- **Khoảng tin cậy:** Garwood Exact Chi-square Poisson 95% Confidence Interval.
- **Quy chuẩn danh xưng:** Thuật ngữ được ghi nhận chính xác là **Alert-Event Rate (/h)** đối với tập video giám sát danh nghĩa âm tính. Thuật ngữ **False Alarm Rate (FAR)** sẽ được chuẩn hóa khi hoàn tất thẩm định nhãn Ground Truth độc lập ở Scope C.

## 2. Bảng So Sánh Chi Tiết Các Phạm Vi (Scopes)

| Phạm vi đánh giá | Số video | Thời lượng (h) | Baseline Pipeline A | Operating Point Pipeline B2 | Khoảng tin cậy 95% Poisson | Mức giảm (%) | Chỉ tiêu MT3 (>= 50%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scope A (Historical Diagnostic)** | 79 video | 7.1312 h | 138.55 /h | **59.04 /h** | [53.53; 64.95] | **57.39%** | **ĐẠT (PASS)** |
| **Scope B1 (Prior Audit 16 Cams)** | 16 video | 0.1623 h | 166.37 /h | **30.81 /h** | [10.00; 71.90] | **81.48%** | **ĐẠT (PASS)** |
| **Scope B2 (Full 19 Strict-Valid)** | 19 video | 2.7804 h | 58.98 /h | **12.5882 /h** | [8.7681; 17.5071] | **78.66%** | **ĐẠT (PASS)** |
| **Scope B3 (77 Untruncated Streams)**| 77 video | 6.9846 h | 139.02 /h | **59.70 /h** | [54.11; 65.72] | **57.05%** | **ĐẠT (PASS)** |
| **Scope C (Verified Negative GT)** | 0 video | 0.0000 h | N/A | N/A | N/A | N/A | **PENDING_REVIEW** |

## 3. Giải Quyết Mâu Thuẫn Số Liệu Scope 2 (57.05% vs 56.48%)
- **Số liệu 57.05%:** Tính toán dựa trên mẫu số nội bộ của tập 77 video không bị cắt luồng:
  $$\text{Reduction} = \frac{139.0195 - 59.7025}{139.0195} \times 100\% = 57.05\%$$
- **Số liệu 56.48%:** Tính toán khi so sánh tỷ lệ báo giả của 77 video (59.7025/h) với Baseline FAR toàn cục của toàn bộ 7.2026 giờ (137.1727/h):
  $$\text{Reduction} = \frac{137.1727 - 59.7025}{137.1727} \times 100\% = 56.48\%$$
- **Kết luận:** Cả hai cách tính đều đúng về mặt số học trên hệ quy chiếu tương ứng và đều vượt xa yêu cầu giảm >= 50% của Đề cương NCKH.
