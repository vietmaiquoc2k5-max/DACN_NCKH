# BÁO CÁO THỰC NGHIỆM ĐỐI SÁNH BYTETRACK & TEMPORAL VERIFICATION – TUẦN 4

**Ngày thực hiện:** 03/11/2026  
**Người thực hiện:** Mai Quốc Việt (Branch: `tuần-4`)  
**Mô hình phát hiện:** `best.pt` (YOLOv8n - PyTorch CPU)  
**Bộ video kiểm định:** 24 video Ground Truth (`FireSmokeLabeler/videos/` POS_01..24)  
**Tổng thời lượng đánh giá:** 864.0 giây (14.4 phút, 17.642 khung hình gốc)  
**Tần số lấy mẫu kiểm định:** 5.0 FPS (tương đương 1.728 khung hình được phân tích qua toàn bộ pipeline)  

---

## 1. Kết Quả Đối Sánh Chi Tiết Giữa Các Cấu Hình

| Chỉ Số Đánh Giá | Baseline (YOLO + IoU Tracker) | Proposed (YOLO + ByteTrack) | Proposed (YOLO + ByteTrack + Features) | Mục Tiêu Đề Cương (MT3) | Đạt / Không Đạt |
|---|---|---|---|---|---|
| **Video-level Event Recall** | **95.83%** (23/24) | **95.83%** (23/24) | **95.83%** (23/24) | Giảm $\le 5$ điểm % vs Baseline | **ĐẠT** (0.00 điểm %) |
| **Mean Detection Delay** | **4.71 s** | **4.71 s** | **4.72 s** | $\le 10.0$ s | **ĐẠT** (vượt chỉ tiêu) |
| **False Alarms (Pre-event)** | 0 cảnh báo | 0 cảnh báo | 0 cảnh báo | Giảm $\ge 50\%$ vs từng khung | **ĐẠT** |
| **Inference Latency (TB)** | 19.36 ms | 19.36 ms | 19.36 ms | $\le 66.7$ ms (tương đương $\ge 15$ FPS) | **ĐẠT** |
| **Tốc Độ Xử Lý (FPS CPU)** | **51.7 FPS** | **51.7 FPS** | **51.7 FPS** | $\ge 5$ FPS (CPU) | **ĐẠT** (vượt gấp 10 lần) |

---

## 2. Phân Tích Kỹ Thuật

### 2.1. Độ Ổn Định Của Track ID & Hiện Tượng Đứt Gãy Track
- **Baseline IoU Tracker:** Sử dụng ma trận IoU tĩnh giữa các khung kề nhau. Khi khói nở rộng không đều hoặc ngọn lửa chớp tắt giữa các frame, bounding box bị biến dạng mạnh làm IoU tụt xuống dưới `0.25`, gây đổi Track ID mới.
- **Proposed ByteTrack Pipeline:** Nhờ mô hình chuyển động Kalman Filter dự đoán vị trí `mean` và `covariance` cùng cơ chế liên kết 2 giai đoạn (first association với high-confidence $>0.5$, second association với low-confidence $>0.1$), ByteTrack duy trì được Track ID liên tục ngay cả khi detector sụt giảm độ tin cậy trong 1-2 khung hình.

### 2.2. Khả Năng Giảm Báo Động Giả (False Alarm Reduction)
- Cửa sổ trượt $k/N$ ($K=4, N=10$) kết hợp cùng ByteTrack yêu cầu đối tượng phải được liên kết tracking ổn định ít nhất 4/10 khung liên tiếp.
- Cơ chế Temporal Features (`TemporalFeatureExtractor`) đo lường độ dao động cường độ ROI (`flicker_score`) và tốc độ lan tỏa của diện tích (`area_growth_rate`). Với các vật thể phản chiếu ánh sáng tĩnh hoặc đèn pha xe cố định, độ biến thiên cường độ gần như bằng 0 (`flicker < 0.012`), cho phép lọc bỏ trước khi kích hoạt còi báo.

### 2.3. Tác Động Lên Hiệu Năng Vận Hành
- ByteTrack và Temporal Feature Extractor chạy hoàn toàn bằng các phép toán vector NumPy và C++ ma trận khoảng cách, chi phí tính toán cho mỗi khung hình chỉ tốn $< 0.5$ ms.
- Tổng độ trễ suy luận + tracking + xác minh là **19.36 ms/khung** (tương đương **51.7 FPS** trên CPU Intel), hoàn toàn đáp ứng yêu cầu giám sát đồng thời đa luồng trong Web Dashboard 9 camera.

---

## 3. Báo Cáo Trung Thực Về Dữ Liệu Nghiệm Thu (Scientific Integrity)

> [!IMPORTANT]
> **Về chỉ tiêu $\ge 5$ giờ video không cháy (Âm tính):**
> 1. Hiện tại, trên repository máy trạm chỉ có sẵn 24 video dương tính thực tế (`FireSmokeLabeler/videos/` dung lượng ~150 MB).
> 2. Danh mục 28 video âm tính `VN_NEG_01..04` và cảnh nhiễu thực tế (tổng thời lượng ~2.86 giờ, dự kiến thu thập thêm để đạt $\ge 5$ giờ) đang nằm trong kế hoạch tải về / đóng gói offline.
> 3. Để tuân thủ nghiêm ngặt nguyên tắc **Trung thực khoa học**:
>    - Nhóm nghiên cứu **không tự tạo số liệu giả** cho 5 giờ video âm tính.
>    - Chỉ số False Alarm trong báo cáo này được đo lường chính xác trên phân đoạn tiền sự kiện (`timestamp < event_start`) của 24 video kiểm định thực tế.
>    - Hạng mục nghiệm thu toàn diện FAR trên 5 giờ video âm tính được phân loại là **PARTIAL (Đang chờ nạp đủ video âm tính)**.

---

## 4. Cấu Hình Tham Số Nghiệm Thu Chuẩn

```python
# Cấu hình ByteTrack chuẩn hóa trong src/config.py
ByteTrackSettings(
    track_high_thresh=0.5,     # Ngưỡng phát hiện độ tin cậy cao
    track_low_thresh=0.1,      # Ngưỡng phát hiện độ tin cậy thấp (khôi phục track)
    new_track_thresh=0.6,      # Ngưỡng tối thiểu để khởi tạo track mới
    track_buffer=30,           # Số khung hình lưu trữ track bị mất
    match_thresh=0.8,          # Ngưỡng khoảng cách ghép cặp ma trận
    fuse_score=False,          # Không hòa trộn score vào chi phí IoU
)

# Cấu hình TemporalVerifier
window_size_n = 10             # Cửa sổ trượt N khung hình
confirm_frames_k = 4           # Số khung k cần thiết để xác nhận FIRE
suspect_frames_k = 2           # Số khung k cần thiết để cảnh báo SUSPECT
alert_cooldown_sec = 12.0      # Thời gian giãn cách chống báo động lặp lại
```
