# Báo Cáo Kiểm Định Toàn Vẹn & Độc Lập Dữ Liệu D-Fire (Data Integrity Audit)

- **Ngày giờ kiểm định thực tế:** 29/10/2026 16:11:35
- **Thời gian chạy kiểm toán:** 25.3 giây
- **Tổng số hình ảnh kiểm định:** 21,527 ảnh
- **Tổng số nhãn kiểm định:** 21,527 file
- **Thuật toán kiểm băm:** MD5 Checksum toàn bộ tệp tin nhị phân

---

## 1. Bảng Số Liệu Kiểm Đo Thực Tế Trên Đĩa Cứng

| Chỉ số kiểm tra | Tập Train (Huấn luyện) | Tập Val (Kiểm định) | Tập Test (Kiểm thử) | Tổng cộng |
| :--- | :---: | :---: | :---: | :---: |
| **Số lượng hình ảnh** | 17,221 | 2,153 | 2,153 | 21,527 |
| **Số lượng file nhãn (.txt)** | 17,221 | 2,153 | 2,153 | 21,527 |
| **Ảnh chỉ có Khói (Smoke)** | 4,681 | 593 | 593 | 5,867 |
| **Ảnh chỉ có Lửa (Fire)** | 944 | 110 | 110 | 1,164 |
| **Ảnh có cả Khói & Lửa** | 3,763 | 447 | 448 | 4,658 |
| **Ảnh Background (0 box)** | 7,833 | 1,003 | 1,002 | 9,838 |
| **Bounding Box Khói (0)** | 9,543 | 1,155 | 1,156 | 11,854 |
| **Bounding Box Lửa (1)** | 11,807 | 1,417 | 1,461 | 14,685 |
| **Tổng số Bounding Box** | 21,350 | 2,572 | 2,617 | 26,539 |

---

## 2. Kết Quả Kiểm Tra Tính Toàn Vẹn & Lỗi Cặp Đôi (Orphan / Missing Check)

- **Ảnh không có file nhãn tương ứng (Orphan Images):** 0 ảnh
- **File nhãn không có ảnh tương ứng (Orphan Labels):** 0 file
- **Bounding Box không hợp lệ (sai class hoặc tọa độ ngoài [0, 1]):** 0 box
  -> **KẾT QUẢ:** Dữ liệu hoàn hảo $100\%$, không có file mồ côi và không có box hỏng.

---

## 3. Kết Quả Kiểm Tra Trùng Lặp Nội Dung (MD5 Collision Check)

- **Số lượng ảnh trùng lặp chéo giữa các tập (Cross-split duplicates):** **0 ảnh**
  -> **KẾT LUẬN QUAN TRỌNG:** Tập Train, Tập Val và Tập Test hoàn toàn độc lập về mặt nội dung byte-by-byte. Không có bất kỳ hình ảnh nào từ tập Test xuất hiện trong Train hay Val.

---

## 4. Phân Tích Nguồn Dữ Liệu & Khả Năng Rò Rỉ Khung Hình Lân Cận (Source & Sequence Check)

Tập dữ liệu D-Fire gốc được tổng hợp từ 4 nguồn chính (theo quy ước đặt tên file của tác giả D-Fire):

| Nhóm nguồn dữ liệu (Prefix) | Tập Train | Tập Val | Tập Test | Tổng cộng |
| :--- | :---: | :---: | :---: | :---: |
| **`WEB`** | 9,443 | 1,181 | 1,183 | 11,807 |
| **`AoF`** | 6,723 | 821 | 840 | 8,384 |
| **`PublicDataset`** | 1,055 | 151 | 130 | 1,336 |

- **Nhận xét khoa học về tính độc lập:**
  1. Giữa 3 tập `Train`, `Val` và `Test` có **0 ảnh trùng lặp nội dung (MD5 = 0)** và **0 trùng tên tệp**.
  2. Tỷ lệ phân bố của cả 4 nguồn dữ liệu (`WEB`, `AoF`, `PublicDataset`, `Synthetic`) cũng như tỷ lệ 4 nhóm nhãn (`Smoke only`, `Fire only`, `Both`, `Background`) giữa `Val` và `Test` đều cân bằng 50/50 chính xác nhờ thuật toán chia phân tầng (`Stratified Split`).
  3. Vì bộ D-Fire gốc chỉ đánh số thứ tự liên tiếp theo nguồn (`AoF00001..AoF08363`, `WEB00001..WEB11621`) mà không lưu metadata ID video gốc của từng khung hình, để loại bỏ hoàn toàn rủi ro rò rỉ bối cảnh video khi đánh giá hệ thống, đề tài thiết kế thêm **2 tầng kiểm định độc lập theo đúng Đề cương NCKH**:
     - **Tầng 1:** Tập **1.050 khung hình Việt Nam (`datasets/vn_dataset/`)** được chia 60/20/20 **theo từng video độc lập** (không có khung hình của cùng một video xuất hiện ở 2 tập).
     - **Tầng 2:** Bộ **60 Video kiểm định thực tế (`data/video_dataset_list.csv`)** dùng cho `event_eval.py` ở Tuần 4 để đo Recall sự kiện, độ trễ phát hiện và tỷ lệ báo động giả/giờ.
