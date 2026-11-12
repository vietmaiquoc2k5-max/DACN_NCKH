# BÁO CÁO KIỂM TOÁN GROUND TRUTH VÀ KẾ HOẠCH XÁC MINH ĐỘC LẬP
**Dự án:** Computer Vision-based Fire and Smoke Detection and Early Warning from Surveillance Camera Footage  
**Tập tin kiểm toán:** `data/ground_truth/events.jsonl` và `runs/benchmarks/week4_final/ground_truth/ground_truth_master.csv`  
**Ngày lập:** 12/11/2026  
**Thành viên thực hiện:** Mai Quốc Việt & Lê Minh Thiên  
**Trạng thái tiêu chí Ground Truth:** **`PARTIAL` (HOÀN THÀNH MỘT PHẦN / CHỜ XÁC MINH ĐỘC LẬP 24 SỰ KIỆN)**  

---

## 1. Hiện Trạng Ground Truth Độc Lập

Theo quy định liêm chính khoa học của đề cương NCKH, một sự kiện Ground Truth chỉ được xem là đạt chuẩn (`cross_verified`) khi và chỉ khi có sự xác nhận độc lập (blind review) từ ít nhất **hai người gán nhãn độc lập** (Annotator 1: Mai Quốc Việt, Annotator 2: Lê Minh Thiên), không có sự trao đổi trước về mốc giây bắt đầu để tránh thiên lệch chủ quan.

Thống kê hiện trạng trong `data/ground_truth/events.jsonl`:
- **Tổng số sự kiện:** **35 sự kiện** trên 24 video dương tính chuẩn hóa (`POS_01` đến `POS_24`).
- **Đã kiểm chứng chéo (`cross_verified`):** **11 sự kiện (31,4%)** trên 8 video (`POS_02, POS_03, POS_04, POS_05, POS_10, POS_12, POS_23, POS_24`).
- **Đang chờ xác minh độc lập (`pending_verification`):** **24 sự kiện (68,6%)** trên 16 video.

> [!IMPORTANT]
> **Nguyên tắc liêm chính khoa học:** Hệ thống TUYỆT ĐỐI KHÔNG tự động chuyển trạng thái `pending_verification` thành `cross_verified` bằng script. Mọi sự thay đổi phải xuất phát từ hành động kiểm chứng thật của con người (Human-in-the-loop). Do đó, hạng mục Ground Truth được xếp loại **`PARTIAL`** cho đến khi Annotator 2 hoàn thành thẩm định.

---

## 2. Danh Sách 24 Sự Kiện Chờ Thẩm Định Độc Lập (Action Checklist cho Lê Minh Thiên)

Dưới đây là bảng phân công chi tiết 24 sự kiện cần Annotator thứ hai (Lê Minh Thiên) rà soát độc lập bằng công cụ VLC / OpenCV Player, ghi nhận mốc giây ngọn lửa/khói xuất hiện thực tế và đối chiếu với mốc đề xuất:

| STT | Video ID | Đường dẫn video | Nhãn | Mốc đề xuất ($T_{start} \rightarrow T_{end}$) | Annotator 1 (Đã gán) | Annotator 2 (Cần xác nhận) | Trạng thái hiện tại | Hướng dẫn kiểm tra chéo |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `POS_01` | `datasets/test_videos/pos/posVideo1.868.avi` | FIRE | 2.4s $\rightarrow$ 29.3s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Quan sát ngọn lửa bùng lên ở giây thứ 2 |
| **2** | `POS_01` | `datasets/test_videos/pos/posVideo1.868.avi` | SMOKE | 6.8s $\rightarrow$ 29.3s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Quan sát luồng khói bốc lên sau lửa |
| **3** | `POS_06` | `datasets/test_videos/pos/posVideo6.873.avi` | FIRE | 0.1s $\rightarrow$ 55.22s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Lửa xuất hiện ngay đầu clip |
| **4** | `POS_06` | `datasets/test_videos/pos/posVideo6.873.avi` | SMOKE | 34.6s $\rightarrow$ 55.22s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Khói xám xuất hiện ở nửa sau |
| **5** | `POS_07` | `datasets/test_videos/pos/posVideo7.874.avi` | FIRE | 0.1s $\rightarrow$ 80.28s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Lửa nền xuất hiện từ frame đầu |
| **6** | `POS_08` | `datasets/test_videos/pos/posVideo8.875.avi` | FIRE | 0.4s $\rightarrow$ 25.8s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Bắt lửa nhanh góc dưới |
| **7** | `POS_08` | `datasets/test_videos/pos/posVideo8.875.avi` | SMOKE | 8.8s $\rightarrow$ 25.8s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Khói bốc cao sau 8 giây |
| **8** | `POS_09` | `datasets/test_videos/pos/posVideo9.876.avi` | FIRE | 1.2s $\rightarrow$ 54.7s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Lửa cháy lan từ giây 1.2 |
| **9** | `POS_11` | `datasets/test_videos/pos/posVideo11.878.avi` | FIRE | 0.1s $\rightarrow$ 22.12s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Đám cháy trong nhà xưởng |
| **10** | `POS_11` | `datasets/test_videos/pos/posVideo11.878.avi` | SMOKE | 16.1s $\rightarrow$ 22.12s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Khói dày đặc ở cuối video |
| **11** | `POS_13` | `datasets/test_videos/pos/posVideo13.880.avi` | FIRE | 0.1s $\rightarrow$ 57.66s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Đám cháy ngoài trời |
| **12** | `POS_13` | `datasets/test_videos/pos/posVideo13.880.avi` | SMOKE | 0.1s $\rightarrow$ 57.66s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Khói đen cuộn ngay từ đầu |
| **13** | `POS_14` | `datasets/test_videos/pos/posVideo14.881.avi` | FIRE | 12.9s $\rightarrow$ 21.47s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Ngọn lửa phát bùng ở giây 12.9 |
| **14** | `POS_14` | `datasets/test_videos/pos/posVideo14.881.avi` | SMOKE | 1.3s $\rightarrow$ 21.47s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Khói xuất hiện trước lửa |
| **15** | `POS_15` | `datasets/test_videos/pos/posVideo15.882.avi` | FIRE | 64.4s $\rightarrow$ 78.56s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Lửa bốc mạnh ở cuối clip |
| **16** | `POS_15` | `datasets/test_videos/pos/posVideo15.882.avi` | SMOKE | 0.2s $\rightarrow$ 78.56s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Khói âm ỉ từ đầu clip |
| **17** | `POS_16` | `datasets/test_videos/pos/posVideo16.883.avi` | SMOKE | 1.7s $\rightarrow$ 109.58s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Clip chỉ có khói, không có lửa |
| **18** | `POS_17` | `datasets/test_videos/pos/posVideo17.884.avi` | SMOKE | 0.4s $\rightarrow$ 20.1s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Khói trắng bốc lên |
| **19** | `POS_18` | `datasets/test_videos/pos/posVideo18.885.avi` | SMOKE | 11.2s $\rightarrow$ 70.1s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Khói xuất hiện sau 11 giây |
| **20** | `POS_19` | `datasets/test_videos/pos/posVideo19.886.avi` | SMOKE | 1.2s $\rightarrow$ 24.2s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Khói bốc cao trong rừng |
| **21** | `POS_20` | `datasets/test_videos/pos/posVideo20.887.avi` | SMOKE | 0.1s $\rightarrow$ 27.6s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Khói lan tỏa |
| **22** | `POS_21` | `datasets/test_videos/pos/posVideo21.888.avi` | FIRE | 37.6s $\rightarrow$ 49.05s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Lửa xuất hiện ở giây 37.6 |
| **23** | `POS_21` | `datasets/test_videos/pos/posVideo21.888.avi` | SMOKE | 0.1s $\rightarrow$ 49.05s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Khói xuất hiện từ đầu |
| **24** | `POS_22` | `datasets/test_videos/pos/posVideo22.889.avi` | SMOKE | 0.6s $\rightarrow$ 27.12s | Mai Quốc Việt | Lê Minh Thiên | `pending_verification` | Khói bốc lên ở góc xa |

---

## 3. Tách Biệt Kết Quả Đánh Giá Hiệu Năng (Evaluation Splits)

Để bảo đảm tính minh bạch khi báo cáo hội nghị và bảo vệ luận án:

### Phân hệ 1: Strict Verified Evaluation (Chỉ dùng 11 sự kiện `cross_verified`)
- Đây là cơ sở đánh giá chính thức có giá trị bằng chứng khoa học cao nhất tại thời điểm hiện tại.
- **Tập dữ liệu:** 8 video đã có xác thực độc lập 2 bên (`POS_02, POS_03, POS_04, POS_05, POS_10, POS_12, POS_23, POS_24`).
- **Kết quả:**
  - Video-level Recall: **100% (8/8 video phát hiện thành công)**.
  - Event Instance Recall: **81,82% (9/11 events)**.
  - Timely Detections (TP): 9 sự kiện.
  - Missed Detections (FN): 2 sự kiện.
  - Mean Detection Delay: **0,98 giây** (rất nhanh, đạt chuẩn $\le 10$s).

### Phân hệ 2: Exploratory Full Evaluation (Toàn bộ 35 sự kiện gồm cả 24 `pending`)
- Đây là đánh giá khảo sát toàn cảnh giả định mốc thời gian đề xuất của Annotator 1 là đúng.
- **Tập dữ liệu:** 24 video dương tính chuẩn hóa.
- **Kết quả:**
  - Video-level Recall: **95,83% (23/24 video)**.
  - Event Instance Recall: **71,43% (25/35 events)**.
  - Event Precision: **80,65%**.
  - Event F1-Score: **75,76%**.
  - Mean Detection Delay: **2,13 giây**.

---

## 4. Hành Động Tiếp Theo & Blocker Cần Giải Quyết

1. **Người dùng (Lê Minh Thiên):** Mở 16 video trong bảng mục 2, kiểm tra mốc thời gian độc lập và ghi nhận vào cột `Annotator 2`.
2. Nếu đồng thuận (sai lệch $\le 1.0$ giây): Cập nhật `verification_status` từ `pending_verification` thành `cross_verified`.
3. Nếu có bất đồng (sai lệch $> 1.0$ giây): Họp trao đổi và lấy mốc trung bình hoặc mốc sớm nhất khi ngọn lửa/khói rõ ràng.
4. Sau khi 100% của 24 sự kiện chuyển thành `cross_verified`, tiến hành chạy khóa locked test set một lần duy nhất để chuyển Operating Point từ `PROVISIONAL` sang **`OFFICIAL`**.

