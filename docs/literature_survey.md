# Báo Cáo Khảo Sát Tài Liệu Nghiên Cứu (Literature Survey)

**Đề tài:** Nghiên cứu ứng dụng computer vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
**Tác giả:** Lê Minh Thiên, Mai Quốc Việt  
**Giảng viên hướng dẫn:** ThS. Nguyễn Hữu Trung  
**Thời gian hoàn thành:** Tuần 1 (19/10/2026 – 25/10/2026)  

---

## 1. Đặt Vấn Đề & Thách Thức Trong Camera Giám Sát CCTV

Hệ thống báo cháy truyền thống (cảm biến khói quang điện, cảm biến nhiệt, ion hóa) hoạt động hiệu quả trong không gian kín nhưng bộc lộ nhiều hạn chế nghiêm trọng trong không gian rộng (nhà kho, bãi đỗ xe, xưởng sản xuất, khuôn viên ngoài trời):
- **Độ trễ cao:** Khói và nhiệt phải khuếch tán tới vị trí cảm biến mới kích hoạt báo động.
- **Không có bằng chứng trực quan:** Người vận hành không thể xác thực ngay tình huống khẩn cấp từ xa.

Việc ứng dụng Thị giác máy tính (Computer Vision) trên hạ tầng camera giám sát sẵn có khắc phục triệt để các nhược điểm trên nhờ khả năng phát hiện tức thì (ngay từ giai đoạn bốc khói) và truyền tải hình ảnh snapshot trực quan. Tuy nhiên, thách thức lớn nhất là **tỷ lệ báo động giả (False Alarm Rate)** do ánh sáng phản chiếu, đèn xe, hoàng hôn, khói nấu ăn, và sự thay đổi góc quay/độ phân giải camera.

---

## 2. Phân Tích & So Sánh Các Bộ Phát Hiện (Detection Models & Baselines)

Theo đề cương NCKH (Mục tiêu MT2), nghiên cứu triển khai khảo sát và so sánh $\ge$ 4 mô hình học sâu hiện đại cùng 2 baseline tham chiếu:

### 2.1. Hai phương pháp Baseline (Mốc đối chứng)
* **Baseline 1 (B1 - Phương pháp xử lý ảnh truyền thống):**
  - *Cơ chế:* Kết hợp luật màu không gian HSV / YCbCr (đặc trưng sắc độ của ngọn lửa và khói xám) với thuật toán trừ nền thích nghi GMM (Gaussian Mixture Model / MOG2) để tách chuyển động.
  - *Ưu điểm:* Tiêu tốn cực ít tài nguyên tính toán, chạy thời gian thực trên CPU yếu.
  - *Nhược điểm:* Cực kỳ nhạy cảm với sự thay đổi ánh sáng đột ngột, đèn pha ô tô ban đêm hoặc vật thể chuyển động màu đỏ/vàng.
* **Baseline 2 (B2 - Mạng phân loại nhẹ MobileNetV3):**
  - *Cơ chế:* Trích xuất đặc trưng và phân loại ảnh cấp khung hình (frame-level classification) hoặc vùng ứng viên bằng kiến trúc MobileNetV3-Small.
  - *Ưu điểm:* Tốc độ suy luận cao, phù hợp biên (edge devices).
  - *Nhược điểm:* Không định vị được tọa độ hộp bao (bounding box) chính xác, khó phân biệt đám cháy cục bộ trong khung cảnh rộng.

### 2.2. Các bộ phát hiện hiện đại (State-of-the-Art Detectors)
* **YOLOv8 (Ultralytics - v8n, v8s):**
  - Kiến trúc Anchor-free với C2f module, tách biệt nhánh classification và regression (Decoupled Head).
  - Tối ưu hóa cực tốt cho ứng dụng thời gian thực, độ nhạy cao với các đốm lửa nhỏ.
* **YOLO11 (Ultralytics - 11n, 11s):**
  - Cải tiến với kiến trúc C3k2, SPPF nâng cấp và cơ chế Attention nhẹ C2PSA.
  - Tăng độ chính xác trích xuất đặc trưng khói mỏng và khói phân tán trong điều kiện ánh sáng phức tạp.
* **RT-DETR-R18 (Real-Time DEtection TRansformer):**
  - Mô hình Transformer phát hiện đối tượng thời gian thực đầu tiên, loại bỏ hoàn toàn hậu xử lý NMS (Non-Maximum Suppression).
  - Hiệu quả vượt trội trong việc nắm bắt tương quan ngữ cảnh toàn cục (global context), giúp phân biệt khói với sương mù/mây trời.

---

## 3. Kỹ Thuật Xác Nhận Theo Thời Gian (Temporal Verification - MT3)

Khác biệt cốt lõi giữa hệ thống nghiên cứu này và các bài báo phát hiện từng khung hình (frame-by-frame detector) truyền thống nằm ở module **Xác nhận theo thời gian** (`temporal_verifier.py`):

1. **Bộ theo dõi đa đối tượng (ByteTrack):**
   - Sử dụng thuật toán ByteTrack kết hợp bộ lọc Kalman Filter để liên kết các hộp bao qua từng khung hình (Object Association).
   - Duy trì ID duy nhất cho từng đám cháy/vùng khói, lọc bỏ các phát hiện chập chờn chỉ xuất hiện 1–2 khung hình (đèn flash, phản xạ gương).
2. **Cơ chế Cửa sổ trượt $k/N$ khung hình:**
   - Cảnh báo chỉ được kích hoạt khi đối tượng được phát hiện tối thiểu trong $k$ khung hình trên tổng số $N$ khung hình gần nhất ($k/N$).
3. **Phân tích biến động đặc trưng (Feature Dynamics):**
   - Ngọn lửa và khói có đặc tính động học đặc trưng: diện tích hộp bao có xu hướng mở rộng bất đối xứng, chu vi biến thiên liên tục và mật độ màu sắc dao động (flickering effect).
   - Module tính toán độ lệch chuẩn diện tích và tỷ lệ co giãn để loại trừ các nguồn sáng tĩnh (biển quảng cáo, đèn đường).
4. **Vùng loại trừ tĩnh (ROI Masking):**
   - Thiết lập vùng mặt nạ loại trừ cho từng camera riêng biệt (ví dụ: bếp nấu ăn cố định, đèn tín hiệu giao thông).
5. **Cơ chế 2 mức cảnh báo:**
   - **Mức 1: "Nghi ngờ" (Suspect / Yellow Alert)** - Khi thỏa điều kiện $k_1/N$, hệ thống ghi nhận và tăng tần suất kiểm tra.
   - **Mức 2: "Báo cháy" (Fire Alert / Red Alert)** - Khi thỏa mãn điều kiện $k_2/N$ kết hợp biến động diện tích $\rightarrow$ Kích hoạt còi hú và gửi Telegram/email.

---

## 4. Kỹ Thuật Thích Nghi Miền (Domain Adaptation - MT4)

### 4.1. Vấn đề Chệch Miền (Domain Shift)
Tập huấn luyện chính **D-Fire** gồm các hình ảnh thu thập ở môi trường phương Tây/mở với điều kiện ánh sáng, độ phân giải và góc máy khác biệt lớn so với hệ thống camera giám sát thực tế tại Việt Nam (thường có độ phân giải 720p/1080p, góc nhìn từ trên cao xuống, nhiều xe máy và bối cảnh đường phố nhiều bảng hiệu).

### 4.2. Ba chiến lược thích nghi miền thực nghiệm:
1. **Fine-tuning toàn bộ (Full Fine-tuning):** Cập nhật toàn bộ trọng số mạng trên tập dữ liệu Việt Nam.
2. **Đóng băng Backbone (Freeze Backbone):** Giữ nguyên tầng trích xuất đặc trưng cơ bản (Backbone), chỉ tinh chỉnh tầng cổ (Neck) và đầu phát hiện (Head).
3. **Đóng băng + Replay 20% D-Fire:** Kết hợp đóng băng tầng dưới với việc nạp lại 20% dữ liệu D-Fire trong quá trình huấn luyện nhằm chống hiện tượng quên thảm khốc (Catastrophic Forgetting).
*(Cơ sở lý thuyết tham khảo từ nguyên lý Nested Learning: phân tầng biểu diễn tri thức ổn định và tri thức thích nghi).*

---

## 5. Danh Mục Tài Liệu Tham Khảo (IEEE References)

1. **J. T. Gaias et al.**, *"D-Fire: An image dataset for fire and smoke detection,"* IEEE Latin America Transactions, 2021.
2. **C. Li et al.**, *"YOLOv8: Real-Time Object Detection,"* Ultralytics GitHub Repository, 2023.
3. **W. Lv et al.**, *"Detrs beat yolos on real-time object detection,"* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
4. **Y. Zhang et al.**, *"ByteTrack: Multi-Object Tracking by Associating Every Detection Box,"* European Conference on Computer Vision (ECCV), 2022.
5. **T. H. Chen, P. H. Wu, and Y. C. Chiou**, *"An early fire-detection method based on image processing,"* IEEE International Conference on Image Processing (ICIP), 2004.
6. **A. Howard et al.**, *"Searching for MobileNetV3,"* IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
7. **P. F. Felzenszwalb et al.**, *"Object Detection with Discriminatively Trained Part-Based Models,"* IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 32, no. 9, 2010.
8. **Z. Wang et al.**, *"Towards Real-Time Multi-Object Tracking,"* European Conference on Computer Vision (ECCV), 2020.
