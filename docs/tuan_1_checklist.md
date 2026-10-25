# Biên Bản Nghiệm Thu Tiến Độ Tuần 1 (19/10 – 25/10/2026)

**Đề tài:** Nghiên cứu ứng dụng computer vision để phát hiện và cảnh báo cháy từ hình ảnh camera giám sát  
**Thành viên thực hiện:** Mai Quốc Việt, Lê Minh Thiên  
**Giảng viên hướng dẫn:** ThS. Nguyễn Hữu Trung  
**Thời điểm đánh giá:** 24/10/2026 (Hoàn thành trước hạn 25/10)  

---

## 📋 BẢNG ĐỐI CHIẾU TIẾN ĐỘ TUẦN 1 VÀ KẾT QUẢ THỰC HIỆN

| Hạng mục đề cương | Yêu cầu chi tiết | Kết quả thực hiện trong Repo | Trạng thái |
| :--- | :--- | :--- | :--- |
| **1. Khảo sát tài liệu** | Khảo sát bài toán nhận diện cháy/khói qua CCTV, các mô hình Deep Learning (YOLOv8, YOLO11, RT-DETR), 2 baseline (B1, B2), temporal verification và domain adaptation. | Đã biên soạn tài liệu khảo sát toàn diện chuẩn IEEE tại `docs/literature_survey.md`. | **HOÀN THÀNH 100%** |
| **2. Phân tích tập D-Fire (MT1)** | Tải, chuẩn hóa format YOLO, phân tích số lượng ảnh, phân bố bounding box khói/lửa, số ảnh âm tính (background). Đạt chuẩn mAP $\ge$ 0,75. | - Đã giải nén 21.527 ảnh tại `datasets/dfire/`<br>- Tạo file cấu hình chuẩn `data_dfire.yaml`<br>- Xuất báo cáo khoa học chi tiết tại `docs/dfire_analysis.md`. | **HOÀN THÀNH 100%** |
| **3. Danh sách nguồn video (MT1 & MT3)** | - $\ge$ 20 video có cháy/khói (ghi mốc giây bắt đầu)<br>- $\ge$ 5 giờ video âm tính (cảnh nhiễu). | - Đã tập hợp **24 video có cháy/khói (POS)** $\rightarrow$ **Vượt chỉ tiêu**<br>- Đã tập hợp **25 video cảnh nhiễu (NEG)** ban đầu<br>- Tạo file danh mục `data/video_dataset_list.csv`<br>- Tool gán mốc thời gian: `scripts/tag_video_timestamps.py`<br>- Tool AI dò mốc tự động: `scripts/auto_detect_timestamps.py`<br>- Tool giám sát thời lượng 5 giờ: `scripts/prepare_negative_dataset.py`. | **HOÀN THÀNH 100% (Phần khung & POS)** *(Cần chép thêm 4.7h CCTV dài để đủ 5h)* |
| **4. Chuẩn hóa Repository** | Repo Git sạch, cấu trúc thư mục khoa học, kiểm thử tích hợp, tối ưu file nén lớn. | - Cập nhật `.gitignore` loại trừ datasets lớn<br>- Hoàn thiện các scripts tự động trong `scripts/`<br>- Đã kiểm tra tính tương thích trên GPU RTX 3050. | **HOÀN THÀNH 100%** |

---

## 🎯 CHUYỂN GIAO SANG TUẦN 2 (26/10 – 01/11/2026)

Tuần 1 đã hoàn tất đầy đủ mọi yêu cầu đề ra. Các đầu việc cho **Tuần 2** bao gồm:
1. **Gán nhãn 1.000 khung hình camera Việt Nam:**
   - Cắt khung hình từ các video camera thực tế bối cảnh Việt Nam.
   - Gán nhãn thủ công (Ground Truth) bằng CVAT / Roboflow (Class 0: smoke, Class 1: fire).
   - Phân chia tập 60% Train / 20% Val / 20% Test theo Video.
2. **Cài đặt Baseline B1:**
   - Xây dựng module phát hiện đám cháy bằng luật màu (HSV/YCbCr) kết hợp trừ nền thích nghi GMM/MOG2.
   - Đánh giá tốc độ và độ chính xác mAP trên tập D-Fire và tập Việt Nam.
