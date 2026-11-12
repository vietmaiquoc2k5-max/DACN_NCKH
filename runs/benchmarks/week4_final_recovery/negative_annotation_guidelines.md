# Hướng Dẫn Thẩm Định Độc Lập Dữ Liệu Video Âm Tính (Negative Ground Truth Protocol)

## 1. Mục Đích
Thẩm định tính chất âm tính độc lập (Ground Truth) cho 79 luồng video giám sát thực tế nhằm đảm bảo tính khoa học tuyệt đối khi đo lường Tỷ Lệ Báo Động Giả (False Alarm Rate - FAR).

## 2. Nguyên Tắc Bắt Buộc
1. **Độc lập hoàn toàn với AI:** Người thẩm định (Annotator) chỉ quan sát trực quan tệp video nguồn gốc, tuyệt đối không được xem trước kết quả dự đoán (bounding boxes / alert timestamps) của mô hình YOLO hay ByteTrack.
2. **Quy tắc 2 Thẩm Định Viên Độc Lập (Dual-Annotator Consensus):**
   - Annotator 01 và Annotator 02 gán nhãn độc lập.
   - Trạng thái `VERIFIED_NEGATIVE` chỉ được công nhận khi cả hai Annotator đều đồng thuận rằng toàn bộ luồng video không chứa khói hoặc lửa thật.
3. **Phân Loại Nhãn Semantics:**
   - `VERIFIED_NEGATIVE`: 100% video hoàn toàn không có khói/cháy.
   - `CONTAINS_FIRE`: Có ngọn lửa thật (ghi rõ mốc giây bắt đầu và kết thúc).
   - `CONTAINS_SMOKE`: Có khói cháy thật (ghi rõ mốc giây).
   - `UNCERTAIN`: Hình ảnh mờ, khuất tầm nhìn, nghi ngờ cần hội chẩn (Adjudication).
   - `PENDING_REVIEW`: Đang chờ thẩm định viên xem xét.
4. **Xử Lý Nhiễu Quang Học (Nhiễu Không Phải Cháy):**
   - Đèn xe, ánh hoàng hôn, đèn pha rọi, bóng nước phản chiếu, hơi nước công nghiệp, bụi mù, tia lửa hàn điện... phải được ghi nhận rõ là CẢNH NHIỄU ÂM TÍNH (Hard Negatives), không được gán nhãn cháy.
