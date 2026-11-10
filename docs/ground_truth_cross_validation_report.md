# Báo Cáo Quy Trình Kiểm Định Chéo Ground Truth Đa Thẩm Định (Task 1)

**Dự án:** DACN_NCKH - Phát hiện và cảnh báo cháy từ camera giám sát  
**Ngày thực hiện:** 10/11/2026  
**Phiên bản chuẩn hóa:** `Ground Truth Schema v1.1`  
**Branch Git:** `fix/week4-scientific-validation`  

---

## 1. Mục Đích & Nguyên Tắc Khoa Học

Để khắc phục vấn đề tự động nâng trạng thái `pending_verification` thành `verified` dựa thuần túy trên sự tồn tại của tệp video và timestamp hợp lệ, hệ thống tái thiết lập quy trình kiểm định chéo độc lập (Dual-Annotator Cross-Validation Protocol) theo chuẩn nghiên cứu khoa học:
1. **Tuyệt đối không tự động nâng trạng thái:** Mọi sự kiện chỉ có thể được công nhận là cơ sở đánh giá khi có bằng chứng kiểm định độc lập của hai thẩm định viên (`annotator_01` và `annotator_02`).
2. **Không bịa đặt dữ liệu thẩm định:** Nếu chưa có người kiểm định thứ hai thực sự xem xét, sự kiện bắt buộc phải giữ nguyên ở trạng thái `pending_verification` hoặc `annotator_reviewed`.
3. **Phân định rõ 4 trạng thái kiểm định:**
   - `pending_verification`: Sự kiện thô, trích xuất từ metadata hoặc chưa qua kiểm định đầy đủ.
   - `annotator_reviewed`: Đã có thẩm định viên 1 xem xét và đánh dấu khung hình onset/offset.
   - `cross_verified`: Đã có thẩm định viên 2 kiểm tra độc lập, độ lệch $|t_1 - t_2| \le 0.5\text{ giây}$.
   - `disputed`: Có tranh chấp về nhãn hoặc độ lệch thời gian giữa hai thẩm định viên $> 0.5\text{ giây}$.

---

## 2. Thống Kê Hiện Trạng Bộ Nhãn Ground Truth (Phiên Bản v1.1)

- **Tổng số sự kiện:** 35 sự kiện trên 24 video dương tính vật lý (`FireSmokeLabeler/videos/*.avi`).
- **Phân loại nhãn độc lập:** 16 sự kiện `FIRE` và 19 sự kiện `SMOKE`.
- **Số sự kiện `cross_verified`:** **11 sự kiện** trên 8 video (`POS_02`, `POS_03`, `POS_04`, `POS_05`, `POS_10`, `POS_12`, `POS_23`, `POS_24`).
  - Validation Split: 3 sự kiện (`POS_03`, `POS_05`, `POS_23`).
  - Locked Test Split: 8 sự kiện (`POS_02`, `POS_04`, `POS_10`, `POS_12`, `POS_24`).
  - Sai lệch giữa 2 annotator: $0.0\text{ giây} \le 0.5\text{ giây}$.
- **Số sự kiện `pending_verification`:** **24 sự kiện** trên 16 video (`POS_01`, `POS_06`, `POS_07`, `POS_08`, `POS_09`, `POS_11`, `POS_13`, `POS_14`, `POS_15`, `POS_16`, `POS_17`, `POS_18`, `POS_19`, `POS_20`, `POS_21`, `POS_22`).
  - Trạng thái: Giữ nguyên `pending_verification` vì chưa có thẩm định viên thứ hai kiểm định chéo độc lập.
  - Các sự kiện này bị loại trừ hoàn toàn khỏi bộ Ground Truth tính toán Recall/Precision khi chạy chính sách `verified_only`.

---

## 3. Quản Lý Snapshot & Checksum Toàn Vẹn

1. **Snapshot v1.0 (Lưu trữ lịch sử đối chiếu):**
   - Đường dẫn: [`data/ground_truth/events_v1.0_snapshot.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events_v1.0_snapshot.jsonl)
   - SHA256: `1271a9badf316813d44ca4abccb959fb6c59bd088da2c6e462c7b09573da1133`
2. **Snapshot v1.1 (Chuẩn hóa đa thẩm định):**
   - Đường dẫn: [`data/ground_truth/events_v1.1_snapshot.jsonl`](file:///d:/workspace/DACN_NCKH/data/ground_truth/events_v1.1_snapshot.jsonl)
   - Tệp Checksum: [`data/ground_truth/checksums_v1.1.sha256`](file:///d:/workspace/DACN_NCKH/data/ground_truth/checksums_v1.1.sha256)
   - SHA256: `8c6f30bbd057d5b5ae453feb64eb010387c2087ab9350ccea922ba04325bf7c3`

