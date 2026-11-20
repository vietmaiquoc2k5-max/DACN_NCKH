# BÁO CÁO ĐỐI SOÁT GIAO THỨC DỮ LIỆU ÂM TÍNH VÀ THỜI LƯỢNG PHƠI NHIỄM FAR TUẦN 6
## (WEEK 6 PROTOCOL RECONCILIATION & SCIENTIFIC INTEGRITY PROPOSAL)

- **Đơn vị thực hiện:** FireWatch AI Research & MLOps Team
- **Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ camera giám sát (DACN_NCKH)
- **Thời điểm lập:** 20/11/2026
- **Git Branch:** `fix-4` | **Base Commit SHA:** `3701d2e03b899778fe8ed5fa2f7d4f84527b4187`
- **Tình trạng Cổng Validation Gate:** **`BLOCKED` (BẮT BUỘC KHÓA CHẶT)**

---

## 1. TỔNG QUAN VÀ BỐI CẢNH VẤN ĐỀ

Trong khuôn khổ Đề cương NCKH được phê duyệt:
- **Chỉ tiêu MT1 (Dữ liệu):** $\ge 5$ giờ video không cháy có cảnh gây nhiễu (Negative Videos with Hard Negatives / Distractors).
- **Chỉ tiêu MT3 (Giảm báo giả):** Giảm $\ge 50\%$ tỷ lệ báo động giả mỗi giờ (FAR - False Alarm Rate per hour) so với cảnh báo từng khung hình; độ trễ trung bình $\le 10$ giây; độ suy giảm recall sự kiện $\le 5$ điểm $\%$.

Trước đợt kiểm toán Task 7, tồn tại sự mập mờ phương pháp luận nghiêm trọng:
1. Gộp toàn bộ 79 video âm tính hiện có trong ổ đĩa ($\approx 7.13$ giờ) thành một mẫu số chung để tuyên bố "đã vượt chỉ tiêu 5.0 giờ của Đề cương".
2. Bỏ qua sự thật khoa học rằng trong 79 video đó:
   - **37 video** ($\approx 3.45$ giờ) đã bị cắt khung hình đưa vào tập huấn luyện `vn_dataset` (Train-contaminated).
   - **13 video** ($\approx 2.10$ giờ) là tập kiểm định độc lập (Independent Test Set / Held-Out) dành riêng cho nghiệm thu cuối cùng ở Tuần 8.
   - **Chỉ có 13 video** ($\approx 1.43$ giờ) thực sự thuộc tập `val` (Validation Set) được phép dùng để tinh chỉnh tham số và chọn Operating Point.

Báo cáo này phân tích chi tiết toán học và phương pháp luận, đồng thời thiết lập ranh giới bảo vệ khoa học tuyệt đối.

---

## 2. PHÂN TẦNG ĐỘC LẬP 5 PHẠM VI DỮ LIỆU ÂM TÍNH (THE 5 DISJOINT SCOPES)

Toàn bộ 79 video âm tính ($25.672,14$ giây $\approx 7,1312$ giờ) được hệ thống kiểm toán kỹ thuật `scripts/audit_negative_video_technical_integrity.py` tách biệt thành 5 phân tầng không trùng lặp:

| Phân tầng phạm vi (Scope) | Số video | Thời lượng (giây) | Thời lượng (giờ) | Tỷ trọng | Mục đích & Trạng thái bảo vệ khoa học |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Validation FAR Exposure (`val`)** | **13** | **$5.131,32$ s** | **$1,4254$ h** | $20,0\%$ | **Duy nhất được phép đưa vào mẫu số FAR để chọn Operating Point Tuần 6.** Hiện còn thiếu **$3,5746$ giờ** so với mốc 5.0h. |
| **2. Independent Test Set (`test`)** | **13** | **$7.550,93$ s** | **$2,0975$ h** | $29,4\%$ | **Held-Out Test Set khóa chặt cho nghiệm thu Tuần 8.** Tuyệt đối cấm gộp vào Validation để làm đẹp số liệu. |
| **3. Train-Contaminated (`train`)** | **37** | **$12.405,64$ s** | **$3,4460$ h** | $48,3\%$ | **Bị loại trừ 100% khỏi mẫu số FAR.** Các video này có khung hình nằm trong tập Train của mô hình; đo FAR trên đây sẽ gây gian lận vì mô hình đã "học thuộc" bối cảnh. |
| **4. Unresolved / Held-Out Inventory** | **16** | **$584,25$ s** | **$0,1623$ h** | $2,3\%$ | Các clip ngắn/bổ sung chưa hoàn thiện định danh phân chia; giữ cách ly an toàn. |
| **5. Overall Physical Inventory** | **79** | **$25.672,14$ s** | **$7,1312$ h** | $100,0\%$ | Tổng kho tài nguyên âm tính vật lý trên đĩa lưu trữ. |

---

## 3. NGUYÊN TẮC KHOA HỌC KHÔNG THỂ THỎA HIỆP (NON-NEGOTIABLE SCIENTIFIC PRINCIPLES)

### 3.1. Nguyên tắc chống rò rỉ tập kiểm định (No Test-to-Validation Leakage)
- Trong học máy thực nghiệm, **Test Set** phản ánh khả năng khái quát hóa trên dữ liệu hoàn toàn chưa từng thấy.
- Nếu lấy 13 video Test ($2,10$ giờ) gộp vào Validation ($1,43$ giờ) để đạt $3,53$ giờ, hoặc gộp thêm để ép cho đủ $5,0$ giờ:
  - Đây là hành vi **Data Snooping / Information Leakage** nghiêm trọng trong khoa học.
  - Sẽ làm mất tính vô tư của tập Test khi nghiệm thu IEEE/Đề cương ở Tuần 8.
- **Quy tắc:** Mọi script đánh giá và Gate Auditor phải cô lập $100\%$ biến `test_far_eligible_duration_sec` khỏi `val_far_eligible_duration_sec`.

### 3.2. Nguyên tắc loại trừ rò rỉ tập huấn luyện (No Train-to-Validation Leakage)
- 37 video âm tính ($3,45$ giờ) đã được trích xuất hình ảnh để dạy cho detector nhận biết cảnh không cháy.
- Nếu đưa các video này vào mẫu số đo FAR:
  - Mô hình sẽ phát ra rất ít báo động giả trên các video này do hiện tượng ghi nhớ (memorization/overfitting).
  - FAR đo được sẽ bị dìm xuống mức thấp giả tạo, không phản ánh năng lực lọc nhiễu thực tế khi triển khai camera thực.
- **Quy tắc:** Gán nhãn `TRAIN_SPLIT_LEAKAGE_RISK` và tự động đánh trượt quyền tham gia mẫu số FAR cho toàn bộ 37 video này.

### 3.3. Nguyên tắc thẩm định kép bởi con người (Dual-Human Consensus)
- Bất kỳ video nào đưa vào mẫu số FAR phải được con người xem đủ $100\%$ thời lượng và xác nhận không có bất kỳ ngọn lửa/tàn lửa/khói nguy hiểm nào (`attest_full_video_viewed = TRUE`).
- Bất kỳ sự kiện dương tính nào đưa vào tập Validation phải có sự đồng thuận độc lập giữa Annotator 1 và Annotator 2 với độ lệch thời điểm bắt đầu cháy $\Delta t_{start} \le 0,5$ giây.

---

## 4. PHÂN TÍCH THÂM HỤT THỜI LƯỢNG VÀ TRẠNG THÁI GATE

$$\text{Shortfall}_{\text{Validation}} = 18.000\text{ s} - 5.131,32\text{ s} = 12.868,68\text{ s} \approx 3,5746\text{ giờ}$$

### Đánh giá trạng thái Validation Gate:
1. **Phơi nhiễm Validation âm tính:** Hiện có $1,43$ giờ trên tổng số $5,0$ giờ yêu cầu $\rightarrow$ **THÂM HỤT $71,5\%$.**
2. **Thẩm định con người trên tập Validation âm tính:** 13 video đang ở trạng thái `PENDING_HUMAN_REVIEW` trong worksheet kiểm toán $\rightarrow$ thời lượng xác nhận thực tế hợp lệ hiện tại là **$0,0$ giờ.**
3. **Thẩm định kép dương tính:** 5 sự kiện dương tính của Validation vẫn đang chờ nộp biên bản độc lập từ Annotator 2 $\rightarrow$ **CHƯA CÓ GROUND TRUTH ĐỒNG THUẬN.**
4. **Lỗi kỹ thuật:** Video `VN_NEG_48` bị đứt luồng $70\%$ (`TRUNCATED_STREAM`) phải tiếp tục loại bỏ.

$\Rightarrow$ **KẾT LUẬN: CỔNG VALIDATION GATE PHẢI BẢO LƯU TRẠNG THÁI `BLOCKED` (KHÓA CHẶT).**

---

## 5. PHƯƠNG ÁN HÀNH ĐỘNG VÀ ĐỀ XUẤT CHO BAN CHỦ NHIỆM

Để unblock cổng nghiệm thu một cách chuẩn mực khoa học mà không vi phạm quy tắc phân tầng dữ liệu:

1. **Giai đoạn 1 (Ngay lập tức):**
   - Ký duyệt và khóa cứng 13 video Test ($2,10$ giờ) vào két bảo mật (Held-out Vault).
   - Tổ chức cho chuyên viên thẩm định rà soát $100\%$ thời lượng của 13 video Validation âm tính ($1,43$ giờ) để kích hoạt tính hợp lệ nội dung.
   - Hoàn thành nộp bảng chấm độc lập của Annotator 2 cho 5 sự kiện Validation dương tính.

2. **Giai đoạn 2 (Thu thập bổ sung):**
   - Thu thập thêm $\ge 3,6$ giờ video âm tính thực tế tại Việt Nam (các cảnh dễ gây báo giả: đèn pha ô tô ban đêm, ánh nắng phản chiếu cửa kính, người mặc áo phản quang, hơi nước nhà tắm/nồi hơi, bóng râm lay động).
   - Gán phân tầng trực tiếp vào tập `val` (không đưa vào `train`).

3. **Giai đoạn 3 (Thực thi Grid Search & Chọn Operating Point):**
   - Khi và chỉ khi tập `val` đạt đủ $\ge 5,0$ giờ đã xác thực và Ground Truth dương tính đồng thuận $100\%$, mới mở khóa cờ chạy `scripts/run_week6_ablation.py`.
   - Tuyệt đối nghiêm cấm hạ thấp tiêu chuẩn hoặc gian lận số liệu để chạy trước thời hạn.

