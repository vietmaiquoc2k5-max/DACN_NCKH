# Week5 — khả năng sử dụng VN_POS và D-Fire

Ngày (Asia/Saigon): 2026-11-14T01:58:33.051196+07:00. Branch `fix-4`, HEAD `6942fa567683ed9fdc3ef455188db1a74cef2c7f`. **BLOCKED cho training; đã kiểm kê và chuẩn bị candidates, không train.**

| Corpus | Đo hiện tại | Thiếu hụt |
|---|---|---|
| VN | 3.121 images/labels, train1868/val623/test630; 0 missing, 0 lỗi syntax, 0 duplicate SHA crossing trong VN | Legacy lineage không có source video SHA runtime, time từ frame/FPS; flags is_reviewed không có signoff đủ để chứng minh 10% cross-annotation |
| VN_POS nguồn | 80 file,79 strict eligible,78 SHA hợp lệ khác nhau | GT video chưa Submit; source/camera/scene chưa verified; một duplicate alias và một truncated |
| FireSmokeLabeler | 3.538 ảnh; configured train2873/val665 | Reference corpus hiện có, không tự nhận là snapshot train gốc của checkpoint |
| Legacy | 7.754 ảnh tìm thấy trong root; config train=val cùng đường dẫn7407 ảnh | Mapping0=fire/1=smoke đảo với VN/D-Fire; không trộn class IDs hoặc dùng thay D-Fire |
| D-Fire | Không tìm thấy ở các configured/discovered locations đã kiểm | Không thể audit content/scene overlap, replay hoặc mAP D-Fire; không lặp lại claim21.527 ảnh từ tài liệu cũ như số hiện có |

Đã kiểm 6 checkpoint thực bằng torch.load trên CPU, SHA không đổi, không gọi predict/train. 5 checkpoints benchmark khai báo data_dfire.yaml; root best.pt khai báo archive config ngoài máy hiện tại. Original training content-hash manifests không hiện có: **không chứng minh test chưa từng vào train**, cũng không khẳng định chắc tất cả VN_POS đã train. Chi tiết sanitized trong run checkpoint_provenance.json; nguyên đường dẫn máy cũ giữ local.

Manifest mới chỉ có 27 TRAIN_CANDIDATE,11 VALIDATION_CANDIDATE,13 INDEPENDENT_TEST_CANDIDATE blocked,29 blocked/unassigned. Không ép60/20/20 hoặc chuyển test sang train. 691 frame rows của 27 POS train candidates là con số lineage cũ, **chưa là ảnh đã được duyệt train**. Mọi val/test-reserved/blocked source cùng scene group phải nằm ngoài training cả ba chiến lược; data_vn.yaml cũ không tự áp dụng allowlist mới. Giữ nguyên dataset cũ, chuẩn bị dataset/version mới sau khi người thật duyệt groups, labels và nguồn.

Đối chiếu semantic review thật: 15 VN_NEG IDs chứa khói/uncertain có 129 ảnh đang mang label rỗng. Video-level decision không chứng minh từng ảnh đều có khói; cần review frame, **không tự đổi empty labels hoặc tái phân loại VN_POS thành NEG**. Xem `week5_legacy_semantic_gaps.json` và `vn_frame_integrity.csv` trong run.

Full fine-tune / freeze backbone / replay20% vẫn là kế hoạch. Freeze phải theo backbone đúng checkpoint và log parameter names; task này không thay parameter hoặc strategy. Replay20% là D-Fire train-only trong combined epoch (ceil(VN_train/4) image presentations), tuyệt đối không replay val/test. Corpus D-Fire chưa có nên **replay không khả thi hiện tại**. Chưa có mAP/Delta mAP/catastrophic forgetting mới.

Ưu tiên tiếp theo: review scene candidates (có false positives branding), GT/source lineage và bbox thủ công; tạo allowlist train theo manifest version được duyệt; phục hồi D-Fire từ nguồn và manifest hợp lệ; audit SHA + near/source/scene overlap cả hai corpus. Chỉ sau các bước đó mới chốt ba training runs. Nếu muốn giữ positive test cho model thích nghi miền, không cho bất kỳ frame/footage cùng group của nó vào bất kỳ strategy nào. **SPLIT_READY không đồng nghĩa TEST_INDEPENDENCE_VERIFIED.**

Tests cuối: Targeted: **75 passed**, exit 0; full unittest: **320 tests, OK**, exit 0. Cả hai là kết quả chạy sau bản sửa cuối, không phải số dự kiến. Các nhãn tổng hợp trong tests không là nhãn nghiên cứu; không tạo submission vào DB thật. [Báo cáo audit](VN_POS_PREANNOTATION_AUDIT_REPORT.md), [hướng dẫn annotation](VN_POS_ANNOTATION_START_GUIDE.md).
