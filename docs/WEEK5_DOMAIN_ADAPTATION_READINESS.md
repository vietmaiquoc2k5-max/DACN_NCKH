# Tuần 5 — readiness thực đo

STATUS: BLOCKED, không phải PASS. Audit bằng scripts/prepare_week456_acceptance.py.
VN:3121 frame, video-ID leakage0, duplicate SHA qua split0, missing images0,
missing/invalid labels0. Chưa kiểm định xong nội dung nhãn bởi người; format đúng
không chứng minh GT đúng. D-Fire theo data_dfire.yaml: train/val/test đều0 ảnh.
Không tự chuyển D-Fire validation/test sang train và không chạy split script có
thao tác di chuyển/sửa nhãn trong tác vụ này. Cần phục hồi dataset đúng nguồn,
hash/split manifest và chạy lại content-hash leakage audit giữa cả hai corpus.

Ba JSON trong configs/research: full_finetune, freeze_backbone, replay_20_percent.
Cùng checkpoint seed42 SHA f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6,
30 epochs, seeds42/123/456, imgsz640, class0 smoke/1 fire. Đây là thiết kế đề xuất,
không phải thí nghiệm đã chạy. Freeze dựa backbone model YAML thực tế: module0–9,
81 parameter tensors; danh sách tên nằm trong evidence/week5_readiness.json.
Training runner phải kiểm tra requires_grad/optimizer groups sau khi áp dụng freeze.

Replay20% được định nghĩa trên image presentations của epoch hỗn hợp, không phải
20% toàn bộ D-Fire corpus: chọn ceil(VN_train/4) ảnh D-Fire train, seed cố định,
không replacement. Hàm src.research_readiness.replay_plan kiểm tra leakage/duplicates,
trả tỷ lệ hiệu dụng sau làm tròn và BLOCKED khi thiếu dữ liệu. Không giả tạo list ảnh.
Augmentation/preprocessing và compute budget phải giữ đồng nhất; checkpoint đầu
và sau được hash; không chọn model trên test. Không tự launch training từ plan JSON.

Bảng sau training phải có strategy/seed/class, VN mAP50/mAP50:95, D-Fire mAP50/
mAP50:95, baseline, delta và CI/variation theo seeds khi phù hợp. Delta D-Fire =
after-before; số âm là forgetting. Re-evaluate video cùng phạm vi A/B và GT protocol,
FAR/Delay; không chọn best chỉ vì VN mAP tăng khi FAR/Delay suy giảm.
Toàn bộ mAP/delta/checkpoint thích nghi và selected model hiện chưa có, để null.
