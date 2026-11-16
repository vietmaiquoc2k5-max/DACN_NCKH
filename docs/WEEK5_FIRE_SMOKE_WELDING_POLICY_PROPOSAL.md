# Đề xuất ontology FIRE/SMOKE và hoạt động hàn — chưa được phê duyệt

Version `week5_ontology_proposal_v1`, ngày16/11/2026. **PROPOSED_PENDING_HUMAN_APPROVAL**. Không cấp quyền relabel hoặc thay đổi nhãn đã duyệt. Cần người phụ trách nghiên cứu/reviewer phê duyệt văn bản và checksum policy trước mọi thay đổi quy ước, sau đó review từng ảnh bị ảnh hưởng trong annotation version mới.

## Hai cấp nhãn

Giữ YOLO class0=SMOKE, class1=FIRE theo mapping đã kiểm ở corpus/checkpoint. Bbox mô tả vật thể/hiện tượng trong frame; event GT mô tả sự kiện theo thời gian và ngữ cảnh. Dự đoán FIRE một frame không đủ xác nhận cháy ngoài ý muốn. Không dùng event79VN_POS để tự tạo bbox hay suy tất cả frame có nhãn giống event.

| Quan sát | Đề xuất bbox cấp frame | Metadata riêng | Event/việc cần người thật xác nhận |
|---|---|---|---|
| FIRE | Class1 cho ngọn lửa nhìn thấy, cần người thật kiểm | visible_flame, context nếu biết | Không tự suy ra incident |
| SMOKE | Class0 cho khói thuộc mục tiêu nghiên cứu đã thống nhất | nguồn khói/context/uncertainty | Khói có mặt không tự xác nhận cháy ngoài ý muốn |
| WELDING_SPARKS | Tia/hạt sáng đơn lẻ không tự trở thành FIRE | welding_sparks, hard_negative_candidate | Xem toàn ảnh để loại trừ ngọn lửa thật; không tự gán empty/negative |
| WELDING_FUME | Aerosol do hàn không tự trở thành SMOKE của sự kiện cháy | welding_fume, hard_negative_candidate | Nếu kèm vật liệu cháy/khói thật, review riêng; không tự xóa bbox SMOKE cũ |
| CONTROLLED_FLAME | Đề xuất vẫn class1 nếu ontology frame là hình thái ngọn lửa nhìn thấy | controlled_flame, operating_context | Đây là lựa chọn ontology cần phê duyệt; không tự coi burner/torch là incident hoặc là ảnh âm tính bbox |
| ACTUAL_FIRE_INCIDENT | Không tạo bbox từ tag event | event-level uncontrolled/unintended burning, time intervals | GT sự kiện do người thật xác nhận theo temporal protocol |
| Mơ hồ/không phân biệt được | Không tự đổi class hay tạo nhãn âm tính | HUMAN_REVIEW_REQUIRED, uncertainty reason | Loại khỏi evaluation/training GT cho đến khi xử lý theo protocol |

Hard-negative tag là metadata của bài toán cảnh báo/sự kiện hoặc một candidate cần review, không tự đồng nghĩa labelYOLO rỗng. Ảnh label rỗng chỉ được chấp thuận sau negative attestation của người thật. Với controlled flame, có thể là positive của detector hình thái nhưng negative của cháy bất thường; phải giữ hai cấp này trong đánh giá FAR/event.

## D-Fire và bảo toàn quy ước

[README chính thức D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset) mô tả fire/smoke occurrence, YOLO normalized coordinates và tập hợp ảnh từ camera/web/public datasets. Phần tài liệu đã đọc không cung cấp ontology riêng đủ chi tiết cho welding sparks/fumes/controlled flame. Không suy ra rằng tác giả mặc định xếp hàn vào FIRE/SMOKE hoặc negative; không áp quy ước đề xuất này để tự relabel D-Fire.

Original D-Fire labels giữ nguyên và có checksum. Nếu nghiên cứu cần ontology hẹp hơn, tạo derived dataset version, ghi parent source/hash, quyết định mapping/annotation bằng người thật và evaluation scope mới. Không sửa trực tiếp corpus gốc hoặc so mAP qua hai ontology mà không công bố.

## Approval và impact assessment

Người phê duyệt cần quyết định rõ mục tiêu smoke (nguồn cháy/combustion, các nguồn khói hoạt động bình thường), controlled flame ở cấp bbox và tag hard-negative ở cấp event. Approval phải có version, policySHA, reviewer/personID, UTCtime và evidence/biên bản. Helper `require_policy_approval` chỉ kiểm schema; receipt thật vẫn phải được kiểm hash và thẩm quyền trước migration. Lượt này không có receipt phê duyệt policy và không có migration.

Impact report máy đọc sẽ giữ snapshot annotation hiện tại, phân bố bbox/class và danh sách text-evidence có từ khóa hàn/welding như **candidate**, không coi keyword là nhãn semantic. Nếu không có metadata welding, số ảnh thật thuộc hàn là **UNKNOWN**, không báo0. Mọi annotation cũ vẫn bất biến; người thật chỉ cần review phạm vi được xác định sau khi policy được phê duyệt và có chứng cứ ảnh/ngữ cảnh.
