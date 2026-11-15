# Week 5 — đối chiếu queue bbox QA

Ngày 15/11/2026. Branch `fix-4`; HEAD `ed406cca8fced883ec2a35de6a912987e9d69f2e`.
Trạng thái: **DONE cho sửa navigation/backend đã kiểm thử; PARTIAL cho nghiệm thu dữ liệu**.
Snapshot số liệu: **2026-11-15T14:42:01.531334+07:00 (Asia/Saigon)**. Ngài tiếp tục review nên số trực tiếp trong Studio có thể tăng.

## Nguyên nhân và phạm vi

129 ảnh không bị thất lạc: cả 129 có trong CSV ưu tiên và queue deduplicate, file ảnh/nhãn đúng SHA, nguồn candidate tồn tại. UI cũ chỉ có “Chỉ ảnh tôi chưa xử lý” (mặc định bật) và selectbox tên ảnh; thiếu bộ lọc ưu tiên, nhãn rỗng, video/ID, lý do queue và phân trang. Những ảnh đã duyệt còn bị ẩn khi bật Pending. Không có bằng chứng rằng cả 129 đã từng bị loại khỏi backend. Không thu được filter state trong trình duyệt của ngài; không suy diễn trạng thái phiên đó.

Catalog bbox đọc từ CSV, không cần import 129 ảnh vào bảng catalog video/event SQLite. Thiếu lineage là điều kiện training, không phải lý do để âm thầm giấu ảnh cần kiểm tra. View `Needs data investigation` hiển thị thiếu lineage/source/label và trạng thái quarantine; UI cho lưu quyết định bbox riêng, ghi rõ chưa cho training.

## Đối chiếu tập hợp đo bằng code

| Phạm vi | Số lượng |
|---|---:|
| Đường dẫn/ID frame ứng viên | 3.121 |
| SHA ảnh duy nhất | 3.101 |
| Nhóm exact duplicate | 20 nhóm, mỗi nhóm 2 đường dẫn |
| Exact duplicate qua legacy split | 0 nhóm |
| Queue thông thường | 1.904 |
| Queue ưu tiên nhãn rỗng | 129 |
| Giao hai queue theo SHA | 111 |
| Hợp queue sau dedup | 1.922 |
| Ngoài queue | 1.199 đường dẫn / 1.179 SHA |
| Ảnh ưu tiên tìm được theo ID và phân trang | 129/129 |
| Ảnh/nhãn thiếu hoặc sai hash trong 3.121 ứng viên | 0 / 0 |

20 cặp exact duplicate nằm ở VN_POS_50/51, thuộc phạm vi bị chặn; không dùng chúng để nhân đôi dữ liệu. Tập 1.199 ngoài queue gồm scope blocked/reserved chưa được chọn, không phải 1.199 ảnh tự động hợp lệ hay đã duyệt. Manifest liệt kê từng đường dẫn và lý do.

Queue có 691 TRAIN_CANDIDATE + 306 VALIDATION_CANDIDATE + 685 TRAIN_LEGACY_RESERVATION + 222 VAL_LEGACY_RESERVATION + 18 TEST_LEGACY_RESERVATION. 18 ảnh test được đưa vào kiểm tra nhãn rỗng để QA, **không chuyển sang train/validation**.

## Review thật và bảo toàn

Prompt ghi 34 ảnh; snapshot đầu thấy 38 ảnh, snapshot cuối thấy **108 phiên bản / 107 ảnh**. Cả 107 quyết định mới nhất là APPROVE, lưu bằng schema `SUBMITTED_BBOX_REVIEW`; không phải 107 ảnh được phép training. Có 74 negative attestations và 33 ảnh có bbox do người dùng duyệt. Không mặc nhiên coi 129 empty label là sai hoặc âm tính.

| Trạng thái snapshot | Toàn queue | Nhóm 129 |
|---|---:|---:|
| Quyết định APPROVE mới nhất | 107 | 107 |
| REJECT mới nhất | 0 | 0 |
| PENDING | 1.815 | 22 |

Không có persistent DRAFT trong schema bbox hiện tại; bản đang viết chưa có checksum marker không được tính như committed. Một ảnh có hai version, cả hai được giữ. SQLite backup API tạo hai snapshot nhất quán; `PRAGMA integrity_check=ok`. Toàn bộ 114 file thuộc 38 phiên bản snapshot đầu không đổi, các phiên bản snapshot cuối vẫn tồn tại. Backup/file review/raw SQLite chỉ lưu ở `.tools/week5_readiness_ed406cca/`, đã bị Git ignore. Không khôi phục đè DB.

## Cách dùng bản sửa

```powershell
python -m streamlit run ground_truth_studio/app.py --server.address 127.0.0.1 --server.port 8503
```

Đăng nhập tài khoản Studio hiện có → **BBox QA** → **Nhóm dữ liệu: Suspicious empty labels**. Chọn trạng thái `ALL` để thấy cả đã duyệt hoặc `PENDING` để tiếp tục phần còn lại. Có `Priority`, `Empty Label`, `Suspicious Negative`, tìm Image ID / Video ID / SHA / path, chọn 25/50/100 ảnh mỗi trang. Số nghi vấn lấy từ manifest; số label hiện rỗng đọc từ file thực, không hardcode 129. Tên/path, reservation, trạng thái và lý do queue hiển thị cạnh ảnh.

Chuyển trang hoặc bộ lọc không gọi submit. Mở ảnh đã xử lý dùng bbox version mới nhất của chính tài khoản, không quay về nhãn gốc. Chỉ nút phê duyệt/từ chối mới tạo version; nhãn rỗng cần negative attestation. Chưa có thao tác phê duyệt production nào do agent tạo.

## Bằng chứng và giới hạn

- [Manifest 129 ảnh và trạng thái](../runs/benchmarks/week5_domain_adaptation/readiness_20261115_ed406cca_v2/suspicious_empty_labels.json).
- [Manifest đủ 3.121 đường dẫn](../runs/benchmarks/week5_domain_adaptation/readiness_20261115_ed406cca_v2/vn_queue_reconciliation.json).
- [Kiểm tra navigation trên catalog thật](../runs/benchmarks/week5_domain_adaptation/readiness_20261115_ed406cca_v2/actual_queue_navigation_check.json).
- [Bảo toàn review](../runs/benchmarks/week5_domain_adaptation/readiness_20261115_ed406cca_v2/review_preservation.json).
- [Test logs và exit codes](../runs/benchmarks/week5_domain_adaptation/readiness_20261115_ed406cca_v2/test_results.json).

Kiểm thử AppTest kiểm tra render, tìm kiếm và giữ filter; backend kiểm cả 129 ID thật và toàn bộ trang; SQLite concurrency test dùng dữ liệu tổng hợp. Chưa tuyên bố nghiệm thu thao tác chuột/canvas end-to-end trên trình duyệt thật của ngài. Event review, frozen GT và video gốc được giữ nguyên.
