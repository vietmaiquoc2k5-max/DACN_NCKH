# Chính sách lưu giữ artifacts

Code, tests, cấu hình mẫu, manifest/split, frozen Ground Truth/checksum, phương pháp,
scope/provenance và các summary nhỏ đã duyệt tiếp tục nằm trong Git. Gitignore không
bỏ theo dõi file đã có trong index; mọi migration phải có danh sách path cụ thể.

## Phân loại và quyết định

| Nhóm | Xử lý |
|---|---|
| A: code/config/manifest/frozen GT/summary khoa học | Giữ và track; kiểm tra thông tin nhạy cảm trước khi thêm mới |
| B: preview huấn luyện và label cache có thể tái tạo, không được evidence tham chiếu | Giữ local; chỉ untrack sau kiểm tra references và backup/checksum |
| C: cache Python/pytest chưa tracked, không nằm trong provenance | Có thể xóa sau kiểm tra path, junction/symlink và file lock |
| D: dữ liệu gốc, checkpoint, human records, raw evidence, legacy/backup chưa chứng minh dư thừa | Giữ nguyên; proposal riêng và chờ quyết định |

Các path và SHA của66 file B đã xử lý nằm trong
`REPOSITORY_CLEANUP_UNTRACKED_FILES.csv` và `cleanup_regenerable_storage_manifest.json`.
Chúng vẫn tồn tại local. Không xóa file chỉ vì cùng SHA hay kích thước0.

## Benchmark và training outputs

`runs/benchmarks` mặc định ignore file mới. Chỉ các basename summary/scope/provenance/
config được định danh trong `.gitignore`, cùng structural record.json/record.sha256
và audit_records JSON, được mở lại. Không có allowlist toàn cục cho JSON/CSV/PNG.
Những output khác cần review rõ vai trò, kích thước và thông tin nhạy cảm trước
khi thêm rule cụ thể; không mở lại `!runs/benchmarks/**`.

Metadata dự kiến dưới1MiB mỗi file; Gitignore không áp ngưỡng byte, nên operator
phải kiểm tra trước khi stage. Summary có nhãn chưa xác minh không trở thành GT.
Human exports/database và dữ liệu nhận dạng giữ private; environment/provenance
chứa đường dẫn cá nhân cần bản public đã biên tập, không sửa raw evidence gốc.

Full stdout/stderr, packet/frame dumps, inference frame JSONL, runtime logs/caches
không được tự thêm cho run mới. File stderr rỗng có trong checksum vẫn là evidence.
Các file khoa học đã tracked hiện tại **được giữ tracked và giữ local** cho đến
khi migration có bảo toàn tương đương và được duyệt. Các training curve, config,
results và checkpoint đã tracked cũng được giữ; 65 preview JPG và một label cache
đã xác minh là ngoại lệ B.

`uploads/`, video của FireSmokeLabeler và raw images mới trong curated dataset được
ignore để tránh nạp thêm dữ liệu local. File dữ liệu đã tracked không bị xóa/untrack;
split manifests/labels/code vẫn giữ. Không ignore toàn bộ FireSmokeLabeler hoặc Studio.

## Archive audit v2 và phục hồi

`artifact_storage_manifest.json` ánh xạ416 raw artifacts tới SHA, size, source run,
member ZIP và vị trí local tương đối. Archive đầy đủ548 member đã được kiểm tra CRC
và SHA sau giải nén; raw source vẫn giữ nguyên. Bản ZIP trong `.tools/` nằm cùng ổ
với nguồn, **không phải backup offsite/bền vững khi ổ hỏng**. Không untrack nhóm raw
này trong cleanup hiện tại khi chưa có xác nhận của người dùng hoặc kho lưu phù hợp.

Kiểm tra bản local hoặc bản sao byte-identical trên ổ khác:

```powershell
python scripts/restore_cleanup_archive.py --archive .tools/repository_cleanup_20261113/audit_v2_complete.zip --manifest docs/artifact_storage_manifest.json
```

Mặc định dry-run. Sau kiểm tra, thêm `--apply` để phục hồi file còn thiếu; không ghi
đè file khác bytes. `--target-root` chỉ định repository đích. Công cụ kiểm tra SHA
archive, CRC/member, path traversal và mọi conflict trước khi ghi file đầu tiên.
Kiểm tra checksum không thay thế việc xác minh archive thật vẫn truy cập/đọc được.

Khôi phục preview/cache từ bản supplementary copy nếu cần:

```powershell
python scripts/restore_cleanup_archive.py --archive .tools/repository_cleanup_20261113/regenerable_files.zip --manifest docs/cleanup_regenerable_storage_manifest.json
```

Không rewrite history, force push hoặc chạy gc/prune để tuyên bố giảm dung lượng.
`git rm --cached` chỉ giảm index/commit kế tiếp; object lịch sử vẫn còn. Git LFS hoặc
history rewrite cho dataset/checkpoint chỉ là phương án riêng cần review/approval.

## Tái lập nghiên cứu

Giữ nguyên các script/docs Studio, evaluator và audit v1/v2. Audit v2 vẫn dùng
source63 và pool61, 6.8223424028 giờ, scope SHA
`cc77669f0d5b4695fb7f1f043a6a446ba5ca949ecf46a7cc3f455e667cd89a78`.
Không biến pool kỹ thuật thành FAR exposure. Frozen GT, inference artifacts,
Operating Point và kết quả kiểm thử lịch sử không được sửa để phục vụ cleanup.
