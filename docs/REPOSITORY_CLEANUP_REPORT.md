# Repository cleanup — báo cáo 13/11/2026

BRANCH: `fix-4`

HEAD: `fe6e22998adb1269ec4b0be081fa1815ea94714a`

WORKTREE BEFORE: sạch, không có thay đổi staged/unstaged.

WORKTREE AFTER: `.gitignore` sửa nhưng chưa stage;66 generated files được staged
delete bằng git rm --cached, bản local còn nguyên SHA. Docs/manifest/restore tool/
test mới chưa stage. Không commit/push/force-push/rewrite/gc/prune/reset/clean.

## INVENTORY

| Số đo | Trước | Sau |
|---|---:|---:|
| Tracked files trong index | 29,861 | 29,795 |
| Bytes file tracked trên working tree | 1,993,331,057 | 1,972,306,366 |
| Logical blob bytes trong index | 1,992,035,341 | 1,971,007,185 |
| Working tree bytes, không gồm .git | 27,994,077,210 | 28,065,279,505 |
| .git directory bytes | 2,048,242,487 | 2,048,233,834 |
| Untracked/ignored bytes trên working tree | 26,000,746,153 | 26,092,973,139 |
| Regular files, gồm .git | 73,678 | 73,642 |

Đây là logical file lengths tại lúc kiểm kê, không phải allocated disk clusters.
Các snapshot được thu theo lượt scan, gồm ignored files; không đi theo reparse point.
File báo cáo/kiểm kê có thể phát sinh sau thời điểm snapshot. Không coi working tree
là Git history. Blob bytes trong index khác bytes trên ổ vì line-ending normalization.

Git count-objects trước/sau đều3464 loose objects,701.13MiB;4 packs,1.21GiB.
History/object storage **không giảm**. Chênh lệch .git nhỏ thuộc index/metadata;
không có rewrite/gc. Working tree tăng 71,202,295 bytes vì archive, inventory và kiểm
chứng local; không báo đây là giảm dung lượng ổ đĩa.

Top100 tracked/untracked, bytes và file count từng thư mục, toàn bộ path inventory
nằm trong `.tools/repository_cleanup_20261113/inventory_before.json`,
`inventory_after.json`, `files_before.jsonl`, `files_after.jsonl`.
Không thêm các inventory hàng chục nghìn dòng này vào Git.

Duplicates: 1127 nhóm cùng SHA256, hash 1,258,913,622 bytes
trong các nhóm trùng kích thước; không tính empty files hay Git objects, không xóa
theo hash trùng. Danh sách ở `duplicates_sha256.json` trong thư mục local trên.
Phân loại toàn bộ file baseline: {'A': 4190, 'D': 69124, 'C': 82, 'B': 282}; đầy đủ trong `classification.jsonl`.
Nhóm D gồm environment/tools/Git internals nên không phải toàn bộ đều là ứng viên dọn.

## GITIGNORE

- Thay allow-all benchmark/training output bằng default-deny và allowlist basename
  summary/config/scope/provenance, structural record và curated training reports.
- Raw stdout/stderr/log/frame JSONL, video cache, preview JPG/intermediate weights
  mới bị ignore. Không có rule toàn cục *.json/*.csv/*.pt/*.md/*.png.
- Thêm bảo vệ media upload/FireSmokeLabeler video và dataset images mới, local DB,
  proxy/cache, profiling, temporary/editor backups. Source/Studio/tests/manifests giữ.
- Existing tracked video/checkpoint/evidence vẫn tracked; ignore không tự untrack.
-23 probe git check-ignore --no-index đã PASS; danh sách/rule ở gitignore_checks.json.
- Tập tracked giảm đúng66 path đã duyệt, không có source bị staged để xóa.

## CLEANUP

- Untrack66 file B:65 preview huấn luyện JPG và1 Ultralytics label cache.
- Logical blob content bỏ khỏi index: **21,028,156 bytes**. Local files vẫn nguyên.
- Danh sách/SHA/reason: `REPOSITORY_CLEANUP_UNTRACKED_FILES.csv`.
- Physical removal: 197 cache file instances, 127 path
  khác nhau.82 cache baseline bị xóa trước test;115 cache do validation tạo được
  dọn lại sau test. Tổng byte của các lần unlink: 3,723,023;
  không cộng lặp số này làm net saving. Cache baseline giải phóng1,624,177 byte.
- Danh sách từng lần xóa/phase/SHA: `REPOSITORY_CLEANUP_REMOVED_CACHE.csv`.
- Chỉ xóa project __pycache__/pytest cache đã kiểm tra untracked, references,
  lock bằng exclusive open, path containment và không symlink/junction. Không đụng
  .venv/.tools/Git internals, datasets, models, checkpoints, runs hoặc private review.
- Local verified supplementary archives: regenerable_files.zip
  (16,915,612 bytes,66 members) và audit_v2_complete.zip
  (6,249,315 bytes,548 members).
  CRC và SHA sau giải nén từng member khớp nguồn; restore dry-run thật548/548 identical.
- External/durable archives: **0**. Cả hai ZIP trong ignored `.tools/` trên cùng ổ,
  không phải backup chống mất ổ. Không xóa raw evidence và chưa untrack nhóm này.

## PROTECTED / REPRODUCIBILITY

-1,240 tracked source/config/report/benchmark/checkpoint files được kiểm SHA,
  không mismatch. Toàn547 member trong audit-v2 artifact_sha256.json vẫn khớp.
- Frozen GT SHA `8c6f30bbd057d5b5ae453feb64eb010387c2087ab9350ccea922ba04325bf7c3`; manifest và raw v1/v2 nguyên trạng.
- Pool61/63, 6.8223424028 giờ và scope
  `cc77669f0d5b4695fb7f1f043a6a446ba5ca949ecf46a7cc3f455e667cd89a78` giữ nguyên.
- Catalog thực đọc được63 VN_NEG,80 VN_POS,24 POS historical và16 missing references.
  AppTest38/38, camera dashboard integration trong full suite vẫn PASS.
- DB human review ngoài repository không bị ghi; kiểm read-only còn6 DRAFT.
  Không tạo account/submission/signoff, không sửa CSV private. Video/annotation/
  checkpoint/model/legacy source không bị xóa hay di chuyển.
- Inference artifacts, Operating Point và kết quả test lịch sử giữ nguyên.
- `studio_catalog_fix/` giữ vì còn reference báo cáo lịch sử và tests riêng.
  Backup app giữ:3 hàm khác mainline, chưa chứng minh mọi behavior độc nhất dư thừa.
- Danh sách giữ tracked: `kept_tracked_files.txt`; ưu tiên manual review media/model/
  legacy: `priority_manual_review.csv`, đều trong ignored local inventory folder.

## TESTS

| Command python -m | Kết quả | Exit |
|---|---|---:|
| compileall -q src ground_truth_studio scripts | PASS |0|
| unittest discover -s tests -p test_gt_studio_*.py -v |38/38 PASS|0|
| unittest discover -s tests -p test_video_audit_v2.py -v |23/23 PASS|0|
| unittest discover -s tests -p test_cleanup_restore.py -v |4/4 PASS|0|
| unittest discover -s tests -p test_*.py -v |276/276 PASS|0|

Không failures/errors/skips trong các lượt unittest trên. Bản kết quả public đã
bỏ absolute interpreter path: `REPOSITORY_CLEANUP_TEST_RESULTS.json`; raw logs ở
`.tools/repository_cleanup_20261113/`. Không sửa kết quả test lịch sử. Full-suite
legacy ghi báo cáo RTSP được phục hồi đúng byte trước test bằng finally.
Sau khi dọn lại cache, smoke import/catalog/checksum chạy với
PYTHONDONTWRITEBYTECODE=1. git diff --check và git diff --cached --check PASS.
Windows đã chạy thực; chưa tuyên bố Linux/macOS PASS.

## RISK / APPROVAL PROPOSAL

-416 raw audit-v2 artifacts (57,966,195 worktree bytes)
  vẫn tracked/local, gồm stdout/stderr/log và root-cause stdout JSON. Full mapping,
  archive SHA/size/member/restore method trong `artifact_storage_manifest.json`.
- Theo mục7 yêu cầu cleanup: khi chưa có nơi lưu bền vững, chỉ untrack raw khoa học
  sau xác nhận của người dùng. ZIP cùng ổ đã kiểm chứng **chưa đủ** bỏ yêu cầu này.
- Đề xuất cho ngài duyệt riêng: untrack đúng416 raw path trong manifest, giữ local
  và ZIP; hoặc giữ tracked cho đến khi có bản sao bền vững đã xác minh. Chưa thực hiện.
-79 historical inference .video_cache records giữ lại: có16 nguồn thiếu, không
  chứng minh tái tạo đầy đủ. Các raw benchmark khác, video, checkpoint và backup
  có giá trị lịch sử/không rõ giữ nguyên, không tối ưu theo tên file đơn thuần.
- Git LFS/history rewrite chỉ là phương án tương lai cần approval; chưa chạy.

## SCIENTIFIC STATUS / FINAL STATUS

FAR/CI: null. Human GT chưa hoàn tất; Operating Point PROVISIONAL; Scientific Week4
PARTIAL. Cleanup không tạo acceptance khoa học mới.

FINAL STATUS: **PARTIAL — safe cleanup và kiểm thử hoàn tất; raw migration chờ
xác nhận hoặc kho lưu bền vững**. Không có phần kỹ thuật bị lỗi/blocked.

NEXT ACTIONS:

1. Review các deletion đã staged (đều --cached), `.gitignore`, policy và manifests.
2. Xác nhận phương án416 raw files hoặc cung cấp vị trí lưu bền vững để kiểm chứng.
3. Giữ raw local; không xóa ZIP/nguồn/checksum để giảm working tree tùy tiện.
4. Hướng dẫn restore không overwrite trong `ARTIFACT_RETENTION_POLICY.md`; công cụ
   mặc định dry-run, chỉ ghi file còn thiếu khi dùng --apply.
5. Chưa commit/push; người dùng quyết định bước Git tiếp theo.
